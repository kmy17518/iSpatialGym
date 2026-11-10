"""Audit and atomically publish a monotonic global model-reservation extension."""
from contextlib import ExitStack, closing
from dataclasses import asdict
import json
from pathlib import Path
import re
import shutil
import sqlite3
import tempfile
import time

from scripts.campaign_20261110.contracts import ObjectRef, Template, canonical_json, digest, file_digest, require, write_json_atomic
from scripts.campaign_20261110.group_lane import check_reservations
from scripts.campaign_20261110.reservation_policy import policy_digest
from scripts.campaign_20261110.storage import atomic_directory_lock, durable_atomic_copy, local_sqlite_transaction

CAMERA_STATES=("camera_quota.json","camera_obstructed_quota.json","camera_category_quota.json")


def read(path):
    return json.loads(Path(path).read_text())


def monotonic_extension(old,new,old_hashes,new_hashes):
    """A reservation can grow; existing models, category rules and rooms stay fixed."""
    additions={}
    for key in ("model_holdout_models","category_holdout_models","train_exclude_models"):
        before,after=set(old[key]),set(new[key])
        require(before <= after,f"reservation migration removed {key}")
        additions[key]=sorted(after-before)
    for key in ("category_holdout_categories","category_holdout_rooms","pending_custom_hard_reservations_not_ready"):
        require(digest(old[key])==digest(new[key]),f"reservation migration changed fixed {key}")
    if "train_exclude_categories" in old:
        require(set(old["train_exclude_categories"])==set(new["train_exclude_categories"]),"category exclusions changed")
    require(set(new["model_holdout_models"]).isdisjoint(new["category_holdout_models"]),"holdout pools overlap")
    added_held=set(additions["model_holdout_models"])|set(additions["category_holdout_models"])
    require(added_held and set(additions["train_exclude_models"])==added_held,"extension must reserve every added heldout model and nothing else")
    for hashes,reservations in ((old_hashes,old),(new_hashes,new)):
        require(hashes.get("schema")=="campaign-reserved-asset-hashes-v1","invalid asset audit")
        require(set(hashes["reserved_model_keys"])==set(reservations["train_exclude_models"]),"asset audit model set differs")
        require(set(hashes["ready_assets"])|set(hashes["pending_unready_model_keys"])==set(hashes["reserved_model_keys"]),"asset audit is incomplete")
        require(set(hashes["reserved_asset_hashes"])=={v["asset_sha256"] for v in hashes["ready_assets"].values()},"asset hashes incomplete")
        require(all(row.get("ready") is True for row in hashes["ready_assets"].values()),"unready reserved-asset entry")
    require(set(old_hashes["ready_assets"]) <= set(new_hashes["ready_assets"]),"ready reservations became unresolved")
    for key,row in old_hashes["ready_assets"].items():
        require(row["asset_sha256"]==new_hashes["ready_assets"][key]["asset_sha256"],"existing reserved asset bytes changed")
    require(added_held <= set(new_hashes["ready_assets"]),"new reserved model is not ready")
    return additions


def contract(raw):
    raw=dict(raw);raw.pop("fingerprint",None)
    raw["objects"]=tuple(ObjectRef(**obj) for obj in raw["objects"])
    return Template(**raw)


def config_contracts(root,needed):
    result={}
    for area in ("groups","validation_inputs","production_inputs"):
        for path in (root/area).glob("*/campaign_config.json"):
            raw=read(path)
            if "template" not in raw:
                continue
            if digest(raw["template"]) not in needed:
                continue
            template=contract(raw["template"])
            require(template.fingerprint==raw["validation_key"]["template_sha256"],"candidate config fingerprint changed")
            result[template.fingerprint]=template
    return result


def audit_records(records,templates,new,new_hashes):
    checked,seen={},set()
    for record in records:
        iid=record["instance_id"]
        require(iid not in seen,"instance appears in multiple authoritative states")
        seen.add(iid)
        key=record["validation_key"]["template_sha256"]
        template=contract(record["contract"]) if record.get("contract") else templates.get(key)
        require(template is not None and template.fingerprint==key,"accepted record lacks fingerprint-bound ready contract")
        require(record["split"]==template.split or (template.split=="eval_standard" and record["split"]=="train"),"accepted split contradicts contract")
        check_reservations(template,new,new_hashes)
        checked[key]={"template_sha256":key,"task":template.task,"class_id":template.class_id,
            "model_keys":sorted(obj.model_key for obj in template.objects),
            "asset_sha256":sorted(obj.asset_sha256 for obj in template.objects),"contract":asdict(template)}
    return {"accepted_instances":len(seen),"accepted_ids_sha256":digest(sorted(seen)),"templates":list(checked.values())}


def _durable_json(path,data):
    with tempfile.TemporaryDirectory(prefix="isg-reservation-json-",dir="/tmp") as tmp:
        local=Path(tmp)/"record.json";write_json_atomic(local,data);durable_atomic_copy(local,path)


def migrate(*,root,migration_id,old_reservations,new_reservations,old_asset_hashes,new_asset_hashes,source_audit,state_path=None):
    root=Path(root).resolve()
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*",migration_id),"unsafe migration ID")
    paths=[Path(p).resolve() for p in (old_reservations,new_reservations,old_asset_hashes,new_asset_hashes,source_audit)]
    require(all(p.is_relative_to(root) and p.is_file() for p in paths),"migration sources must be immutable campaign files")
    old,new,old_hashes,new_hashes=map(read,paths[:4])
    require(old_hashes["reservation_sha256"]==file_digest(paths[0]) and new_hashes["reservation_sha256"]==file_digest(paths[1]),"reservation source bytes differ from asset audit")
    additions=monotonic_extension(old,new,old_hashes,new_hashes)
    source=read(paths[4])
    added=set(additions["model_holdout_models"])|set(additions["category_holdout_models"])
    require(source.get("old_reservation_sha256")==file_digest(paths[0]) and source.get("new_reservation_sha256")==file_digest(paths[1]),
            "extension source audit does not bind both reservation revisions")
    require(set(source.get("added_model_keys",[]))==added and source.get("reason"),"extension source audit differs from added models")
    require(source.get("added_asset_sha256")=={key:new_hashes["ready_assets"][key]["asset_sha256"] for key in sorted(added)},
            "extension source audit differs from new ready asset bytes")
    old_digest,new_digest=policy_digest(old),policy_digest(new)
    state_path=Path(state_path or root/"dataset/allocations.sqlite").resolve()
    require(state_path.is_relative_to(root) and state_path.is_file(),"missing campaign allocation DB")
    camera_paths=[root/"dataset"/name for name in CAMERA_STATES]
    pending=root/"dataset/reservation_migration_pending.json"
    journal=root/"manifests/reservation_migrations"/(migration_id+".json")
    source_refs={name:{"path":str(path),"sha256":file_digest(path)} for name,path in zip(
        ("old_reservations","new_reservations","old_asset_hashes","new_asset_hashes","source_audit"),paths)}
    identity={"migration_id":migration_id,"old_policy_sha256":old_digest,"new_policy_sha256":new_digest,"sources":source_refs}
    with ExitStack() as locks:
        locks.enter_context(atomic_directory_lock(state_path.with_suffix(".lockdir")))
        for path in sorted(camera_paths):
            locks.enter_context(atomic_directory_lock(path.with_suffix(path.suffix+".lockdir")))
        previous=read(journal) if journal.exists() else None
        if previous is not None:
            require(all(previous.get(k)==v for k,v in identity.items()),"migration identity changed")
        if pending.exists():
            require(read(pending)==identity,"another reservation migration is pending")
        # Stage a read-only SQLite copy: the shared filesystem never hosts a SQLite connection.
        with tempfile.TemporaryDirectory(prefix="isg-reservation-audit-",dir="/tmp") as tmp:
            local=Path(tmp)/"state.sqlite";shutil.copyfile(state_path,local)
            with closing(sqlite3.connect(local)) as db:
                current=db.execute("SELECT value FROM metadata WHERE key='reservations'").fetchone()
                require(current is not None and current[0] in (old_digest,new_digest),"allocation DB has another reservation revision")
                records=[json.loads(row[0]) for row in db.execute("SELECT record FROM members")]
        camera_states={}
        for path in camera_paths:
            if not path.exists():continue
            state=read(path)
            value=state.get("policy",{}).get("reservations_sha256",state.get("reservation_policy_sha256"))
            require(value is None or value in (old_digest,new_digest),"camera state has another reservation revision")
            camera_states[path]=state
            records.extend(state["accepted"].values())
        needed={record["validation_key"]["template_sha256"] for record in records if not record.get("contract")}
        audit=audit_records(records,config_contracts(root,needed),new,new_hashes)
        audit["source_audit"]=source_refs["source_audit"]
        if previous is not None and previous.get("status")=="committed":
            require(current[0]==new_digest,"committed migration DB reverted")
            pointer=read(root/"manifests/active_reservations.json")
            require(pointer["policy_sha256"]==new_digest and pointer["migration_id"]==migration_id,"active migration pointer differs")
            require(all(state.get("reservation_policy_sha256")==new_digest for state in camera_states.values()),"committed camera policy reverted")
            if pending.exists():pending.unlink()
            return previous
        before={str(p):file_digest(p) for p in [state_path,*camera_states]}
        prepared={"schema":"isg-reservation-migration-v1",**identity,"status":"prepared","prepared_at":time.time(),
            "additions":additions,"accepted_audit":audit,"before_state_sha256":before}
        if previous is not None:
            prepared["before_state_sha256"]=previous["before_state_sha256"]
            prepared["prepared_at"]=previous["prepared_at"]
        _durable_json(pending,identity)
        _durable_json(journal,prepared)
        with local_sqlite_transaction(state_path) as local:
            with closing(sqlite3.connect(local)) as db:
                with db:
                    db.execute("UPDATE metadata SET value=? WHERE key='reservations'",(new_digest,))
                    db.execute("INSERT OR REPLACE INTO metadata VALUES (?,?)",("reservation_migration:"+migration_id,canonical_json(identity)))
        for path,state in camera_states.items():
            if "policy" in state:state["policy"]["reservations_sha256"]=new_digest
            state["reservation_policy_sha256"]=new_digest
            state.setdefault("reservation_migrations",{})[migration_id]=identity
            _durable_json(path,state)
        pointer={"schema":"isg-active-reservations-v1","migration_id":migration_id,"policy_sha256":new_digest,
            "reservations":source_refs["new_reservations"],"asset_hashes":source_refs["new_asset_hashes"],"migration_journal":str(journal)}
        _durable_json(root/"manifests/active_reservations.json",pointer)
        committed={**prepared,"status":"committed","committed_at":time.time(),
            "after_state_sha256":{str(p):file_digest(p) for p in [state_path,*camera_states]}}
        _durable_json(journal,committed)
        pending.unlink()
        return read(journal)
