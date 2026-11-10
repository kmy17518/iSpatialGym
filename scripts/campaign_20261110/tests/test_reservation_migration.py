from dataclasses import asdict
import json
from pathlib import Path
import sqlite3

import pytest

from scripts.campaign_20261110 import contracts as c,reservation_migration as m,reservation_policy as p
from scripts.campaign_20261110.tests.test_group_lane import candidate,claim,reservations


def setup(root,added="book/new"):
    f,review=candidate(root,0)
    claim(root,f,review)
    c.write_json_atomic(root/"groups/g0_group/campaign_config.json",{"template":asdict(f.template),"validation_key":review["validation_key"]})
    old=reservations();new={**old,"model_holdout_models":[added],"train_exclude_models":[added]}
    hashes={"schema":"campaign-reserved-asset-hashes-v1","reserved_model_keys":[],"ready_assets":{},
        "pending_unready_model_keys":[],"reserved_asset_hashes":[]}
    paths={key:root/"manifests"/(key+".json") for key in ("old_reservations","new_reservations","old_asset_hashes","new_asset_hashes","source_audit")}
    for key,data in (("old_reservations",old),("new_reservations",new)):
        c.write_json_atomic(paths[key],data)
    c.write_json_atomic(paths["old_asset_hashes"],{**hashes,"reservation_sha256":c.file_digest(paths["old_reservations"])})
    sha=c.digest("new-ready-asset")
    c.write_json_atomic(paths["new_asset_hashes"],{**hashes,"reservation_sha256":c.file_digest(paths["new_reservations"]),
        "reserved_model_keys":[added],"ready_assets":{added:{"asset_sha256":sha,"ready":True,"readiness_source":"fixture"}},"reserved_asset_hashes":[sha]})
    c.write_json_atomic(paths["source_audit"],{"old_reservation_sha256":c.file_digest(paths["old_reservations"]),
        "new_reservation_sha256":c.file_digest(paths["new_reservations"]),"added_model_keys":[added],"added_asset_sha256":{added:sha},"reason":"replace unready model; old identity remains reserved"})
    camera=root/"dataset/camera_quota.json"
    c.write_json_atomic(camera,{"task":"camera_relocalization-standard","campaign_id":"campaign","accepted":{}})
    return dict(root=root,migration_id="reserve-new",state_path=root/"dataset/quotas.sqlite3",**paths),old,new,camera


def test_migration_preserves_accepted_data_and_freezes_active_revision(tmp_path):
    args,old,new,camera=setup(tmp_path)
    before=json.loads((tmp_path/"dataset/groups/g0_group.json").read_text())
    result=m.migrate(**args)
    assert result["status"]=="committed" and result["accepted_audit"]["accepted_instances"]==2
    assert json.loads((tmp_path/"dataset/groups/g0_group.json").read_text())==before
    assert m.read(camera)["reservation_policy_sha256"]==p.policy_digest(new)
    with sqlite3.connect(args["state_path"]) as db:
        assert db.execute("SELECT value FROM metadata WHERE key='reservations'").fetchone()[0]==p.policy_digest(new)
    assert p.active_policy(tmp_path)[1]==new
    with pytest.raises(c.ContractError,match="superseded"):
        p.require_active_policy(tmp_path,old)
    assert m.migrate(**args)==result


def test_an_already_accepted_model_cannot_move_into_holdout(tmp_path):
    args,_,_,camera=setup(tmp_path,added="apple/abc")
    before={path:c.file_digest(path) for path in (args["state_path"],camera)}
    with pytest.raises(c.ContractError,match="held-out model in held-in"):
        m.migrate(**args)
    assert {path:c.file_digest(path) for path in before}==before
    assert not (tmp_path/"dataset/reservation_migration_pending.json").exists()


def test_interrupted_publication_blocks_claims_and_recovers_idempotently(tmp_path,monkeypatch):
    args,_,new,camera=setup(tmp_path)
    original=m.durable_atomic_copy
    def fail_camera(source,destination):
        if Path(destination)==camera:raise OSError("simulated publication interruption")
        return original(source,destination)
    monkeypatch.setattr(m,"durable_atomic_copy",fail_camera)
    with pytest.raises(OSError,match="interruption"):
        m.migrate(**args)
    with pytest.raises(c.ContractError,match="migration is pending"):
        p.require_active_policy(tmp_path,new)
    monkeypatch.setattr(m,"durable_atomic_copy",original)
    result=m.migrate(**args)
    assert result["status"]=="committed" and not (tmp_path/"dataset/reservation_migration_pending.json").exists()


def test_monotonic_policy_rejects_removal_and_reserved_byte_change():
    old=reservations();old["model_holdout_models"]=["old/a"];old["train_exclude_models"]=["old/a"]
    new={**old,"model_holdout_models":["new/b"],"train_exclude_models":["new/b"]}
    with pytest.raises(c.ContractError,match="removed"):
        m.monotonic_extension(old,new,{}, {})
