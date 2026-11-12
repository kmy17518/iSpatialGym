"""Author isolated current-runtime regression groups; never claim old evidence as new."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read(p):
    return json.loads(Path(p).read_text())


def ref(p):
    p = Path(p)
    return {"path": str(p), "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def checked(r):
    assert ref(r["path"])["sha256"] == r["sha256"], "input bytes changed: " + r["path"]
    return read(r["path"])


def immutable(path, value):
    raw = (json.dumps(value, indent=2, sort_keys=True)+"\n").encode()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert path.read_bytes() == raw, "immutable record already differs"
    else:
        with path.open("xb") as stream:
            stream.write(raw)
        os.chmod(path, 0o640)
    return ref(path)


def preserved_environment(values):
    """Preserve behavior controls; regenerate execution paths and old group IDs."""
    protected = {"ISPATIALGYM_CAMPAIGN_CONFIG", "ISPATIALGYM_CAMPAIGN_GOAL_PLAN", "ISPATIALGYM_CAMPAIGN_GOAL_QUEUE",
                 "ISPATIALGYM_RUNTIME_SUITE_RUN", "ISPATIALGYM_RUNTIME_SUITE_ROW", "ISPATIALGYM_RUNTIME_SUITE_ROW_SHA256"}
    other = {"ISG_CUSTOMS", "KEEP_SCENE", "START_PER_ATTEMPT", "TRUNK_CHECK", "STILL_EXIT"}
    kept = {k:v for k,v in values.items() if k not in protected and
            (k.startswith("ISPATIALGYM_") or k in other or
             (k.startswith("ISGF_") and k not in {"ISGF_ROOT","ISGF_SCRIPTS","ISGF_GRASPS","ISGF_JB_NFS","ISGF_PREDEMO_ROOT",
               "ISGF_ASSET_CACHE","ISGF_SAMPLING_CACHE","ISGF_EXP_PATH","ISGF_GIT_SAFE_ALL","ISGF_COLLECT_EXTRA"}))}
    return kept, sorted(set(values)-set(kept))


def group_range(template, counts):
    from scripts.campaign_20261110.controller_groups import ranges, quotas
    from scripts.campaign_20261110.contracts import task_quotas
    if template.task == "object_scaling":
        options = ((.05, .10), (.10, .15), (.15, .30)); targets = next(q.standard_buckets for q in task_quotas() if q.task == template.task)
    elif template.task == "mental_rotation":
        options = ((30,45), (45,75), (75,120)); targets = next(q.standard_buckets for q in task_quotas() if q.task == template.task)
    else:
        options, targets = ranges(template), quotas(template)
    deficits = [max(0, n-int(counts.get(str(i), counts.get(i, 0)))) for i,n in enumerate(targets)]
    # Existing quota buckets determine scheduling, never a successful/easier draw.
    bucket = max(range(len(options)), key=lambda i: (deficits[i], -i))
    return options[bucket], bucket


def author(root, run_dir, catalog_path, ordinal):
    from scripts.campaign_20261110 import goal_groups as g
    from scripts.campaign_20261110.contracts import Template, ObjectRef
    from scripts.campaign_20261110.group_lane import check_reservations
    from scripts.campaign_20261110.reservation_policy import active_policy
    from scripts.campaign_20261110.infra_dispatch import prepare_intent_ledger
    root, run_dir = Path(root).resolve(), Path(run_dir).resolve()
    assert run_dir.is_relative_to(root/"runtime_validation")
    release = read(run_dir/"release.json"); catalog = read(catalog_path)
    runtime = release["frozen_runtime"]; native = Path(runtime["path"])
    assert ref(native/".snapshot_commit")["sha256"] == runtime["snapshot_sha256"]
    assert (native/".snapshot_commit").read_text().strip() == runtime["commit"]
    assert catalog["class_id"] == catalog["exact_configuration"]["class_id"]
    assert not catalog.get("missing_inputs"), "named input blockers: " + str(catalog.get("missing_inputs"))
    definition = dict(catalog["exact_configuration"])
    definition["objects"] = tuple(ObjectRef(**o) for o in definition["objects"])
    template = Template(**definition)
    assert template.split == "eval_standard"
    policy = active_policy(root); assert policy is not None
    check_reservations(template, policy[1], policy[2])
    reservation = read(run_dir/"index_reservation.json")
    assert 0 <= ordinal < 100
    index = reservation["proof"]["start"] + ordinal*3
    token = "current100_r1_" + hashlib.sha256(template.class_id.encode()).hexdigest()[:20]
    destination = root/"groups"/token
    row_path = run_dir/"rows"/(template.class_id+".json")
    if row_path.exists():
        previous = read(row_path)
        assert previous["fresh_start"] == index and previous["frozen_runtime"] == runtime
        checked(previous["campaign_config"])
        return previous
    assert not destination.exists(), "partial authoring must be diagnosed before reuse"
    source_ref = catalog["source_configuration_ref"]
    source = checked(source_ref)
    assert ref(catalog["source_environment_ref"]["path"]) == catalog["source_environment_ref"]
    original_env = g.read_env(catalog["source_environment_ref"]["path"])
    required, replaced = preserved_environment(catalog["required_environment"])
    camera = bool(catalog.get("camera_exempt", catalog.get("camera_group_exempt", template.task.startswith("camera_relocalization"))))
    source_limits = catalog["anchor_allocation"]
    budget = max(900, int(source_limits.get("sample_limit_s") or 2700)-1800)
    if original_env.get("ISPATIALGYM_CAMPAIGN_GOAL_PLAN"):
        budget = int(read(original_env["ISPATIALGYM_CAMPAIGN_GOAL_PLAN"])["budget_seconds"])
    quota = read(run_dir/"quota_observation.json")["classes"][template.class_id]
    magnitude, bucket = ((None, None) if camera else group_range(template, quota["accepted_bucket_counts"]))
    if camera:
        destination.mkdir(parents=True)
        source["purpose"] = "validation"
        source["validation_key"]["code_sha256"] = runtime["snapshot_sha256"]
        immutable(destination/"campaign_config.json", source)
        immutable(destination/"empty_eval_index.json", {"schema":"isg-instance-index-v1","n":0,"instances":[]})
        source_template = original_env["TEMPLATE"]
        config_path = destination/"campaign_config.json"
        command = [sys.executable, str(native/"scripts/trainfast/submit_fast.py"), "submit", template.class_id, "1",
                   "--flow", "fast", "--code", str(native), "--root", str(root), "--base-index", str(index),
                   "--tag", "current100_r1", "--job-prefix", "isg261110_", "--grasps", "v2only", "--v2-root", catalog["v2_root"],
                   "--alias-memory", "off", "--no-memory-aggregate", "--kcollect", "2", "--sample-limit-s", "2700", "--time-s", "8100"]
        generated = {"ISPATIALGYM_CAMPAIGN_CONFIG":str(config_path),"TEMPLATE":source_template,"EVAL_INDEX":str(destination/"empty_eval_index.json"),
                     "ISGF_COLLECT_EXTRA":"--no-goal-override","ISGF_SAMPLING_CACHE":str(root/"cache/groups"/token),
                     "ISGF_ASSET_CACHE":str(root/"asset_cache"),"ISGF_JB_NFS":str(root/"jobs"),"ISGF_PREDEMO_ROOT":str(root),
                     "KIT_TPL_NFS_ROOT":str(root/"kit_templates"),"KIT_TEX_STORE":str(root/"kit_textures"),"ISGF_EXP_PATH":str(root/"isaacsim_overlay/apps"),
                     "ISPATIALGYM_FAST_BAND_SHARE":"0","ISPATIALGYM_FAST_COLLECTOR_PREBUILD":"0"}
        for name,value in generated.items():
            required.pop(name,None)
            command += ["--extra-env",name+"="+value]
        goal_ref, count = None, 1
    else:
        anchor = g.successful_anchor(catalog["anchor_job"],campaign_root=root,ledger=catalog["anchor_ledger"],
                                    campaign_started_at=float(read(run_dir/"campaign_start.json")["campaign_started_at"]))
        anchor["goal_proposal_source"] = "approved_recorded_goal_as_input_only; fresh_current_runtime_members_required"
        g.prepare_group(anchor=anchor,template=template,campaign_id=root.name,group_id=token,output=destination,
                        code_sha256=runtime["snapshot_sha256"],physics_sha256=source["validation_key"]["physics_sha256"],
                        data_sha256=source["validation_key"]["data_sha256"],magnitude_range=magnitude,
                        budget_seconds=budget,max_members=3,reuse_anchor_goal=True)
        command = g.submission_command(destination,snapshot=native,campaign_root=root,v2_root=catalog["v2_root"],base_index=index)
        config_path = destination/"campaign_config.json"
        generated = dict(command[i+1].split("=",1) for i,v in enumerate(command) if v == "--extra-env")
        for name in generated:
            required.pop(name,None)
        goal_ref, count = ref(destination/"goal_plan.json"), 3
    # The catalog's behavior settings are audited separately. Job/runtime routing,
    # output identity and goal-plan binding always come from this new invocation.
    forbidden = {"ISG_CODE","ISGF_CODE","ISG_ROOT","ISGF_ROOT","ISGE_CODE","ISGF_JB","START","N","PYTHONPATH"}
    assert not forbidden.intersection(required), "catalog contains old execution identity"
    for name,value in required.items():
        assert isinstance(value,str)
        command += ["--extra-env",name+"="+value]
    if original_env.get("START_PER_ATTEMPT") == "0": command += ["--customs","template"]
    command[command.index("--tag")+1] = "current100_r1"
    ledger = prepare_intent_ledger(root,token)
    if "--ledger" in command: command[command.index("--ledger")+1] = str(ledger)
    else: command += ["--ledger",str(ledger)]
    command += ["--max-attempts",str(source_limits.get("max_attempts") or 15),"--extra-env","DEMO_LIMIT_S="+str(source_limits.get("demo_limit_s") or 3600)]
    row = {"schema":"isg-current-runtime-class-launch-v1","campaign_id":root.name,"run_id":release["run_id"],
           "run_started_at":release["run_started_at"],"class_id":template.class_id,"frozen_runtime":runtime,
           "release":ref(run_dir/"release.json"),"author":ref(__file__),"quota_observation":ref(run_dir/"quota_observation.json"),"catalog_row":ref(catalog_path),"campaign_config":ref(config_path),
           "group_id":None if camera else token,"goal_plan":goal_ref,"fresh_start":index,"fresh_count":count,
           "camera_exempt":camera,"owner":"/root","dispatch_token":token,"allocation_ledger":str(ledger),
           "selected_bucket":bucket,"effective_limits":{"group_budget_seconds":None if camera else budget,
               "sampling_limit_seconds":int(command[command.index("--sample-limit-s")+1]),"allocation_limit_seconds":int(command[command.index("--time-s")+1])},
           "quota_ineligible_at_authoring":bool(quota["current_eligible_standard"]>=10),
           "source_environment_replaced_keys":replaced,
           "historical_proof_used_as_new_success":False,"production_authorization":False,"created_at":time.time()}
    row_ref = immutable(row_path,row)
    command += ["--extra-env","ISPATIALGYM_RUNTIME_SUITE_RUN="+release["run_id"],
                "--extra-env","ISPATIALGYM_RUNTIME_SUITE_ROW="+str(row_path),
                "--extra-env","ISPATIALGYM_RUNTIME_SUITE_ROW_SHA256="+row_ref["sha256"]]
    immutable(run_dir/"authoring"/(template.class_id+".json"),{"row":row_ref,"command":command,"directory":str(destination),
              "status":"ready_unsubmitted","count":count,"index":index,"validation_key":read(config_path)["validation_key"]})
    return row


def main():
    p=argparse.ArgumentParser();p.add_argument("--root",type=Path,required=True);p.add_argument("--run-dir",type=Path,required=True)
    p.add_argument("--catalog-row",type=Path,required=True);p.add_argument("--ordinal",type=int,required=True)
    a=p.parse_args();print(json.dumps(author(a.root,a.run_dir,a.catalog_row,a.ordinal)))


if __name__ == "__main__":main()
