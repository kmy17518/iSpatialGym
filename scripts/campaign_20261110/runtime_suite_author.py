"""Author isolated current-runtime regression groups; never claim old evidence as new."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from ispatialgym.sampling.campaign_bucket_policy import POLICY as BUCKET_POLICY


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
    # Native task samplers retain physical bounds and ignore these legacy
    # descriptive ranges; no bucket deficit may hold an otherwise ready class.
    return (options[0][0], options[-1][1]), 0


def recorded_opening_range(template, anchor):
    """Bind a fixed approved joint goal to its actual strict bucket."""
    from scripts.campaign_20261110.controller_groups import ranges
    assert template.task in ("configuration_matching-articulation_open_small_scale",
                             "configuration_matching-articulation_open_large_scale"), "recorded goal range requires opening"
    targets = [obj for obj in template.objects if obj.role == "target"]
    assert len(targets) == 1 and targets[0].joint
    target = targets[0]
    limits = anchor["template_meta"]["template_facts"]["limits"]
    assert len(limits) == 2 and limits[0] < limits[1]
    names = anchor["joint_names"][target.instance]
    values = anchor["planned_goal_joints"][target.instance]
    assert len(names) == len(values) and target.joint in names
    fraction = (values[names.index(target.joint)]-limits[0])/(limits[1]-limits[0])
    choices = [(bounds, i) for i, bounds in enumerate(ranges(template)) if bounds[0] <= fraction <= bounds[1]]
    assert choices, "recorded approved opening goal lies outside every strict bucket"
    return choices[0]


def recorded_disembedding_evidence(template, anchor):
    """Bind the unchanged active selection to its measured historical bucket.

    Quota deficits cannot change occupied area when reusing a recorded pool and
    goal. Require both the native geometry profile and the hash-bound successful
    source to agree; this does not authorize changing the pool or any threshold.
    """
    from scripts.campaign_20261110.contracts import digest, require
    from scripts.campaign_20261110.controller_groups import ranges
    from scripts.campaign_20261110.disembedding_coverage import measure
    require(template.task == "disembedding" and template.split == "eval_standard",
            "recorded coverage requires standard disembedding")
    for name in ("tro", "summary"):
        require(checked(anchor["sources"][name]) == anchor[name], "anchor " + name + " changed")
    metadata_path = Path(anchor["template_dir"])/"template_meta.json"
    require(read(metadata_path) == anchor["template_meta"], "native coverage metadata changed")
    profile = anchor["template_meta"].get("template_facts", {}).get("campaign_coverage_profile", {})
    coverage, bucket, objects = measure(template, profile)
    bounds = tuple(ranges(template)[bucket])
    campaign = anchor["tro"]["ispatialgym"]["campaign"]
    require(campaign["class_id"] == template.class_id and campaign["split"] == template.split,
            "recorded coverage source class/split differs")
    summary_group = anchor["summary"].get("campaign_goal_group", {})
    runtime_group = campaign.get("runtime", {}).get("goal_group", {})
    for group in (summary_group, runtime_group):
        actual = group.get("actual_crowdedness")
        require(isinstance(actual, (int, float)) and math.isfinite(actual) and 0 < actual <= 1
                and math.isclose(actual, coverage, abs_tol=1e-6),
                "recorded actual crowdedness differs from measured native geometry")
    require(summary_group.get("group_id") and summary_group.get("plan_sha256") and
            all(summary_group[k] == runtime_group.get(k) for k in ("group_id", "plan_sha256")),
            "recorded source group identities differ")
    return {"schema": "isg-current-runtime-recorded-disembedding-range-v1", "bucket": bucket,
            "strict_range": list(bounds), "coverage": coverage, "active_objects_sha256": objects,
            "native_template_metadata": ref(metadata_path), "native_profile_sha256": digest(profile),
            "source_tro": anchor["sources"]["tro"], "source_summary": anchor["sources"]["summary"],
            "historical_group_id": summary_group["group_id"], "historical_plan_sha256": summary_group["plan_sha256"],
            "historical_actual_crowdedness": summary_group["actual_crowdedness"],
            "pool_geometry_goal_changed": False, "bucket_threshold_changed": False}


def recorded_disembedding_range(template, anchor):
    evidence = recorded_disembedding_evidence(template, anchor)
    return tuple(evidence["strict_range"]), evidence["bucket"]


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
    recorded_range_evidence = None
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
        if template.task in ("configuration_matching-articulation_open_small_scale",
                             "configuration_matching-articulation_open_large_scale"):
            magnitude, bucket = recorded_opening_range(template, anchor)
        elif template.task == "disembedding":
            recorded_range_evidence = recorded_disembedding_evidence(template, anchor)
            magnitude, bucket = tuple(recorded_range_evidence["strict_range"]), recorded_range_evidence["bucket"]
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
    command += ["--extra-env","MAX_ATTEMPTS="+str(source_limits.get("max_attempts") or 15),"--extra-env","DEMO_LIMIT_S="+str(source_limits.get("demo_limit_s") or 3600)]
    row = {"schema":"isg-current-runtime-class-launch-v1","campaign_id":root.name,"run_id":release["run_id"],
           "run_started_at":release["run_started_at"],"class_id":template.class_id,"frozen_runtime":runtime,
           "release":ref(run_dir/"release.json"),"author":ref(__file__),"quota_observation":ref(run_dir/"quota_observation.json"),"catalog_row":ref(catalog_path),"campaign_config":ref(config_path),
           "group_id":None if camera else token,"goal_plan":goal_ref,"fresh_start":index,"fresh_count":count,
           "camera_exempt":camera,"owner":"/root","dispatch_token":token,"allocation_ledger":str(ledger),
           "bucket_policy": BUCKET_POLICY,
           "selected_bucket":bucket,"effective_limits":{"group_budget_seconds":None if camera else budget,
               "sampling_limit_seconds":int(command[command.index("--sample-limit-s")+1]),"allocation_limit_seconds":int(command[command.index("--time-s")+1])},
           "quota_ineligible_at_authoring":bool(quota["current_eligible_standard"]>=10),
           "source_environment_replaced_keys":replaced,
           "historical_proof_used_as_new_success":False,"production_authorization":False,"created_at":time.time()}
    if recorded_range_evidence is not None:
        row["recorded_range_evidence"] = recorded_range_evidence
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
