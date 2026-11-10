"""Bind ready contracts to native runtime definitions and submit fresh anchors."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
import sys

from scripts.campaign_20261110.contracts import Perturbations, ScaleState, ValidationKey, file_digest, require, write_json_atomic
from scripts.campaign_20261110.goal_groups import bind_runtime_definition, load_template_contract
from scripts.campaign_20261110.group_lane import check_reservations
from scripts.campaign_20261110.reservation_policy import active_policy


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "snapshot", "plan", "contracts", "physics", "data-identity"):
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument("--name", required=True)
    p.add_argument("--base-index", required=True, type=int)
    p.add_argument("--submit", action="store_true")
    p.add_argument("--submitter", type=Path)
    p.add_argument("--vast-gpu", choices=["RTX 4090", "RTX 5090"])
    a = p.parse_args()
    root, snapshot = a.root.resolve(), a.snapshot.resolve()
    require(snapshot.is_relative_to(root / "code") and a.name.isidentifier(), "unsafe author path")
    results = []
    active = active_policy(root)
    require(active is not None, "missing committed holdout policy")
    _, reservations, reserved_hashes = active
    for index, row in enumerate(json.loads(a.plan.read_text())):
        key = row["key"]
        result = {"class_id": key, "index": a.base_index + index * 100}
        results.append(result)
        try:
            out = root / "groups" / a.name / key
            require(out.resolve().is_relative_to(root / "groups" / a.name), "unsafe class key")
            template = load_template_contract(a.contracts, key)
            check_reservations(template, reservations, reserved_hashes)
            sources = ([Path(row["source_template"]) / "template_meta.json"] if row.get("source_template") else
                       list((Path(row["job_dir"]) / "inst").glob("*/*/template_meta.json")))
            require(len(sources) == 1, "ambiguous fresh native source template")
            source = sources[0].parent
            meta = json.loads(sources[0].read_text())
            activity, number = meta["problem"].rsplit("-", 1)
            bddl = snapshot / "ispatialgym/assets/bddl/activity_definitions" / activity / ("problem" + number + ".bddl")
            meta.update(activity_definition=str(bddl), bddl_sha256=file_digest(bddl))
            template = bind_runtime_definition(template, source, runtime_metadata=meta)
            perturb = Perturbations({obj.instance: ScaleState(obj.default_scale, obj.default_scale, "fixed") for obj in template.objects})
            validation = ValidationKey(template.fingerprint, file_digest(snapshot / ".snapshot_commit"), file_digest(a.physics),
                                       file_digest(a.data_identity), perturb.scale_fingerprint)
            config = {"campaign_id": "campaign_20261110", "split": template.split, "purpose": "validation",
                      "template": asdict(template), "perturbations": asdict(perturb), "validation_key": asdict(validation)}
            if out.exists():
                require(json.loads((out / "campaign_config.json").read_text()) == json.loads(json.dumps(config)), "existing anchor config differs")
            else:
                out.mkdir(parents=True)
                write_json_atomic(out / "campaign_config.json", config)
                write_json_atomic(out / "source.json", {"template_dir": str(source), "anchor_allocation": row, "anchor_is_final_member": False})
                write_json_atomic(out / "empty_eval_index.json", {"schema": "isg-instance-index-v1", "n": 0, "instances": []})
            submitter = a.submitter or snapshot / "scripts/campaign_20261110/infra_submit_validation.py"
            require(submitter.resolve().is_relative_to(root), "submit helper outside campaign")
            cmd = [sys.executable, str(submitter),
                   "--root", str(root), "--group", str(out), "--snapshot", str(snapshot), "--key", key,
                   "--tag", "annotated_anchor", "--v2-root", "/vision/u/minyeong/projects/isg_home/oracle_fix/v2maps_int2",
                   "--index", str(result["index"])]
            if a.vast_gpu:
                cmd += ["--vast-gpu", a.vast_gpu]
            result.update(directory=str(out), command=cmd)
            if (out / "submit.txt").exists():
                result["status"] = "previous_submission_requires_ledger_reconciliation"
            else:
                call = subprocess.run(cmd + ([] if a.submit else ["--dry-run"]), text=True, capture_output=True, timeout=180)
                require(call.returncode == 0, call.stderr[-2000:])
                result.update(status="submitted" if a.submit else "dry_run_passed", output=call.stdout.strip())
        except Exception as exc:
            result.update(status="blocked", error=str(exc))
        write_json_atomic(root / "manifests" / (a.name + "_results.json"), results)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
