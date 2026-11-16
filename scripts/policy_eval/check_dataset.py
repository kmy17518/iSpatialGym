#!/usr/bin/env python3
"""Static check of an eval dataset against the evaluator, without the simulator: every instance resolves (run_isg.resolve_instance's
dataset path, ispatialgym.eval.instances.template_files) to its own index-record template + template meta, and to a BDDL problem
file whose sha256 is the instance file's ``bddl_sha256`` and whose variant is the instance file's ``task_variant``.

    PYTHONPATH=<code>:<code>/Behavior-1K/bddl3 ISPATIALGYM_EVAL_BDDL_OVERLAY=<overlay> check_dataset.py --instances <root>
"""
import argparse
import collections
import hashlib
import json
import os
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--instances", required=True)
    a = ap.parse_args()
    from ispatialgym.activity.problem import canonical_variant, parse_problem_file
    from ispatialgym.eval import instances as INST

    bad, n = collections.Counter(), 0
    examples = {}
    for rec in INST.load_index(a.instances):
        n += 1
        tro_path = rec.path("tro_state")
        files = INST.template_files(tro_path)
        tro = json.load(open(tro_path))["ispatialgym"]
        checks = {}
        checks["template_is_record"] = files is not None and os.path.abspath(files["template"]) == os.path.abspath(rec.path("template"))
        meta = json.load(open(files["template_meta"])) if files else {}
        act, did = (meta.get("problem") or "-0").rsplit("-", 1)
        bddl = os.path.join(files["activity_definitions"], act, f"problem{did}.bddl") if files else ""
        checks["bddl_exists"] = os.path.exists(bddl)
        checks["bddl_sha"] = checks["bddl_exists"] and hashlib.sha256(open(bddl, "rb").read()).hexdigest() == tro.get("bddl_sha256")
        if checks["bddl_exists"]:
            prob = parse_problem_file(bddl, act, int(did))
            checks["variant"] = canonical_variant(prob.variant) == canonical_variant(tro.get("task_variant"))
        checks["tro_did"] = int(tro["activity_definition_id"]) == int(did)
        for k, ok in checks.items():
            if not ok:
                bad[k] += 1
                examples.setdefault(k, rec.instance_id)
    print(json.dumps({"instances": n, "failures": bad, "examples": examples}, indent=1))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
