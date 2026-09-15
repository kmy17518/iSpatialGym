"""``multiply_isg``: one template -> N instances (spec §9).

Example::

    OMNIGIBSON_HEADLESS=1 python -m ispatialgym.sampling.multiply_isg --template <out>/<alias>/<template_id> --instances 5 \
        [--path-buckets 3:10,5:10,10:5] [--seed 1] [--customs task_custom_lists.json]

Reads ``template_meta.json`` (alias, scene, model choice, customs path), rebuilds the world of the template (same rooms
and scene variant), runs the sampler's ``setup()`` (template facts, cached) and ``run_instances(N)`` continuing the
instance numbering after the existing instances (instance 0 exists from ``sample_isg``). Per instance the per-instance
parts of the rules run (draws, exact IK, transitions, paths) with the customs constraints as priors; camera
relocalization fills the path buckets (``--path-buckets max_m:n,...`` overrides ``generation.path_buckets``); every
success appends constraints (written back to the customs source unless ``--no-write-back``); rejections are counted per
reason in ``summary.json`` (``problems.<alias>.<template>.{n_written, time_s, rejections}``) and the run ends with the
``SAMPLING_REPORT`` stdout line.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

os.environ.setdefault("OMNIGIBSON_HEADLESS", "1")
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

from ispatialgym.sampling.v2 import SAMPLER_VERSION  # noqa: E402
from ispatialgym.sampling.v2.problem import GroundedProblem, TemplateChoice, new_run_summary  # noqa: E402

log = logging.getLogger("ispatialgym.sampling.v2")


def parse_path_buckets(text: Optional[str]) -> Optional[List[Dict[str, Any]]]:
    """``"3:10,5:10,10:5"`` -> ``[{"max_m": 3, "n": 10}, ...]`` (§2.3 ``generation.path_buckets``)."""
    if not text:
        return None
    out = []
    for part in text.split(","):
        m, n = part.split(":")
        out.append({"max_m": float(m), "n": int(n)})
    return out


def load_template(template_dir: str) -> Dict[str, Any]:
    with open(os.path.join(template_dir, "template_meta.json")) as f:
        return json.load(f)


def problem_for_template(meta: Dict[str, Any], customs: Optional[str], bddl_root: Optional[str]) -> GroundedProblem:
    """The grounded problem of the template: from the customs file (``--customs``, else the one recorded in the meta,
    else the copy next to the alias directory)."""
    alias = meta["alias"]
    template_dir = meta.get("_dir")
    candidates = [customs, meta.get("customs")]
    if template_dir:
        candidates.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(template_dir))), "task_custom_lists.json"))
    for c in candidates:
        if c and os.path.exists(c):
            try:
                return GroundedProblem.load(alias, c, bddl_root)
            except Exception as e:  # the copy may lack the grounded BDDL (v1 facade run): fall through
                log.warning("customs %s: %s", c, e)
                with open(c) as f:
                    data = json.load(f)
                if alias in data:
                    gp = GroundedProblem(alias, data[alias], customs_path=c, bddl_root=bddl_root, subtask=meta.get("subtask"))
                    return gp
    raise SystemExit(f"no customs entry for alias {alias!r} (tried {candidates})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--template", required=True, help="<out>/<alias>/<template_id> directory")
    parser.add_argument("--instances", type=int, required=True, help="instances to add (instance 0 exists)")
    parser.add_argument("--path-buckets", default=None, help="camera relocalization: max_m:n,... (overrides generation.path_buckets)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--customs", default=None)
    parser.add_argument("--bddl-root", default=None)
    parser.add_argument("--max-attempts", type=int, default=30)
    parser.add_argument("--no-write-back", action="store_true")
    parser.add_argument("--log-level", default=os.environ.get("ISPATIALGYM_LOG_LEVEL", "INFO"))
    args = parser.parse_args(argv)

    logging.basicConfig(level=args.log_level, format="[%(levelname)s %(name)s] %(message)s", stream=sys.stdout, force=True)
    template_dir = os.path.abspath(args.template)
    meta = load_template(template_dir)
    meta["_dir"] = template_dir
    out_root = os.path.dirname(os.path.dirname(template_dir))
    gp = problem_for_template(meta, args.customs, args.bddl_root)
    buckets = parse_path_buckets(args.path_buckets)
    if buckets is not None:
        gp.generation["path_buckets"] = buckets
    choice = TemplateChoice.from_json(meta["choice"])
    if choice.template_id != os.path.basename(template_dir):
        log.warning("template id of the meta (%s) differs from the directory (%s)", choice.template_id, os.path.basename(template_dir))
    report = new_run_summary(SAMPLER_VERSION, gp.customs_path, out_root, vars(args))
    import omnigibson as og

    from ispatialgym.sampling.sample_isg import WorldPool, run_template

    pool = WorldPool(out_root)
    try:
        world, kit = pool.get(gp, choice.scene)
        existing = len([f for f in os.listdir(os.path.join(template_dir, "instances")) if f.endswith("-tro_state.json")]) if os.path.isdir(
            os.path.join(template_dir, "instances")) else 0
        entry = run_template(gp, choice, world, kit, out_root, args.instances, args.seed * 7919 + existing, args.max_attempts, meta.get("variant"),
                             None, start_index=existing)
        entry["start_index"] = existing
        report["problems"].setdefault(gp.alias, {})[choice.template_id] = entry
        pool.cleanup_template()
        if not args.no_write_back and gp.customs_path:
            gp.save_customs()
            copy = os.path.join(out_root, "task_custom_lists.json")
            if os.path.abspath(copy) != os.path.abspath(gp.customs_path):
                shutil.copy(gp.customs_path, copy)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        with open(os.path.join(template_dir, f"multiply_summary_{int(time.time())}.json"), "w") as f:
            json.dump(report, f, indent=2, default=str)
        print("SAMPLING_REPORT " + json.dumps({"sampler_version": SAMPLER_VERSION, "problems": {a: {t: {k: v for k, v in e.items() if k in ('n_written', 'time_s', 'rejections', 'error')}
                                                                                                        for t, e in ts.items()} for a, ts in report["problems"].items()}}, default=str), flush=True)
        og.shutdown()


if __name__ == "__main__":
    main()
