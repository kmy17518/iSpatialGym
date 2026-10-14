"""Prepare (and optionally submit) one harness episode per eval instance.

    python -m ispatialgym.harness.run --instances eval85-counted --out <root> --model openai/gpt-6-astra \
        [--model-option effort=medium ...] [--set max_calls=170 ...] [--perturb negate_xyz=true ...] [--only <substring>] \
        [--limit N] [--submit [--sbatch-args "..."]]
    python -m ispatialgym.harness.run --instances eval85-counted --out <root> --rerun-unscored

``--instances``: ``eval85-counted`` (the 84 counted eval instances: s2_3tasks + the counted rows of s3_55tasks /
iors_large ``SELECTED.tsv``) or a file with one eval run directory per line (``<run>/inst/...`` and ``<run>/demo/...``).
Each episode directory ``<root>/<tag>/<rel>/`` gets a fresh copy of the post-demo instance (``instance/``), the goal
image (the demo's ``final_rgb.png``), ``harness_config.json`` and, when run, ``run/<iid>/`` (run_isg's meta.json, videos,
``harness/`` trace). Nothing is written into the eval run directories.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Tuple

from ispatialgym.harness.config import HarnessConfig

EVAL85 = "/vision/u/minyeong/projects/isg_home/eval85/runs"
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JOB = os.path.join(REPO, "scripts", "harness", "job_episode.sh")
#: viscam11 runs the Cursor server and is excluded until a warm Kit start there is re-verified (cluster rule)
DEFAULT_SBATCH = "-p svl -A vision --gres=gpu:1 --constraint=turing|ampere|ada --exclude=visionlab-dgx1,viscam11 -c 8 --mem=24G --time=04:00:00"
S2_FAMILIES = ("camera_relocalization", "mental_rotation", "object_scaling")


def eval85_counted(root: str = EVAL85) -> List[Tuple[str, str, str]]:
    """``(tag, rel, run_dir)`` of the counted eval instances."""
    out: List[Tuple[str, str, str]] = []
    s2 = os.path.join(root, "s2_3tasks")
    for a in sorted(os.listdir(s2)):
        d = os.path.join(s2, a)
        if os.path.isdir(d) and "." not in a and a.split("-")[0] in S2_FAMILIES and glob.glob(f"{d}/demo/*/meta.json"):
            out.append(("s2_3tasks", a, d))
    for tag in ("s3_55tasks", "iors_large"):
        with open(os.path.join(root, tag, "SELECTED.tsv")) as f:
            for r in csv.DictReader(f, delimiter="\t"):
                if r["counted"] == "yes":
                    out.append((tag, r["run"], os.path.join(root, tag, r["run"])))
    return out


def instances_from_file(path: str) -> List[Tuple[str, str, str]]:
    out = []
    for line in open(path):
        d = line.strip()
        if not d or d.startswith("#"):
            continue
        d = os.path.abspath(d)
        out.append((os.path.basename(os.path.dirname(d)), os.path.basename(d), d))
    return out


def _parse_kv(items: List[str]) -> Dict[str, Any]:
    d: Dict[str, Any] = {}
    for it in items or []:
        k, sep, v = it.partition("=")
        if not sep:
            raise SystemExit(f"expected key=value, got {it!r}")
        try:
            d[k] = json.loads(v)
        except ValueError:
            d[k] = v
    return d


def prepare_episode(run_dir: str, ep_dir: str, base: Dict[str, Any], overwrite: bool = False) -> Dict[str, Any]:
    tros = glob.glob(os.path.join(run_dir, "inst", "*", "*", "instances", "*-tro_state.json"))
    if len(tros) != 1:
        raise RuntimeError(f"{run_dir}: expected one instance, found {len(tros)}")
    tro = tros[0]
    iid = os.path.basename(tro)[: -len("-tro_state.json")]
    alias = iid.split("__")[0]
    goal = os.path.join(run_dir, "demo", iid, "final_rgb.png")
    if os.path.exists(os.path.join(ep_dir, "harness_config.json")) and not overwrite:
        raise FileExistsError(f"{ep_dir} is already prepared (--overwrite to redo)")
    if os.path.exists(ep_dir) and overwrite:
        shutil.rmtree(ep_dir)
    os.makedirs(ep_dir, exist_ok=True)
    shutil.copytree(os.path.dirname(os.path.dirname(tro)), os.path.join(ep_dir, "instance"))
    cfg = HarnessConfig.from_dict({**base, "alias": alias, "goal_image": goal if os.path.exists(goal) else None})
    cfg.save(os.path.join(ep_dir, "harness_config.json"))
    with open(os.path.join(ep_dir, "source.json"), "w") as f:
        json.dump({"run_dir": run_dir, "iid": iid, "alias": alias, "tro": tro, "goal_image": cfg.goal_image}, f, indent=1)
    return {"iid": iid, "alias": alias, "ep_dir": ep_dir}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--instances", required=True, help="eval85-counted or a file of eval run directories")
    p.add_argument("--out", required=True, help="root of the episode directories")
    p.add_argument("--model", default=HarnessConfig.model)
    p.add_argument("--planner", default=HarnessConfig.planner)
    p.add_argument("--model-option", action="append", default=[], help="key=value (JSON values allowed), e.g. effort=medium")
    p.add_argument("--set", action="append", default=[], help="harness config key=value, e.g. max_calls=50")
    p.add_argument("--perturb", action="append", default=[], help="probe key=value, e.g. negate_xyz=true, mask_cameras='[\"head\"]'")
    p.add_argument("--only", action="append", default=[], help="keep instances whose tag/rel contains this substring")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--submit", action="store_true", help="sbatch one job per episode")
    p.add_argument("--rerun-unscored", action="store_true",
                   help="resubmit the prepared episodes under --out whose last run was not scored (status error: an outage, a harness bug); "
                        "their old run/ is kept as run_unscored_<time>/")
    p.add_argument("--sbatch-args", default=DEFAULT_SBATCH)
    p.add_argument("--timeout-s", type=int, default=13800, help="run_isg time limit inside the job")
    args = p.parse_args(argv)

    if args.rerun_unscored:
        return rerun_unscored(args)
    insts = eval85_counted() if args.instances == "eval85-counted" else instances_from_file(args.instances)
    if args.only:
        insts = [t for t in insts if any(s in f"{t[0]}/{t[1]}" for s in args.only)]
    if args.limit:
        insts = insts[: args.limit]
    base = {"planner": args.planner, "model": args.model, "model_options": _parse_kv(args.model_option), "perturb": _parse_kv(args.perturb)}
    base.update(_parse_kv(args.set))
    HarnessConfig.from_dict(base)  # fail fast on unknown keys
    os.makedirs(args.out, exist_ok=True)
    manifest = os.path.join(args.out, "episodes.tsv")
    new = not os.path.exists(manifest)
    with open(manifest, "a") as mf:
        if new:
            mf.write("tag\trel\tep_dir\talias\tjob\n")
        for tag, rel, run_dir in insts:
            ep_dir = os.path.join(args.out, tag, rel)
            try:
                info = prepare_episode(run_dir, ep_dir, base, overwrite=args.overwrite)
            except FileExistsError as e:
                print(f"skip: {e}", file=sys.stderr)
                continue
            job = submit(ep_dir, args) if args.submit else ""
            mf.write(f"{tag}\t{rel}\t{ep_dir}\t{info['alias']}\t{job}\n")
            print(f"{tag}/{rel}\t{info['alias']}\t{job or 'prepared'}")
    return 0


def submit(ep_dir: str, args) -> str:
    cmd = ["sbatch", "--parsable", "-J", "isg_harness", "-o", os.path.join(ep_dir, "slurm_%j.log")] + shlex.split(args.sbatch_args) + [JOB, ep_dir]
    env = dict(os.environ, ISG_HARNESS_CODE=REPO, ISG_HARNESS_TIMEOUT_S=str(args.timeout_s))
    r = subprocess.run(cmd, capture_output=True, text=True, env=env)
    return r.stdout.strip() if r.returncode == 0 else f"SUBMIT_FAILED: {r.stderr.strip()[:200]}"


def rerun_unscored(args) -> int:
    from ispatialgym.harness.summarize import run_meta

    stamp = time.strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(args.out, "episodes.tsv"), "a") as mf:
        for cfg_path in sorted(glob.glob(os.path.join(args.out, "**", "harness_config.json"), recursive=True)):
            ep_dir = os.path.dirname(cfg_path)
            if run_meta(ep_dir).get("status") != "error":
                continue
            os.rename(os.path.join(ep_dir, "run"), os.path.join(ep_dir, f"run_unscored_{stamp}"))
            job = submit(ep_dir, args)
            rel = os.path.relpath(ep_dir, args.out)
            alias = json.load(open(cfg_path)).get("alias") or ""
            mf.write(f"{os.path.dirname(rel)}\t{os.path.basename(rel)}\t{ep_dir}\t{alias}\t{job}\n")
            print(f"{rel}\t{alias}\trerun {job}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
