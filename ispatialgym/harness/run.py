"""Prepare (and optionally submit) one harness episode per eval dataset instance, named as ``run_isg --task`` names them.

    python -m ispatialgym.harness.run --task alignment_axial --instance-indices 0 1 --out <root> --model openai/gpt-6-astra \
        [--model-option effort=medium ...] [--set max_calls=170 ...] [--perturb negate_xyz=true ...] \
        [--max-steps N] [--settle-steps N] [--submit [--sbatch-args "..."]]
    python -m ispatialgym.harness.run --all --out <root> ...         # every instance of the dataset
    python -m ispatialgym.harness.run --out <root> --rerun-unscored

The instances are the eval dataset's (``ispatialgym.eval.instances``: ``Behavior-1K/datasets/ispatialgym-eval-instances``):
``--task`` (a task family, task or subtask) and ``--instance-indices`` resolve exactly as in ``run_isg --task``. Each episode
directory ``<root>/<subtask>/<instance>/`` gets ``harness_config.json`` (the instance as its subtask and index there,
``task`` / ``instance_index``) and ``source.json``, and when run ``run/<iid>/`` (run_isg's meta.json, videos, ``harness/``
trace). The job (``scripts/harness/job_episode.sh``) runs ``run_isg --task <subtask> --instance-indices <instance> --mode
policy --policy ispatialgym.harness.policy:HarnessPolicy``: the instance is loaded from the dataset and the step budget is
run_isg's, ``max(300, 2 x`` the demo's length``)``, unless ``--max-steps`` sets it.

Without this script, run_isg itself runs the harness the same way (several indices each in a process of its own):

    ISG_HARNESS_CONFIG=<harness_config.json> python -m ispatialgym.oracle.run_isg --task alignment_axial --instance-indices 0 1 \
        --mode policy --policy ispatialgym.harness.policy:HarnessPolicy --out <root>
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import shlex
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List

from ispatialgym.eval import instances as INST
from ispatialgym.harness.config import HarnessConfig

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
JOB = os.path.join(REPO, "scripts", "harness", "job_episode.sh")
#: viscam7 / viscam11 stay excluded as in the eval release's job script (viscam11 runs the Cursor server: cluster rule)
DEFAULT_SBATCH = "-p svl -A vision --gres=gpu:1 --constraint=turing|ampere|ada --exclude=viscam7,viscam11,visionlab-dgx1 -c 8 --mem=24G --time=04:00:00"
MANIFEST_COLUMNS = ("subtask", "instance", "ep_dir", "alias", "max_steps", "job")


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


def sampling_key(rec: INST.InstanceRecord) -> str:
    """The instance's sampling key (run_isg's ``variant``): its dataset source, else its template metadata."""
    key = (rec.source or {}).get("variant")
    if not key and rec.files.get("template_meta"):
        with open(rec.path("template_meta")) as f:
            key = json.load(f).get("variant")
    return key or rec.subtask


def prepare_episode(rec: INST.InstanceRecord, ep_dir: str, base: Dict[str, Any], overwrite: bool = False) -> Dict[str, Any]:
    """``harness_config.json`` and ``source.json`` of the episode of dataset instance ``rec``; the step budget run_isg will give it
    (raises ValueError, as run_isg would, when the settle window does not fit)."""
    if os.path.exists(os.path.join(ep_dir, "harness_config.json")) and not overwrite:
        raise FileExistsError(f"{ep_dir} is already prepared (--overwrite to redo)")
    budget = INST.step_budget(rec.episode.get("length"), base.get("max_steps"), base.get("settle_steps"), mode="policy")
    if os.path.exists(ep_dir) and overwrite:
        shutil.rmtree(ep_dir)
    os.makedirs(ep_dir, exist_ok=True)
    alias = sampling_key(rec)
    cfg = HarnessConfig.from_dict({**base, "task": rec.subtask, "instance_index": rec.instance, "alias": alias})
    cfg.save(os.path.join(ep_dir, "harness_config.json"))
    source = {"task_family": rec.task_family, "task": rec.task, "subtask": rec.subtask, "instance": rec.instance, "alias": alias,
              "tro": rec.path("tro_state"), "demo_length": rec.episode.get("length"), "max_steps": budget["max_steps"],
              "settle_steps": budget["settle_steps"]}
    with open(os.path.join(ep_dir, "source.json"), "w") as f:
        json.dump(source, f, indent=1)
    return {"alias": alias, "ep_dir": ep_dir, "max_steps": budget["max_steps"]}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default=None, help="a task family, task or subtask of the eval dataset (as run_isg --task)")
    p.add_argument("--instance-indices", type=int, nargs="+", default=[0], help="--task: instance indices (as run_isg --instance-indices)")
    p.add_argument("--all", action="store_true", help="every instance of the dataset")
    p.add_argument("--instances-root", default=None, help="dataset root (default: the dataset in the simulator data path, as run_isg)")
    p.add_argument("--out", required=True, help="root of the episode directories")
    p.add_argument("--model", default=HarnessConfig.model)
    p.add_argument("--planner", default=HarnessConfig.planner)
    p.add_argument("--model-option", action="append", default=[], help="key=value (JSON values allowed), e.g. effort=medium")
    p.add_argument("--set", action="append", default=[], help="harness config key=value, e.g. max_calls=50")
    p.add_argument("--perturb", action="append", default=[], help="probe key=value, e.g. negate_xyz=true, mask_cameras='[\"head\"]'")
    p.add_argument("--max-steps", type=int, default=None, help="run_isg --max-steps (default: run_isg's budget, max(300, 2 x the demo's length))")
    p.add_argument("--settle-steps", type=int, default=None, help="run_isg --settle-steps (default 45; at most --max-steps)")
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
    if bool(args.task) == bool(args.all):
        p.error("give exactly one of --task and --all")
    recs = INST.load_index(args.instances_root) if args.all else INST.resolve(args.task, args.instance_indices, args.instances_root)
    base = {"planner": args.planner, "model": args.model, "model_options": _parse_kv(args.model_option), "perturb": _parse_kv(args.perturb),
            "max_steps": args.max_steps, "settle_steps": args.settle_steps,
            "instances_root": os.path.abspath(args.instances_root) if args.instances_root else None}
    base.update(_parse_kv(args.set))
    HarnessConfig.from_dict(base)  # fail fast on unknown keys
    os.makedirs(args.out, exist_ok=True)
    manifest = os.path.join(args.out, "episodes.tsv")
    new = not os.path.exists(manifest)
    with open(manifest, "a") as mf:
        if new:
            mf.write("\t".join(MANIFEST_COLUMNS) + "\n")
        for rec in recs:
            ep_dir = os.path.join(os.path.abspath(args.out), rec.subtask, str(rec.instance))
            try:
                info = prepare_episode(rec, ep_dir, base, overwrite=args.overwrite)
            except FileExistsError as e:
                print(f"skip: {e}", file=sys.stderr)
                continue
            job = submit(ep_dir, args) if args.submit else ""
            mf.write("\t".join([rec.subtask, str(rec.instance), ep_dir, info["alias"], str(info["max_steps"]), job]) + "\n")
            print(f"{rec.subtask}/{rec.instance}\t{info['alias']}\tmax_steps {info['max_steps']}\t{job or 'prepared'}")
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
            cfg = json.load(open(cfg_path))
            source = json.load(open(os.path.join(ep_dir, "source.json"))) if os.path.exists(os.path.join(ep_dir, "source.json")) else {}
            mf.write("\t".join([str(cfg.get("task")), str(cfg.get("instance_index")), ep_dir, cfg.get("alias") or "", str(source.get("max_steps", "")), job]) + "\n")
            print(f"{os.path.relpath(ep_dir, args.out)}\t{cfg.get('alias') or ''}\trerun {job}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
