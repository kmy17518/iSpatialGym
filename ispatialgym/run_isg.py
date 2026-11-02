"""Policy evaluation entrypoint: delegates to :mod:`ispatialgym.oracle.run_isg` with ``--mode policy`` as the default.

    python -m ispatialgym.run_isg --task alignment_axial --instance-indices 0 --policy my_package.policy:MyPolicy
    python -m ispatialgym.run_isg --config configs/eval.yaml --task alignment --policy my_package.policy:MyPolicy

Options may also come from a YAML file (``--config``; keys are the option names with underscores, plus
``policy_kwargs``); command-line options override it. Each instance runs in a process of its own and writes
``<out>/<instance id>/``: ``meta.json`` (verdict, metrics, budget, instance setup), ``config.yaml`` (the resolved
options), ``init_rgb.png``, ``final_rgb.png`` and the ``--record-view`` videos. One ``RUN_ISG_RESULT {...}`` line is
printed per instance; the exit code is 0 on success, 2 on failure and 4 on error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

ENV_DEFAULTS = {
    "OMNIGIBSON_HEADLESS": "1",
    "OMNI_KIT_ACCEPT_EULA": "YES",
    "OMNIGIBSON_CHANGE_DETECTION_NUMPY": "1",
}

EXIT_SUCCESS, EXIT_FAILURE, EXIT_NO_ORACLE, EXIT_ERROR = 0, 2, 3, 4
DEFAULT_IMAGE_SIZE = 1080
DEFAULT_HEAD_APERTURE_MM = 40.0
VIEWS = ("tiled", "head", "external", "left_wrist", "right_wrist")
INSTANCE_OPTIONS = (
    "task",
    "instance_indices",
    "instances_root",
    "instance",
    "queue",
    "claims",
    "done_file",
    "results",
    "deadline",
    "max_instances",
    "poll_s",
)


def record_views(text: str) -> Tuple[str, ...]:
    """``--record-view``: a comma-separated list of :data:`VIEWS`."""
    views = tuple(dict.fromkeys(v.strip() for v in text.split(",") if v.strip()))
    unknown = [v for v in views if v not in VIEWS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown view(s) {', '.join(unknown)}; choose from {', '.join(VIEWS)}")
    return views


def build_parser(default_policy: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="YAML file of option values; command-line options override it")
    p.add_argument("--instance", default=None, help="path to <iid>-tro_state.json of a sampled instance")
    p.add_argument(
        "--task",
        default=None,
        help="instead of --instance: a task family, task or subtask of the eval dataset (ispatialgym.eval.instances); runs "
        "its instances --instance-indices (one in this process, several each in a process of its own)",
    )
    p.add_argument(
        "--instance-indices",
        type=int,
        nargs="+",
        default=[0],
        help="--task: instance indices (a task's instances are its subtasks' instances in the dataset's order)",
    )
    p.add_argument(
        "--instances-root",
        default=None,
        help="--task: dataset root (default $ISPATIALGYM_INSTANCES_PATH or $OMNIGIBSON_DATA_PATH/ispatialgym-eval-instances)",
    )
    p.add_argument(
        "--queue",
        default=None,
        metavar="INST_ROOT",
        help="instead of --instance: replay every instance a sampler writes under INST_ROOT (sample_isg --out), one after "
        "the other in this process's world (run_queue)",
    )
    p.add_argument("--claims", default=None, help="--queue: directory of the claim files (default: <INST_ROOT>/../claims)")
    p.add_argument("--done-file", default=None, help="--queue: once this file exists, exit when no ready instance is left unclaimed")
    p.add_argument("--results", default=None, help="--queue: append one JSON line per instance to this file")
    p.add_argument("--deadline", type=float, default=None, help="--queue: epoch seconds after which no instance starts")
    p.add_argument("--max-instances", type=int, default=None, help="--queue: instances per process")
    p.add_argument("--poll-s", type=float, default=10.0, help="--queue: wait between scans for new instances")
    p.add_argument("--sidecar", default=None, help="oracle sidecar (default: <iid>.oracle.json next to the TRO)")
    p.add_argument("--mode", default="policy" if default_policy else "oracle", choices=["oracle", "replay", "policy"])
    p.add_argument("--replay", default=None, help="replay mode: LeRobot dataset root or demo.hdf5 (default for an eval dataset instance: its demo)")
    p.add_argument("--episode", type=int, default=0, help="replay mode: episode index")
    p.add_argument("--policy", default=None, help="policy mode: module:callable -> policy(obs, info) -> action")
    p.add_argument("--policy-kwargs", type=json.loads, default={}, metavar="JSON", help="keyword arguments of the policy class")
    p.add_argument("--out", default="runs" if default_policy else None, help="output root; the run goes to <out>/<iid>/")
    p.add_argument("--record", default="full", choices=["full", "hdf5", "video"])
    p.add_argument(
        "--record-view",
        type=record_views,
        default=(),
        metavar="VIEW[,VIEW...]",
        help=f"videos to record, each to <out>/<iid>/<view>_rgb.mp4: {', '.join(VIEWS)}",
    )
    p.add_argument("--video-size", type=int, default=None, help="scale every video to this width (aspect kept)")
    p.add_argument("--goal-image", action="store_true", help="capture the goal image after the settled check and write goal_observation into the TRO")
    p.add_argument("--no-goal-override", action="store_true", help="do not write the settled values back into the TRO goal poses")
    p.add_argument("--qa", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--carry-leg", default="straight", choices=["straight", "trajopt"])
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--settle-steps", type=int, default=45 if default_policy else None)
    p.add_argument("--wallclock", type=float, default=None)
    p.add_argument("--settle-eval", default="off" if default_policy else "auto", choices=["auto", "on", "off"])
    p.add_argument("--success-metric", default="geometric", choices=["geometric", "symbolic"])
    p.add_argument("--tolerance-scale", type=float, default=3.0 if default_policy else None)
    p.add_argument("--retries", type=int, default=1, help="retries per action from its checkpoint")
    p.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    p.add_argument("--head-aperture", type=float, default=DEFAULT_HEAD_APERTURE_MM)
    p.add_argument("--robot-name", default=None)
    p.add_argument("--allow-sha-mismatch", action="store_true")
    p.add_argument("--frame-buffer", default="jpeg", choices=["jpeg", "raw"])
    p.add_argument("--all-cameras", action="store_true")
    p.add_argument("--behavior-shard", default=None)
    from ispatialgym.oracle import sleep_assist as SA  # the physics flags of both drivers (run_isg_vec's defaults)

    SA.add_arguments(p)
    p.add_argument("--lean-step", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--log-level", default=os.environ.get("ISPATIALGYM_LOG_LEVEL", "INFO"))
    p.add_argument("--load-only", action="store_true")
    return p


def config_argv(path: str, parser: argparse.ArgumentParser) -> List[str]:
    """A YAML config as command-line tokens, so argparse converts and validates its values like the options."""
    import yaml

    with open(path) as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise SystemExit(f"{path}: a config is a mapping of option names to values")
    known = {
        a.dest: a.option_strings[0] for a in parser._actions if a.option_strings and a.dest not in ("help", "config")
    }
    unknown = sorted(set(config) - set(known))
    if unknown:
        raise SystemExit(f"{path}: unknown option(s) {unknown}; known: {sorted(known)}")
    negatable = {a.dest for a in parser._actions if isinstance(a, argparse.BooleanOptionalAction)}  # --qa / --no-qa, --lean-step / --no-lean-step
    argv: List[str] = []
    for key, value in config.items():
        if value is None or (isinstance(value, (list, tuple)) and not value):
            continue
        if key == "settle_eval" and isinstance(value, bool):  # YAML reads a bare on / off as a boolean
            value = "on" if value else "off"
        if key == "sleep_assist" and isinstance(value, bool):  # likewise: off, or on = the default window
            value = parser.get_default("sleep_assist") if value else "off"
        if key == "record_view" and isinstance(value, (list, tuple)):
            value = ",".join(value)
        if isinstance(value, bool):
            if key in negatable:
                argv.append(known[key] if value else "--no-" + known[key][2:])
            elif value:
                argv.append(known[key])
            continue
        if isinstance(value, dict):
            value = json.dumps(value)
        argv += [known[key], *map(str, value)] if isinstance(value, (list, tuple)) else [known[key], str(value)]
    return argv


def run_options(args: argparse.Namespace) -> Dict[str, Any]:
    """The resolved options of one instance's run (``config.yaml``)."""
    skip = set(INSTANCE_OPTIONS) | {"config"}
    out = {k: v for k, v in vars(args).items() if k not in skip}
    out["record_view"] = list(out["record_view"])
    return out


def parse_args(argv: Optional[Sequence[str]] = None, default_policy: bool = True) -> argparse.Namespace:
    from ispatialgym.oracle import r1pro_model as M

    p = build_parser(default_policy=default_policy)
    argv = list(sys.argv[1:] if argv is None else argv)
    config = p.parse_known_args(argv)[0].config
    args = p.parse_args((config_argv(config, p) if config else []) + argv)
    if sum(x is not None for x in (args.instance, args.queue, args.task)) != 1:
        p.error("exactly one of --instance, --queue and --task")
    if args.mode == "policy" and not args.policy:
        p.error("--policy is required")
    if args.mode == "policy" and not isinstance(args.policy_kwargs, dict):
        p.error("--policy-kwargs must be a JSON object")
    if args.task and args.replay and len(args.instance_indices) > 1:
        p.error("--replay names one demo: give one --instance-indices value (without --replay each instance replays its own demo)")
    if args.all_cameras and M.head_camera_only():
        p.error("--all-cameras records the wrist cameras, which ISPATIALGYM_HEAD_CAMERA_ONLY removes")
    wrist_views = [v for v in args.record_view if v in ("tiled", "left_wrist", "right_wrist")]
    if wrist_views and M.head_camera_only():
        p.error(f"--record-view {','.join(wrist_views)} records the wrist cameras, which ISPATIALGYM_HEAD_CAMERA_ONLY removes")
    if args.behavior_shard and M.head_camera_only():
        p.error("--behavior-shard records the wrist cameras, which ISPATIALGYM_HEAD_CAMERA_ONLY removes")
    if args.behavior_shard and args.record not in ("full", "hdf5"):
        p.error("--behavior-shard needs demo.hdf5 (--record hdf5 or full): retried demos are replayed from it")
    return args


def main(argv: Optional[Sequence[str]] = None, default_policy: bool = True) -> int:
    for key, value in ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)
    args = parse_args(argv, default_policy=default_policy)
    from ispatialgym.oracle import run_isg as _oracle_run_isg

    return _oracle_run_isg._run_with_args(args, argv)


__all__ = [
    "EXIT_ERROR",
    "EXIT_FAILURE",
    "EXIT_NO_ORACLE",
    "EXIT_SUCCESS",
    "INSTANCE_OPTIONS",
    "build_parser",
    "config_argv",
    "main",
    "parse_args",
    "record_views",
    "run_options",
]


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.stderr.flush()
    og = sys.modules.get("omnigibson")
    if og is not None and getattr(og, "app", None) is not None:
        try:
            og.shutdown()
        except BaseException:  # noqa: BLE001
            import traceback

            traceback.print_exc()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
