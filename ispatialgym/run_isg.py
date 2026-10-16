"""Run a policy on iSpatialGym eval instances.

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
import logging
import os
import socket
import subprocess
import sys
import time
import traceback
from collections.abc import Sequence
from typing import Any

import yaml

import ispatialgym
from ispatialgym.eval.instances import record_of, resolve, step_budget
from ispatialgym.eval.loader import DEFAULT_HEAD_APERTURE_MM, DEFAULT_IMAGE_SIZE
from ispatialgym.eval.metrics import SUCCESS_METRICS
from ispatialgym.eval.policy import import_policy, make_policy
from ispatialgym.eval.tolerance_profile import DEFAULT_POLICY_SCALE
from ispatialgym.eval.video import VIEWS

log = logging.getLogger("ispatialgym.run_isg")

EXIT_SUCCESS, EXIT_FAILURE, EXIT_ERROR = 0, 2, 4
ENV_DEFAULTS = {"OMNIGIBSON_HEADLESS": "1", "OMNI_KIT_ACCEPT_EULA": "YES", "OMNIGIBSON_CHANGE_DETECTION_NUMPY": "1"}
INSTANCE_OPTIONS = ("task", "instance_indices", "instances_root", "instance")


def record_views(text: str) -> tuple[str, ...]:
    views = tuple(dict.fromkeys(v.strip() for v in text.split(",") if v.strip()))
    unknown = [v for v in views if v not in VIEWS]
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown view(s) {', '.join(unknown)}; choose from {', '.join(VIEWS)}")
    return views


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m ispatialgym.run_isg", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", help="YAML file of option values; command-line options override it")

    instances = p.add_argument_group("instances")
    instances.add_argument("--task", help="a task family, task or subtask of the eval dataset")
    instances.add_argument(
        "--instance-indices", type=int, nargs="+", default=[0], help="instances of --task, in the dataset's order"
    )
    instances.add_argument(
        "--instances-root",
        help="dataset root (default: $ISPATIALGYM_INSTANCES_PATH or <data path>/ispatialgym-eval-instances)",
    )
    instances.add_argument("--instance", help="instead of --task: an instance file (<instance id>-tro_state.json)")

    policy = p.add_argument_group("policy")
    policy.add_argument(
        "--mode", default="policy", choices=["policy"], help="only policy mode is available in this release"
    )
    policy.add_argument("--policy", help="module:attribute of the policy class or callable")
    policy.add_argument(
        "--policy-kwargs", type=json.loads, default={}, metavar="JSON", help="keyword arguments of the policy class"
    )

    output = p.add_argument_group("output")
    output.add_argument("--out", default="runs", help="output root; a run goes to <out>/<instance id>/")
    output.add_argument(
        "--record-view",
        type=record_views,
        default=(),
        metavar="VIEW[,VIEW...]",
        help=f"videos to record, each to <view>_rgb.mp4: {', '.join(VIEWS)}",
    )
    output.add_argument("--video-size", type=int, help="scale every video to this width")
    output.add_argument("--log-level", default="INFO")

    ev = p.add_argument_group("evaluation conditions (defaults: the standard track)")
    ev.add_argument(
        "--max-steps", type=int, help="control steps (30 Hz) of the episode (default: the instance's max_steps)"
    )
    ev.add_argument("--wallclock", type=float, help="seconds the policy's steps may take (default: 1 s per step)")
    ev.add_argument(
        "--settle-eval",
        choices=["off", "on"],
        default="off",
        help="off: success at the first step the goal holds; on: the goal must also hold through a settled window",
    )
    ev.add_argument("--settle-steps", type=int, default=45, help="the settled window, in steps")
    ev.add_argument(
        "--success-metric",
        choices=SUCCESS_METRICS,
        default="geometric",
        help="what success and the exit code follow: geometric (goal and measured targets) or symbolic (goal)",
    )
    ev.add_argument(
        "--tolerance-scale", type=float, default=DEFAULT_POLICY_SCALE, help="multiplies every isg-v2 tolerance"
    )
    ev.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="camera resolution (square)")
    ev.add_argument(
        "--head-aperture", type=float, default=DEFAULT_HEAD_APERTURE_MM, help="head camera horizontal aperture (mm)"
    )
    return p


def config_argv(path: str, parser: argparse.ArgumentParser) -> list[str]:
    """A YAML config as command-line tokens, so argparse converts and validates its values like the options."""
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
    argv: list[str] = []
    for key, value in config.items():
        if value is None or (isinstance(value, (list, tuple)) and not value):
            continue
        if key == "settle_eval" and isinstance(value, bool):  # YAML reads a bare on / off as a boolean
            value = "on" if value else "off"
        if key == "record_view" and isinstance(value, (list, tuple)):
            value = ",".join(value)
        if isinstance(value, dict):
            value = json.dumps(value)
        argv += [known[key], *map(str, value)] if isinstance(value, (list, tuple)) else [known[key], str(value)]
    return argv


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    argv = list(sys.argv[1:] if argv is None else argv)
    config = parser.parse_known_args(argv)[0].config
    args = parser.parse_args((config_argv(config, parser) if config else []) + argv)
    if (args.task is None) == (args.instance is None):
        parser.error("give exactly one of --task and --instance")
    if not args.policy:
        parser.error("--policy is required")
    if not isinstance(args.policy_kwargs, dict):
        parser.error("--policy-kwargs must be a JSON object")
    return args


def run_options(args: argparse.Namespace) -> dict[str, Any]:
    """The resolved options of one instance's run (``config.yaml``)."""
    skip = set(INSTANCE_OPTIONS) | {"config"}
    out = {k: v for k, v in vars(args).items() if k not in skip}
    out["record_view"] = list(out["record_view"])
    return out


def _json_default(o):
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    return str(o)


def dump_json(path: str, data) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=_json_default)


def _git(path: str, *args: str) -> str | None:
    try:
        out = subprocess.run(["git", "-C", path, *args], capture_output=True, text=True, timeout=60)
    except Exception:  # noqa: BLE001 - git is not installed
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def provenance() -> dict[str, Any]:
    """The code this process runs: the iSpatialGym and BEHAVIOR-1K commits and whether their sources are modified."""
    import omnigibson as og

    isg_root = os.path.dirname(os.path.dirname(os.path.abspath(ispatialgym.__file__)))
    b1k_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(og.__file__))))
    return {
        "ispatialgym_version": ispatialgym.__version__,
        "ispatialgym_commit": _git(isg_root, "rev-parse", "HEAD"),
        "ispatialgym_dirty": bool(_git(isg_root, "status", "--porcelain", "--", "ispatialgym")),
        "behavior_1k_commit": _git(b1k_root, "rev-parse", "HEAD"),
        "behavior_1k_dirty": bool(_git(b1k_root, "status", "--porcelain", "--", "OmniGibson/omnigibson", "bddl3")),
    }


RESULT_KEYS = (
    "success",
    "reason",
    "symbolic_success",
    "symbolic_success_step",
    "geometric_success",
    "geometric_success_step",
    "error_metric",
    "n_steps",
    "wall_clock_exceeded",
)


class InstanceRun:
    """One instance end to end: load, policy episode, ``meta.json`` and the ``RUN_ISG_RESULT`` line."""

    def __init__(self, args: argparse.Namespace, tro_path: str):
        self.args, self.tro_path = args, tro_path
        self.t_start = time.time()
        self.timings: dict[str, float] = {}
        self.meta: dict[str, Any] = {
            "status": "error",
            "mode": "policy",
            "argv": sys.argv[1:],
            "hostname": socket.gethostname(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.t_start)),
            "config": run_options(args),
        }
        self.result: dict[str, Any] = {"status": "error", "mode": "policy"}
        self.out_dir: str | None = None
        self.video = None

    def __call__(self) -> int:
        try:
            exit_code = self._run()
        except BaseException as e:  # noqa: BLE001 - KeyboardInterrupt included: the verdict must be written
            traceback.print_exc()
            self.meta.update({"status": "error", "error": repr(e), "traceback": traceback.format_exc()})
            self.result.update({"status": "error", "success": False, "reason": "error", "error": repr(e)})
            exit_code = EXIT_ERROR
        self._write(exit_code)
        return exit_code

    def _run(self) -> int:
        from ispatialgym.eval.loader import InstanceLoader, LoadError

        args = self.args
        policy_target = import_policy(args.policy)  # before the simulator starts
        external_camera = bool({"external", "tiled"} & set(args.record_view))
        loader = InstanceLoader(self.tro_path, args.image_size, args.head_aperture, external_camera)
        budget = self._prepare(loader)

        t0 = time.time()
        loader.build_env()
        self.meta["provenance"] = provenance()
        loader.task.configure_evaluation(budget["settle_eval"], budget["settle_steps"])
        try:
            obs, info = loader.restore()
        except LoadError as e:
            self.meta.update({"status": "failed", "reason": e.reason, "error": e.message, "loader": loader.report})
            self.result.update({"status": "failed", "success": False, "reason": e.reason, "error": e.message})
            return EXIT_FAILURE
        self.timings["load_s"] = time.time() - t0
        self.meta["loader"] = loader.report
        self.meta["tolerance_profile"] = loader.task.apply_tolerance_profile(args.tolerance_scale)

        episode = self._episode(loader, policy_target, budget, obs, info)
        success = bool(episode["success"])
        self.meta.update(episode)
        self.meta["status"] = "success" if success else "failed"
        self.result.update({"status": self.meta["status"], **{k: episode.get(k) for k in RESULT_KEYS}})
        return EXIT_SUCCESS if success else EXIT_FAILURE

    def _prepare(self, loader) -> dict[str, Any]:
        """The run's folder and identity, and its budget."""
        args = self.args
        self.out_dir = os.path.join(os.path.abspath(args.out), loader.iid)
        os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(run_options(args), f, sort_keys=False)
        names = {k: (loader.tmeta.get("eval") or {}).get(k) for k in ("task_family", "task", "subtask")}
        record = record_of(self.tro_path, args.instances_root or loader.files["root"])
        settle_eval = args.settle_eval == "on"
        budget = step_budget(
            record.max_steps if record else None, args.max_steps, args.settle_steps, args.wallclock, settle_eval
        )
        self.meta.update(
            {
                "instance_id": loader.iid,
                **names,
                "variant": loader.tro.meta.task_variant,
                "problem": loader.tmeta.get("problem"),
                "scene": loader.tmeta.get("scene"),
                "tro_path": loader.tro_path,
                "goal_image": loader.goal_image_path,
                "robot_name": loader.robot_name,
                "out_dir": self.out_dir,
                "budget": budget,
                "settle_eval": settle_eval,
                "settle_steps": budget["settle_steps"],
                "success_metric": args.success_metric,
            }
        )
        self.result.update({"instance": loader.iid, **names, "out": self.out_dir, "max_steps": budget["max_steps"]})
        log.info(
            "%s: %d steps, %.0f s of wall clock -> %s",
            loader.iid,
            budget["max_steps"],
            budget["wallclock_s"],
            self.out_dir,
        )
        return {**budget, "settle_eval": settle_eval}

    def _episode(self, loader, policy_target, budget, obs, info) -> dict[str, Any]:
        """Wait for the textures, then run the policy; returns the episode result."""
        import omnigibson as og

        from ispatialgym import single_env as SE
        from ispatialgym.eval.executor import Executor
        from ispatialgym.eval.loader import EXTERNAL_CAMERA_NAME
        from ispatialgym.eval.texture_wait import wait_for_textures
        from ispatialgym.eval.video import VideoRecorder, save_png

        args, env, robot, cam_key = self.args, loader.env, loader.robot, loader.cam_key
        external = EXTERNAL_CAMERA_NAME if loader.external_camera else None
        t0 = time.time()
        self.meta["texture_wait"], obs = wait_for_textures(
            lambda: SE.get_obs(env)[0], og.sim.render, robot.name, external
        )
        self.timings["texture_wait_s"] = time.time() - t0
        save_png(os.path.join(self.out_dir, "init_rgb.png"), obs[robot.name][cam_key]["rgb"])
        self.meta["tracked_objects"] = {k: v.name for k, v in loader.tracked.items()}
        if args.record_view:
            fps = int(loader.cfg["env"]["action_frequency"])
            self.video = VideoRecorder(
                self.out_dir, args.record_view, robot.name, cam_key, external, fps, args.video_size
            )
            self.video.start(obs)

        executor = Executor(
            env,
            loader.task,
            robot,
            cam_key,
            budget["max_steps"],
            budget["wallclock_s"],
            args.success_metric,
            self.video,
        )
        executor.last_obs, executor.last_info, executor.last_rgb = obs, info, obs[robot.name][cam_key]["rgb"]
        policy = make_policy(policy_target, args.policy_kwargs)
        if callable(getattr(policy, "bind", None)):
            policy.bind(
                executor=executor, loader=loader, meta=self.meta, out_dir=self.out_dir, tro_path=loader.tro_path
            )
        episode = executor.run(policy)
        if callable(getattr(policy, "finalize", None)):
            policy.finalize(episode)
        if episode.get("forced_failure"):
            episode.update({"success": False, "reason": episode.get("reason") or "forced_failure"})
        self.timings["episode_s"] = episode.pop("episode_s")
        save_png(os.path.join(self.out_dir, "final_rgb.png"), executor.last_rgb)
        return episode

    def _write(self, exit_code: int) -> None:
        if self.video is not None:
            self.meta["videos"] = self.video.close()
            if self.video.errors:
                self.meta["video_errors"] = dict(self.video.errors)
        self.timings["total_s"] = time.time() - self.t_start
        self.meta["timings"] = {k: round(v, 2) for k, v in self.timings.items()}
        self.result.update({"wall_time_s": round(self.timings["total_s"], 1), "exit_code": exit_code})
        if self.out_dir is not None:
            dump_json(os.path.join(self.out_dir, "meta.json"), self.meta)
        print("RUN_ISG_RESULT " + json.dumps(self.result, default=_json_default), flush=True)


def run_task(args: argparse.Namespace) -> int:
    """The dataset instances ``--task`` / ``--instance-indices`` resolve to: one runs in this process, several each in
    a process of its own."""
    records = resolve(args.task, args.instance_indices, args.instances_root)
    for i, rec in zip(args.instance_indices, records, strict=True):
        log.info("--task %s index %d -> %s instance %d", args.task, i, rec.subtask, rec.instance)
    if len(records) == 1:
        return InstanceRun(args, records[0].path("tro_state"))()
    codes = []
    for rec in records:
        run_dir = os.path.join(os.path.abspath(args.out), rec.instance_id)
        os.makedirs(run_dir, exist_ok=True)
        config_path = os.path.join(run_dir, "config.yaml")
        with open(config_path, "w") as f:
            yaml.safe_dump(run_options(args), f, sort_keys=False)
        cmd = [
            sys.executable,
            "-m",
            "ispatialgym.run_isg",
            "--config",
            config_path,
            "--instance",
            rec.path("tro_state"),
        ]
        codes.append(subprocess.run(cmd).returncode)
    return max(codes)


def main(argv: Sequence[str] | None = None) -> int:
    for key, value in ENV_DEFAULTS.items():
        os.environ.setdefault(key, value)  # before OmniGibson is imported, by the policy's module or the loader
    args = parse_args(argv)
    logging.basicConfig(format="[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s")
    logging.getLogger("ispatialgym").setLevel(args.log_level)
    if args.task:
        return run_task(args)
    return InstanceRun(args, args.instance)()


if __name__ == "__main__":
    code = main()
    og = sys.modules.get("omnigibson")
    if og is not None and og.app is not None:
        og.cleanup()  # closing Isaac Sim would end the process with exit code 0
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
