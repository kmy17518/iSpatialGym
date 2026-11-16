"""Ground-truth actions as a ``run_isg --mode policy`` policy (``--policy ispatialgym.eval.gt_policy:GTPolicy``).

The positive control of policy evaluation: the instance's own demo (the eval dataset's ``demo.hdf5``) streamed through the
policy path - the policy step budget, stop-on-success, the settled evaluation window - exactly as a checkpoint's actions
are. Its action rows, then its recorded settle rows, then (``hold_last``) the last row repeated until the episode ends, as
a policy holding still would.

``dump_steps`` (optional, e.g. ``"0,1,2,30"``): the observation the policy received at those steps - the three cameras'
rgb (uint8 HWC) and the 61-D proprioception - is written to ``<out>/obs_dump.npz`` at the end of the episode, for the
input-consistency audit (eval-time inputs against the rendered training-format episode of the same instance).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

CAMERAS = ("zed_link", "left_realsense_link", "right_realsense_link")


def _np(x) -> np.ndarray:
    if hasattr(x, "detach"):
        x = x.detach()
    if hasattr(x, "cpu"):
        x = x.cpu()
    return np.asarray(x)


class GTPolicy:
    def __init__(self, hold_last: bool = True, dump_steps: Optional[str] = None, demo: Optional[str] = None):
        self.hold_last = bool(hold_last)
        self.dump_steps = sorted({int(s) for s in str(dump_steps).split(",") if s.strip()}) if dump_steps else []
        self.demo = demo
        self.rows: List[np.ndarray] = []
        self.n = 0
        self.out_dir: Optional[str] = None
        self.dump: Dict[str, List[np.ndarray]] = {}
        self.info: Dict[str, Any] = {}

    def bind(self, executor=None, loader=None, meta=None, out_dir=None, **_):
        from ispatialgym.oracle import run_isg as R

        demo, counts = self.demo, None
        if demo is None:
            rec = R._dataset_record(loader)
            if rec is None:
                raise RuntimeError(f"GTPolicy: {loader.tro_path} is not an eval dataset instance (give demo=<demo.hdf5>)")
            demo = rec.demo_path()
            counts = rec.episode
        n_action, n_settle = R.replay_counts(demo, 0, counts)
        self.rows = [np.asarray(a.numpy() if hasattr(a, "numpy") else a, dtype=np.float32).reshape(-1) for a in R.replay_rows(demo, 0, n_action, n_settle)]
        self.out_dir = out_dir
        self.info = {"demo": demo, "action_rows": n_action, "settle_rows": n_settle, "rows": len(self.rows)}
        if meta is not None:
            meta["gt_policy"] = self.info

    def _record(self, obs) -> None:
        robot = next(iter(k for k in obs if isinstance(obs[k], dict) and "proprio" in obs[k]))
        ro = obs[robot]
        self.dump.setdefault("step", []).append(np.int64(self.n))
        self.dump.setdefault("proprio", []).append(_np(ro["proprio"]).astype(np.float32).reshape(-1))
        for cam in CAMERAS:
            key = next(k for k in ro if k.endswith(f"{cam}:Camera:0"))
            rgb = _np(ro[key]["rgb"])[..., :3]
            self.dump.setdefault(cam, []).append(rgb.astype(np.uint8))

    def __call__(self, obs, info):
        if self.n in self.dump_steps:
            self._record(obs)
        if self.n < len(self.rows):
            a = self.rows[self.n]
        elif self.hold_last and self.rows:
            a = self.rows[-1]
        else:
            return None
        self.n += 1
        return a.copy()

    def finalize(self, result) -> None:
        self.info["steps_streamed"] = self.n
        if self.dump and self.out_dir:
            np.savez_compressed(os.path.join(self.out_dir, "obs_dump.npz"), **{k: np.stack(v) for k, v in self.dump.items()})
