"""Arm-trajectory signature for the episode -> instance match of the manipulation tasks.

A regenerated (strict) instance carries its oracle sidecar with the IK witnesses of the program: the right / left arm
joint configurations the oracle drives through (pre-contact pose, waypoints). The test episode's ``observation.state``
holds the arm joint positions at every frame (R1Pro proprio: ``arm_left_qpos`` = state[3:10], ``arm_right_qpos`` =
state[28:35]). For the right instance the demo passes through (close to) every witness configuration, so

    cost(episode, instance) = mean over witnesses of  min_t || q_arm(t) - q_witness ||_2   (rad)

is small; a different instance (another object pose -> other IK solutions) scores higher. Lenient candidates have no
sidecar and get no signature (NaN).
"""

from __future__ import annotations

import glob
import json
import os
from typing import Dict, List, Optional

import numpy as np

RIGHT_IDX = [11, 13, 15, 17, 19, 21, 23]  # right_arm_joint1..7 in the 28-D sidecar joint vector (arms interleaved)
LEFT_IDX = [10, 12, 14, 16, 18, 20, 22]
STATE_RIGHT = slice(28, 35)
STATE_LEFT = slice(3, 10)


def witness_configs(sidecar_path: str) -> Dict[str, List[np.ndarray]]:
    """{arm: [7-D witness configurations]} from a sidecar's ``ik_witnesses`` (every ``targets.*.q``)."""
    out: Dict[str, List[np.ndarray]] = {"right": [], "left": []}
    try:
        with open(sidecar_path) as f:
            s = json.load(f)
    except Exception:  # noqa: BLE001
        return out
    for w in s.get("ik_witnesses") or []:
        arm = w.get("arm") or "right"
        for tgt in (w.get("targets") or {}).values():
            q = tgt.get("q") if isinstance(tgt, dict) else None
            if q is None or len(q) != 28:
                continue
            q = np.asarray(q, dtype=np.float32)
            out[arm].append(q[RIGHT_IDX] if arm == "right" else q[LEFT_IDX])
    return out


def episode_arm_states(root: str, ep: dict) -> Dict[str, np.ndarray]:
    """{arm: (T, 7)} joint positions of the episode from its data parquet."""
    import pandas as pd

    df = pd.read_parquet(os.path.join(root, "data", f"chunk-{ep['data_chunk']:03d}", f"file-{ep['data_file']:03d}.parquet"),
                         columns=["episode_index", "frame_index", "observation.state"])
    df = df[df["episode_index"] == ep["episode_index"]].sort_values("frame_index")
    st = np.stack(df["observation.state"].to_numpy()).astype(np.float32)
    return {"right": st[:, STATE_RIGHT], "left": st[:, STATE_LEFT]}


def signature_cost(arm_states: Dict[str, np.ndarray], witnesses: Dict[str, List[np.ndarray]]) -> Optional[float]:
    vals = []
    for arm, configs in witnesses.items():
        traj = arm_states.get(arm)
        if traj is None:
            continue
        for q in configs:
            vals.append(float(np.min(np.linalg.norm(traj - q[None], axis=1))))
    return float(np.mean(vals)) if vals else None


def sidecar_of(tro_path: str) -> Optional[str]:
    p = tro_path[: -len("-tro_state.json")] + ".oracle.json"
    return p if os.path.exists(p) else None
