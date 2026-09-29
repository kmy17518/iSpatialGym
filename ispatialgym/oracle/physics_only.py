"""Physics-only collection (opt-in: ``ISPATIALGYM_PHYSICS_ONLY=1``).

The episode steps physics without rendering (``og.sim.render_on_step(False)``), a rollback does not re-render until the
camera shows the restored scene, and only demo.hdf5 is recorded (no video, LeRobot dataset or BEHAVIOR shard: their
observations come from a replay of the recorded states, :mod:`ispatialgym.multienv.replay`).

Rendering is also what keeps Fabric's transforms current: PhysX writes rigid-body transforms into Fabric and Fabric
recomputes the world transforms of child prims (the camera sensors, other child xforms) during the render update, and
OmniGibson reads every world pose from Fabric (``usd_utils.get_world_pose``). Rigid links and objects are read through
PhysX and stay current without a render, but a camera sensor's pose would be stale. :func:`sync_transforms` does what
the render update does to transforms, without rendering: :func:`episode` calls it after every simulator step, and the
executor after a rollback loads a checkpoint. Readers keep calling ``get_position_orientation()`` and get the values a
rendered run reads.
"""

from __future__ import annotations

import contextlib
from typing import Optional

import numpy as np
import torch as th

ENV_FLAG = "ISPATIALGYM_PHYSICS_ONLY"

_FABRIC = {"iface": None}


def sync_transforms(sim=None) -> None:
    """Push PhysX's transforms into Fabric and recompute Fabric's world transforms, as the render update does
    (Isaac's ``SimulationContext.render`` calls the same ``force_update`` first)."""
    if sim is None:
        import omnigibson as og

        sim = og.sim
    if _FABRIC["iface"] is None:
        from omni.physxfabric import get_physx_fabric_interface

        _FABRIC["iface"] = get_physx_fabric_interface()
    _FABRIC["iface"].force_update(sim.get_physics_dt(), sim.current_time)
    sim.fabric_hierarchy.update_world_xforms()


@contextlib.contextmanager
def episode(sim):
    """Run a block without rendering on the simulator's steps, with Fabric's transforms synced after every step."""
    had_own = "step" in sim.__dict__
    prev = sim.__dict__.get("step")
    base = sim.step

    def step(*args, **kwargs):
        out = base(*args, **kwargs)
        sync_transforms(sim)
        return out

    with sim.render_on_step(False):
        sim.step = step
        try:
            yield
        finally:
            if had_own:
                sim.step = prev
            else:
                del sim.step


#: the robot type BEHAVIOR's shard writer is configured with (``behavior_lerobot.ROBOT_TYPE``; not imported from there:
#: importing it sets up lerobot)
ROBOT_TYPE = "R1Pro"
_PROPRIO_KEYS: list = []


def behavior_proprio(robot) -> np.ndarray:
    """BEHAVIOR's proprioception of ``robot`` now, as the shard writer stores it (``observation.state``): the robot's
    proprioception dict concatenated in ``PROPRIOCEPTION_INDICES["R1Pro"]`` key order (what ``get_proprioception``
    returns for a robot configured with those keys, as ``behavior_lerobot.robot_overrides`` does), as float32."""
    if not _PROPRIO_KEYS:
        from omnigibson.eval.utils.eval_utils import PROPRIOCEPTION_INDICES

        _PROPRIO_KEYS.extend(PROPRIOCEPTION_INDICES[ROBOT_TYPE].keys())
    d = robot._get_proprioception_dict()
    return th.cat([d[k] for k in _PROPRIO_KEYS]).float().detach().cpu().numpy()


def check_recording(video: bool, lerobot: bool, behavior_shard: Optional[str]) -> None:
    """Physics-only collection records demo.hdf5 alone: every frame sink would store stale images."""
    sinks = [n for n, on in (("--video", video), ("LeRobot (--record full)", lerobot), ("--behavior-shard", behavior_shard)) if on]
    if sinks:
        raise ValueError(f"{ENV_FLAG}=1 records demo.hdf5 only (use --record hdf5 without {', '.join(sinks)}); "
                         "observations come from ispatialgym.multienv.replay")
