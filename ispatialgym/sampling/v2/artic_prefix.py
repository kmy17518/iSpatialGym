"""The articulation prefix map consumer of ``articulation_small`` (i7-artic 2026-11-07, owner decision: handles stay with the ARTICULATION
library) - under v2 only (``ISPATIALGYM_V2_ONLY=1``) the workspace band of :meth:`ArticulationSmall._brace_band` is the HANDLE-ONLY view of
the alias motion's prefix map (``reach_maps`` kind ``articulation_prefix``, from ``validate_artic motions``' physics-validated segments)
instead of the in-process legacy handle band (``reach_bands.BandComputer``, which v2 only blocks); a miss rejects ``missing_map`` (one
``MAP_REQUEST`` line), never a legacy band.

The lookup (:func:`prefix_spec`, ``reach_maps.find_prefix``): the target's category / model / spawn scale, the placement rest the sampler uses
(identity: ``TT.object_pose_on(..., C.IDENTITY_QUAT, ...)``), the joint, the motion (``push_close`` / ``v2_ajar_pull`` / ``pull_open``), the
start value (``_q_main``: q_init, v2's q_ajar), the travel the band's joint values need beyond it (``_band_q_values``: the first contact
alone for v2 / push-close, the band fraction for a pull), the handle arm, the face yaw relative to the robot (pi - atan2(facing_local),
the legacy band's ``face_off``), the handle signature of the sampler's own handles (``artic_motions.handle_signature``), the support's
height above the floor (the anchors of ``ISPATIALGYM_REACH_MAP_ANCHORS``), the shared contact policy's key, the robot digest and
``ISPATIALGYM_REACH_MAP_REQUIRE``. Maps: ``ISPATIALGYM_ARTIC_PREFIX_MAPS``, default ``<ISPATIALGYM_REACH_MAPS>/_articulation``.

The view only proposes cells: the exact handle IK, the follow / push, the brace and the settle stay the sampler's (the left brace band is
None under v2 only; ``handle_only_band`` is set so the brace candidates are the exact IK's). No support policy is switched on here."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

from ispatialgym.sampling.sim_utils import Reject

log = logging.getLogger("ispatialgym.sampling.v2")


def prefix_maps_root() -> Optional[str]:
    explicit = os.environ.get("ISPATIALGYM_ARTIC_PREFIX_MAPS")
    if explicit:
        return explicit
    from ispatialgym.sampling import reach_maps as RM

    root = os.environ.get("ISPATIALGYM_REACH_MAPS")
    return os.path.join(root, RM.PREFIX_DIR) if root else None


def motion_of(s) -> Optional[str]:
    """The sampler's motion (``artic_motions.motion_kind``'s branches on the live sampler)."""
    if getattr(s, "close_push", False):
        return "push_close"
    if getattr(s, "v2", False):
        return "v2_ajar_pull"
    return "pull_open" if getattr(s, "direction", None) == "open" else None


def _anchors():
    from ispatialgym.sampling.height_anchors import AnchorSet

    ap = os.environ.get("ISPATIALGYM_REACH_MAP_ANCHORS")
    if not ap or not os.path.exists(ap):
        return None
    with open(ap) as f:
        an = json.load(f)
    return AnchorSet(an["anchors"], an.get("max_approx_m", 0.01))


def prefix_spec(s) -> Dict[str, Any]:
    """The lookup spec of the sampler's alias motion (module docstring)."""
    from ispatialgym.grasping import artic_motions as AM
    from ispatialgym.grasping import schema as GS

    mc = s.choice.models[s.role]
    tgt = s.obj(s.role)
    sc = getattr(tgt, "scale", None)
    q0 = float(s._q_main())
    f = [float(v) for v in s.facing_local]
    return {"category": mc.category, "model": mc.model, "scale": list(GS.canonical_scale(sc.tolist() if sc is not None else None)),
            "rest_quat": [0.0, 0.0, 0.0, 1.0], "joint": s.joint_name, "motion": motion_of(s), "q_start": q0,
            "q_values": [float(v) for v in s._band_q_values()], "needed_travel": AM.needed_travel(s._band_q_values(), q0), "arm": s.handle_arm,
            "face_yaw_offset": AM.face_yaw_offset(f), "handle_sig": AM.handle_signature(AM.base_handles(s), s.jinfo),
            "height": float(s.top.z) - float(getattr(s.world, "floor_z", 0.0))}


def prefix_band(s, edge, st):
    """The handle-only band view of the alias motion's prefix map (module docstring); ``Reject("missing_map")`` on a miss."""
    from ispatialgym.grasping import contact_policy as CP
    from ispatialgym.sampling import reach_maps as RM
    from ispatialgym.sampling.v2 import tabletop as TT

    spec = prefix_spec(s)
    root = prefix_maps_root()
    req = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in spec.items() if k not in ("q_values",)}
    req.update(kind=RM.PREFIX_KIND, support=str(getattr(s.top, "name", "")), scene=str(getattr(s.world, "scene_model", "")), edge=getattr(edge, "name", None))
    if spec["motion"] is None:
        TT.map_request(s, **req)
        raise Reject("missing_map", f"{s.gp.variant}: no articulation prefix motion for a {s.direction} without close_push (v2 only: no legacy handle band)")
    cfg_req = json.loads(os.environ.get("ISPATIALGYM_REACH_MAP_REQUIRE") or "{}") or None
    pm = RM.find_prefix(root, spec, anchors=_anchors(), policy_key=CP.DEFAULT.key(spec["arm"]), robot=RM.robot_digest(), config_require=cfg_req) if root else None
    if pm is None:
        log.info("%s: articulation prefix map miss (%s %s, %s arm, q_start %.3f, needed %.3f rad, face yaw %.3f, h %.4f, sig %s) in %s", s.gp.variant,
                 spec["motion"], spec["joint"], spec["arm"], spec["q_start"], spec["needed_travel"], spec["face_yaw_offset"], spec["height"], spec["handle_sig"], root)
        TT.map_request(s, **req)
        raise Reject("missing_map", f"{s.gp.variant}: no articulation prefix map for {spec['category']}/{spec['model']} {spec['motion']} ({spec['arm']} arm, "
                                    f"h {spec['height']:.4f}) in {root} (v2 only: no legacy handle band)")
    view = RM.prefix_bands(pm, spec["needed_travel"], yaw_index=RM.nearest_yaw_index(pm, spec["face_yaw_offset"]))
    view.meta["spec"] = {k: v for k, v in spec.items() if k != "q_values"}
    s.handle_only_band = True
    n = sum(len(c) for cs in view.cells_so.values() for c in cs.values())
    log.info("%s: articulation prefix map hit (%s, %s arm, needed %.3f rad, face yaw %.3f, %s): %d cells over %d stand-offs - %s", s.gp.variant, spec["motion"],
             spec["arm"], spec["needed_travel"], spec["face_yaw_offset"], view.meta["lookup"].get("how"), n, len(view.cells_so), pm.path)
    s.template_facts["artic_prefix_map"] = {"path": pm.path, "key": pm.meta["key"], "lookup": view.meta["lookup"], "needed_travel_rad": round(spec["needed_travel"], 4),
                                            "face_yaw_offset": round(spec["face_yaw_offset"], 4), "handle_sig": spec["handle_sig"], "n_cells": n,
                                            "yaw": round(float(view.meta["yaws"][0]), 4) if view.meta["yaws"] else None}
    return view
