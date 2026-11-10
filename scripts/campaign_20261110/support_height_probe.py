#!/usr/bin/env python3
"""Build and probe one private, bottom-anchored support-height scene variant.

The measured collision geometry supplies the bottom correction. Conservative
collision-piece AABB screening can reject usable candidates; it cannot admit a
candidate whose boxes overlap an unmoved fixture beyond the contact tolerance.
Successful probes supply measured map-height keys, never a template release.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import statistics
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.campaign_20261110.contracts import file_digest, require, write_json_atomic


def overlap_depth(box_a, box_b):
    return min(min(float(box_a[1][i]), float(box_b[1][i])) - max(float(box_a[0][i]), float(box_b[0][i])) for i in range(3))


def corrected_root_height(points, quaternion, root_z, factor):
    """Root-local measured points already include the default object scale."""
    x, y, z, w = [float(v) for v in quaternion]
    require(abs(x*x+y*y+z*z+w*w-1) < 1e-4, "support quaternion is not normalized")
    require(math.isfinite(factor) and 0.8 <= factor <= 1.2, "height factor outside requested range")
    rz = [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
    old_min = min(sum(float(p[i]) * rz[i] for i in range(3)) for p in points)
    new_min = min(float(p[0])*rz[0] + float(p[1])*rz[1] + float(p[2])*factor*rz[2] for p in points)
    return float(root_z) + old_min - new_min


def settle_support(world, obj, steps=90):
    """Kinematic roots have no velocity API; still inspect dynamic child links."""
    kinematic = bool(obj.kinematic_only)
    error = world.settle([] if kinematic else [obj], steps=steps,
                         still=[other for other in world.scene.objects if other.name != obj.name])
    if error or not kinematic:
        return error
    for name, link in obj.links.items():
        if not callable(getattr(link, "get_linear_velocity", None)):
            continue
        linear = math.sqrt(sum(float(v)**2 for v in link.get_linear_velocity()))
        angular = math.sqrt(sum(float(v)**2 for v in link.get_angular_velocity()))
        if linear > world.STABLE_LIN_VEL or angular > world.STABLE_ANG_VEL:
            return f"{obj.name}/{name} moving (v={linear:.3f} m/s, w={angular:.3f} rad/s)"
    return None


def run(args):
    root, output = args.campaign_root.resolve(), args.output.resolve()
    require(output.is_relative_to(root) and output != root, "probe output must be inside the private campaign root")
    require(not output.exists(), "support probe output already exists")
    inventory = json.loads(args.profiles.read_text())
    binding = inventory["bindings"][args.binding]
    profile = inventory["profiles"][binding["profile_id"]]
    require(profile["ready"] is True, "support model is not ready")
    require(args.slot in (0, 1), "height slot must be zero or one")
    scale = profile["absolute_scales"][args.slot]
    require(scale[:2] == profile["default_scale"][:2], "support height changed horizontal scale")
    factor = scale[2] / profile["default_scale"][2]
    scene, name = binding["scene"], binding["name"]
    stable_path = args.stable_root / scene / "json" / f"{scene}_stable.json"
    stable = json.loads(stable_path.read_text())
    initial = stable["objects_info"]["init_info"][name]["args"]
    require(initial["category"] == profile["category"] and initial["model"] == profile["model"], "scene support identity mismatch")
    require(all(abs(a-b) < 1e-6 for a,b in zip(initial["scale"], profile["default_scale"])), "scene default scale differs from profile")
    rooms = initial.get("in_rooms") or []
    require(rooms, "support has no task room")
    output.mkdir(parents=True)
    os.environ.pop("ISPATIALGYM_CAMPAIGN_CONFIG", None)
    os.environ.pop("ISPATIALGYM_CAMPAIGN_GOAL_PLAN", None)
    # Established import order pins Warp before the simulator is imported.
    __import__("ispatialgym.sampling.robot_utils")
    import omnigibson as og
    from ispatialgym.sampling import scene_variants as SV
    from ispatialgym.sampling.sim_utils import SimWorld, _collision_piece_aabbs

    report = {"schema": "isg-support-height-probe-v1", "binding_id": args.binding, "slot": args.slot,
              "scene": scene, "object": name, "profile_id": binding["profile_id"], "absolute_scale": scale,
              "profile_manifest_sha256": file_digest(args.profiles), "source_scene_sha256": file_digest(stable_path),
              "geometry_admitted": False, "production_admitted": False}
    try:
        chairs = SV.chairs_near_tables(stable, rooms, [name])
        path, baseline = SV.write_variant(str(stable_path), chairs, str(output / "baseline"))
        variant = {"path": path, "removed_objects": chairs, "freed_footprints": SV.freed_footprints(stable, chairs),
                   "moved_objects": baseline["metadata"]["ispatialgym"].get("moved_objects", {})}
        world = SimWorld(scene, rooms, scene_variant=variant, cameras=False)
        obj = world.obj(name)
        position, orientation = obj.get_position_orientation()
        lo, _ = obj.aabb
        floor = float(world.floor_z)
        bottom = float(lo[2])
        require(abs(bottom-floor) <= 0.03, "scene support does not rest on the floor; height scaling inadmissible")
        hull = world.local_hull(obj)
        require(len(hull) > 3, "support has insufficient collision geometry")
        root_pose = [float(v) for v in position]
        root_pose[2] = corrected_root_height(hull, orientation, position[2], factor)
        changed = copy.deepcopy(baseline)
        changed["objects_info"]["init_info"][name]["args"]["scale"] = scale
        changed["state"]["registry"]["object_registry"][name]["root_link"].update(pos=root_pose, ori=[float(v) for v in orientation],
                                                                                lin_vel=[0., 0., 0.], ang_vel=[0., 0., 0.])
        changed.setdefault("metadata", {}).setdefault("ispatialgym", {})["campaign_support_height"] = {
            "object": name, "default_scale": profile["default_scale"], "applied_scale": scale,
            "default_bottom_z": bottom, "binding_id": args.binding, "slot": args.slot}
        stable_dir = output / "stable_scenes"
        changed_path = stable_dir / f"{scene}_stable.json"
        write_json_atomic(changed_path, changed)
        report.update(default_bottom_z=bottom, default_floor_z=floor, root_position=root_pose,
                      stable_scene_path=str(changed_path), stable_scene_sha256=file_digest(changed_path), removed_chairs=chairs)
        og.clear()
        variant["path"] = str(changed_path)
        world = SimWorld(scene, rooms, scene_variant=variant, cameras=False)
        obj = world.obj(name)
        require(all(abs(float(a)-b) <= 1e-6 for a,b in zip(obj.scale, scale)), "live support scale differs from planned scale")
        before = obj.get_position_orientation()[0].clone()
        pieces = _collision_piece_aabbs(obj)
        blockers = []
        for other in world.scene.objects:
            if other.name in {name, world.robot.name} or other.category == "floors" or other.name in chairs:
                continue
            other_boxes = _collision_piece_aabbs(other)
            depth = max((overlap_depth(a,b) for a in pieces for b in other_boxes), default=-1.)
            if depth > 0.002:
                blockers.append({"object": other.name, "depth_m": depth})
        report["conservative_overlap_blockers"] = blockers
        require(not blockers, "scaled support overlaps unmoved scene geometry")
        report["settling_mode"] = "kinematic_root_dynamic_child_checks" if obj.kinematic_only else "dynamic_object"
        error = settle_support(world, obj, steps=90)
        require(error is None, f"scaled support did not settle: {error}")
        lo, hi = obj.aabb
        require(abs(float(lo[2])-bottom) <= 0.003, "scaled support bottom was not preserved")
        require(float(((obj.get_position_orientation()[0]-before)**2).sum()) <= 0.003**2, "scaled support moved during settling")
        predicted = float(binding["floor_z_world_m"]) + float(binding["predicted_heights_above_floor_m"][args.slot])
        hits = []
        for ix in range(1, 6):
            for iy in range(1, 6):
                xy = [float(lo[0])+(float(hi[0])-float(lo[0]))*ix/6,
                      float(lo[1])+(float(hi[1])-float(lo[1]))*iy/6]
                hit = world.surface_z_under(xy, predicted+0.04, depth=0.08)
                if hit and hit["body"].startswith(obj.prim_path+"/") and hit["normal"][2] >= 0.95:
                    hits.append({"xy": xy, **hit})
        require(len(hits) >= 3, "scaled support surface could not be measured at the predicted plane")
        height_world = statistics.median(h["z"] for h in hits)
        require(max(abs(h["z"]-height_world) for h in hits) < 0.01, "support probe hits inconsistent planes")
        height = height_world - float(world.floor_z)
        report.update(geometry_admitted=True, measured_height_world=height_world, floor_z_world=float(world.floor_z),
                      actual_height_key=f"{height:.4f}", measured_height_above_floor=height, surface_hits=hits,
                      aabb_lower=lo.tolist(), aabb_upper=hi.tolist(), status="geometry_passed_layout_and_demo_validation_pending")
    except Exception as exc:
        report.update(status="geometry_rejected", error=str(exc), error_type=type(exc).__name__)
        raise
    finally:
        write_json_atomic(output / "height_probe.json", report)
        og.shutdown()
    print(json.dumps(report))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("campaign-root", "profiles", "stable-root", "output"):
        p.add_argument("--"+name, type=Path, required=True)
    p.add_argument("--binding", required=True)
    p.add_argument("--slot", type=int, required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
