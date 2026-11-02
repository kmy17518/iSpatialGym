"""Configuration matching / block design (``configuration_matching-0``): 2 x 2 (or 3 x 3) Kohs blocks.

Init: blocks scattered on the support with distinct up faces (2 x 2: the two remaining faces face the robot on
some block). Goal: the pattern's top faces in a touching grid anchored at the init position of the block that is
upper-left in the image, aligned with the table axes (``layout_reached`` gates, ``pose_reached`` measured).
"""

from __future__ import annotations

import json
import logging
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.bddl_writer import GROUP_NAMES
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import TABLE_CATEGORIES, Draft, TaskSampler

from omnigibson.utils.asset_utils import get_dataset_path  # noqa: E402

log = logging.getLogger("ispatialgym.sampling")

DEFAULT_FACES = {"+z": "red", "-z": "blue", "+x": "white", "-x": "yellow", "+y": "blue|yellow", "-y": "red|white"}


class BlockDesignSampler(TaskSampler):
    def __init__(self, ctx):
        super().__init__(ctx)
        self.top = None
        self.stance = None
        self.cells = None
        self.hull = None
        self.edge = 0.0254
        self.faces = dict(DEFAULT_FACES)

    # ------------------------------------------------------------------ template

    def bind_fixed_roles(self):
        self.bind_floor()
        self._table_candidates = self.fixed_candidates("table", TABLE_CATEGORIES)

    def prepare_bindings(self):
        blocks = self.objs("blocks")
        if not blocks:
            raise Reject("spec", "no blocks spawned")
        self.hull = self.world.local_hull(blocks[0])
        ext = self.hull.max(dim=0).values - self.hull.min(dim=0).values
        self.edge = float(ext.mean())
        choice = self.choices["blocks"]
        faces_path = os.path.join(get_dataset_path(choice.dataset), "objects", choice.category, choice.model, "misc", "isg_block_faces.json")
        if os.path.exists(faces_path):
            with open(faces_path) as f:
                self.faces = dict(json.load(f)["faces"])
        errors = []
        for table in self._table_candidates:
            try:
                self._try_support(table)
                self.bind_scene_object("table", table)
                return
            except Reject as e:
                errors.append(f"{table.name}: {e}")
        raise Reject("no_support", "; ".join(errors))

    def grid(self) -> Tuple[int, int]:
        g = self.gen("grid", [2, 2])
        return int(g[0]), int(g[1])

    def _try_support(self, table):
        self.kit.sync_world()
        top = self.support_top(table)
        n_rows, n_cols = self.grid()
        n_blocks = len(self.objs("blocks"))
        if n_blocks != n_rows * n_cols:
            raise Reject("spec", f"{n_blocks} blocks for a {n_rows}x{n_cols} grid")
        depth_range = tuple(self.gen("depth_range", [0.12, 0.40]))
        lateral_half = float(self.gen("lateral_half", 0.22))
        stance = self.stance_with_view(top, sum(depth_range) / 2, preferred_edge=self.scene_spec.fixed.get("table_front_edge"))
        cells = self.region_cells(top, stance, depth_range, lateral_half, margin=0.03)
        if len(cells) == 0:
            raise Reject("support", "placement region empty")
        vis = self.visible_cells(cells, stance.view.cam_pose, top.z, self.edge, 0.08, self.edge)
        cells = cells[vis]
        if len(cells) == 0:
            raise Reject("support", "no visible cell")
        coarse = self.coarse_cells(cells, float(self.gen("reach_resolution", 0.06)), top.resolution)
        # generic top-down pinch on a 2.5 cm cube: eef origin ~ edge/2 + fingertip length above the table
        mask, tr = self.reachable_cells(coarse, stance, top.z + max(self.edge * 0.6, 0.03), tilts=(0.0, 30.0))
        if int(mask.sum()) < n_blocks + 2:
            raise Reject("unreachable", f"only {int(mask.sum())} reachable cells")
        self.top, self.stance, self.cells = top, stance, coarse[mask]
        self.reach_translates = [t for t, m in zip(tr, mask.tolist()) if m]
        self.template_meta.update({"support": top.to_json(), "stance": stance.to_json(), "n_reachable_cells": int(mask.sum()), "edge": self.edge, "faces": self.faces})

    # ------------------------------------------------------------------ faces / patterns

    def color_to_face(self) -> Dict[str, str]:
        return {c: f for f, c in self.faces.items()}

    def pattern(self, rng) -> Tuple[str, List[str]]:
        patterns = self.gen("patterns", {"default": ["red", "blue", "blue", "red"]})
        override = getattr(self, "sub_variant_overrides", None)
        if override:
            for v in override:
                if v in patterns:
                    return v, list(patterns[v])
        pid = list(patterns)[int(rng.integers(len(patterns)))]
        return pid, list(patterns[pid])

    # ------------------------------------------------------------------ instance

    def sample(self, rng: np.random.Generator) -> Draft:
        d = Draft()
        st, top = self.stance, self.top
        names = [b.name for b in self.bindings.many("blocks")]
        n = len(names)
        n_rows, n_cols = self.grid()
        pid, pattern = self.pattern(rng)
        c2f = self.color_to_face()
        for c in pattern:
            if c not in c2f:
                raise Reject("spec", f"pattern colour {c!r} not on the block ({sorted(c2f)})")
        d.layout_template_id = pid
        # --- init: distinct up faces; remaining faces shown on side faces toward the robot -----------------------
        faces = list(self.faces)
        up_faces = [faces[i] for i in rng.permutation(len(faces))[:n]]
        remaining = [f for f in faces if f not in up_faces]
        front_dir = -st.inward  # side face normal that faces the robot (world xy)
        front_yaw = math.atan2(float(front_dir[1]), float(front_dir[0]))
        quats = []
        show = {}
        for i, up in enumerate(up_faces):
            # candidates: faces adjacent to `up` (not up, not opposite) that can point toward the robot
            side = None
            for f in remaining:
                if f != up and f != G.opposite_face(up) and f not in show.values():
                    side = f
                    break
            q = G.rotation_with_face_up(up, front_face=side) if side is not None else G.rotation_with_face_up(up)
            # rotation_with_face_up points `side` toward -x; rotate so that -x maps to front_dir
            q = G.qmul(G.quat_from_yaw(front_yaw - math.pi), q)
            if side is None:
                q = G.qmul(G.quat_from_yaw(float(rng.uniform(-0.35, 0.35))), q)
            else:
                show[i] = side
                q = G.qmul(G.quat_from_yaw(float(rng.uniform(-0.25, 0.25))), q)
            quats.append(q)
        if len(remaining) > len(show):
            missing = [f for f in remaining if f not in show.values()]
            if missing:
                raise Reject("layout", f"faces {missing} cannot be shown (not adjacent to any up face)")
        # positions: distinct reachable cells with >= gap between block footprints
        gap = float(self.gen("scatter_min_gap", 0.03))
        order = rng.permutation(len(self.cells))
        chosen = []
        for k in order:
            xy = self.cells[k]
            if all(float(th.linalg.norm(xy - c)) >= self.edge * math.sqrt(2) + gap for c in chosen):
                chosen.append(xy)
            if len(chosen) == n:
                break
        if len(chosen) < n:
            raise Reject("layout", "not enough separated reachable cells")
        poses = []
        for xy, q in zip(chosen, quats):
            pos = th.tensor([float(xy[0]), float(xy[1]), top.z + self.world.bottom_offset(self.hull, q) + 0.002])
            poses.append((pos, q))
        corners = th.cat([self.hull_corners_world(self.hull, p) for p in poses])
        if not self.frustum_ok(st.view.cam_pose, corners, 0.05):
            raise Reject("out_of_view")
        # visibility of the shown side faces: their normal must point toward the camera
        cam = st.view.cam_pose
        for i, f in show.items():
            nrm = G.face_normal_world(f, poses[i][1])
            to_cam = G.t(cam[0]) - poses[i][0]
            # foreshortening bound: the camera looks down at ~50-65 deg elevation, so a vertical side face turned toward
            # the robot is seen at that angle; anything past 75 deg is a sliver
            if float(th.dot(nrm, to_cam / th.linalg.norm(to_cam))) < math.cos(math.radians(float(self.gen("max_side_face_angle_deg", 75.0)))):
                raise Reject("layout", f"side face {f} of block {i} not facing the camera")
        for name, pose in zip(names, poses):
            d.object_poses[name] = pose
        d.robot_q = st.q_stance
        d.base_pose2d = st.base_pose2d
        # --- goal layout: anchored at the upper-left block (image space), aligned with the table --------------
        uv = th.stack([self.world.camera_model.project(cam, p[0].unsqueeze(0))[0][0] for p in poses])
        score = uv[:, 0] + uv[:, 1]  # upper-left = smallest u + v
        anchor_idx = int(th.argmin(score))
        anchor = poses[anchor_idx][0]
        ax, ay = top.local_axes()
        # table axes expressed relative to the robot: "right" = -tangent, "toward the robot" = -inward
        right = -st.tangent
        toward = -st.inward
        # snap right / toward onto the table's own axes so the arrangement is aligned with the table edges
        def snap(v):
            cands = [ax, -ax, ay, -ay]
            return max(cands, key=lambda c: float(th.dot(c, v)))
        right, toward = snap(right), snap(toward)
        table_yaw = math.atan2(float(right[1]), float(right[0]))  # block +x along the robot's right
        pitch = self.edge + float(self.gen("block_gap", 0.001))
        slots = {}
        for r in range(n_rows):
            for c in range(n_cols):
                idx = r * n_cols + c
                face = c2f[pattern[idx]]
                q = G.qmul(G.quat_from_yaw(table_yaw), G.rotation_with_face_up(face))
                xy = anchor[:2] + right * (c * pitch) + toward * (r * pitch)
                z = top.z + self.world.bottom_offset(self.hull, q) + 0.001
                slots[f"slot_{idx}"] = {"position": [float(xy[0]), float(xy[1]), z], "orientation": [float(v) for v in q]}
        relations = []
        sids = list(slots)
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                a, b = slots[sids[i]], slots[sids[j]]
                p_rel, q_rel = G.relative((th.tensor(b["position"]), th.tensor(b["orientation"])), (th.tensor(a["position"]), th.tensor(a["orientation"])))
                relations.append({"from": sids[i], "to": sids[j], "position": [float(v) for v in p_rel], "orientation": [float(v) for v in q_rel]})
        d.groups[GROUP_NAMES["blocks"]] = {
            "members": names,
            "assignment": {"mode": "exchangeable", "cost": "layout"},
            "orientation_equivalences": [],
            "target_slots": slots,
            "pose_tolerances": {"position": self.tol("position", 0.01), "orientation": self.tol("orientation", 0.15)},
            "layout_relations": relations,
            "layout_tolerances": {"position": self.tol("position", 0.01), "orientation": self.tol("orientation", 0.15)},
        }
        # goal feasibility: blocks at the slots (identity assignment)
        for name, sid in zip(names, sids):
            s = slots[sid]
            d.goal_poses[name] = (th.tensor(s["position"]), th.tensor(s["orientation"]))
        d.summary.update({"pattern": pid, "pattern_faces": pattern, "up_faces": up_faces, "shown_side_faces": show, "anchor_block": names[anchor_idx],
                          "table_yaw": table_yaw})
        return d

    def refine_targets_from_goal(self, draft: Draft):
        """Snap the slots to the settled goal poses (blocks may sink a fraction of a millimetre) and rebuild relations."""
        names = [b.name for b in self.bindings.many("blocks")]
        group = draft.groups[GROUP_NAMES["blocks"]]
        sids = list(group["target_slots"])
        for name, sid in zip(names, sids):
            p, q = self.objects[name].get_position_orientation()
            ref_q = th.tensor(group["target_slots"][sid]["orientation"])
            if G.quat_angle(q, ref_q) > 0.05 or float(th.linalg.norm(p - th.tensor(group["target_slots"][sid]["position"]))) > 0.005:
                raise Reject("goal_unstable", f"block {name} moved at the goal")
            group["target_slots"][sid] = {"position": p.tolist(), "orientation": q.tolist()}
        relations = []
        for i in range(len(sids)):
            for j in range(i + 1, len(sids)):
                a, b = group["target_slots"][sids[i]], group["target_slots"][sids[j]]
                p_rel, q_rel = G.relative((th.tensor(b["position"]), th.tensor(b["orientation"])), (th.tensor(a["position"]), th.tensor(a["orientation"])))
                relations.append({"from": sids[i], "to": sids[j], "position": [float(v) for v in p_rel], "orientation": [float(v) for v in q_rel]})
        group["layout_relations"] = relations

    def extra_checks(self, draft: Draft):
        # every block's pre-grasp must be reachable (generic top-down pinch) from the stance at some translate
        st, top = self.stance, self.top
        translates = []
        for name in draft.object_poses:
            p = draft.object_poses[name][0]
            centre = th.tensor([float(p[0]), float(p[1]), top.z + max(self.edge * 0.6, 0.03)])
            ok = None
            for pair in self.kit.grasp_candidates_at(centre, [st.base_pose2d[2], st.base_pose2d[2] + math.pi / 2], (0.0, 30.0), st.inward):
                ok = self.reachable_poses(st, pair)
                if ok is not None:
                    break
            if ok is None:
                raise Reject("grasp_unreachable", f"block {name} not reachable")
            translates.append(ok)
        draft.summary["block_reach_translates"] = translates
        draft.summary["view"] = {"translate": st.view.translate, "tilt": st.view.tilt}
