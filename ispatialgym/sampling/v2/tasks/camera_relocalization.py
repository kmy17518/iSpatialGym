"""Camera relocalization (spec §6.1 standard, §6.2 obstructed) on ``SamplerV2`` (TODO D3): v2 port of the v1 sampler
(``sampling/tasks/camera_relocalization.py``, incl. the obstacle placement of ``8c3f56d``) with the path buckets, the
recorded path and the goal block of §7.

Init: base = random free cell of the **eroded** traversability map inside the task room + yaw ~ U(−π, π); trunk and
arms = JoyLo reset pose (trunk 0.5, arms zero, grippers open; §3.3 exception). Goal: a free eroded cell whose
shortest path from init (``get_shortest_path`` on the eroded map) has its length inside the instance's **path bucket**
(``generation.path_buckets``, half-open ranges ``(0, 3]``, ``(3, 5]``, ``(5, 10]`` m; the driver may request a bucket
through ``requested_bucket_m`` / ``generation.requested_bucket_m``, ``multiply_isg --path-buckets`` overrides the
bucket list, otherwise the bucket with the largest deficit among the instances of the template is drawn); trunk
(translate, tilt) ~ U(``goal_translate_range`` ∩ [0, 0.95]) × U(``goal_tilt_range``); the arms stay as at init unless
the goal trunk makes them self-collide → untucked (``goal.arms``); camera forward-axis angle init → goal ≤
``max_heading_deg``; goal render: ≥ ``min_goal_objects`` non-structural objects with ≥ ``min_object_pixels`` each (the
structural categories of §6.1 never count, nor does the robot), ≥ ``min_shared_visible_fraction`` of them visible from
the init; both configurations whole-body collision-free (``collision_margin_m``).

The recorded path (§3.7 / §7 ``paths[]``, ``mode = "global"``: both stances lie on the eroded map, so the eroded-map
shortest path *is* the path and the local connections are empty) is the dense eroded-map shortest path itself (one
waypoint per map cell, 0.1 m; the first / last waypoints are the exact base poses), its waypoints carry the heading of
the following segment (the goal yaw last), and it is whole-body collision-checked every 0.1 m at the navigation posture
(JoyLo reset arms, init trunk). The bucket is decided on the length of the recorded path (= the grid geodesic, what the
oracle drives).

Obstructed (§6.2): one obstacle (group ``obstacles``) stood upright on a cell of the middle 30–70 % of the free
shortest path so that the shortest path on the map **with its footprint blocked** (``blocked_footprints``, recorded in
the TRO scene block like ``freed_footprints`` and in the sidecar ``goal`` block) diverts by ≥ ``obstacle.min_detour_m``
without closing (corridor ≥ ``obstacle.min_corridor_width_m`` on one side), the new path stays inside the bucket, the
obstacle stands stably (settled ``obstacle.settle_steps``, drift ≤ ``obstacle.settle_drift_max_m``), init and goal
stay collision-free with it in the world, it is not a look-alike of an object the init or goal view counts (§3.12), and
its displacement is checked at the goal. **Visibility (owner decision 2026-09-15, spec §6.2 / ``e6a6b68``)**: the
obstacle must be inside the head camera frame (``obstacle.in_frame_margin``) at a waypoint of the new path **before the
path departs from the unobstructed shortest path** — one of the waypoints the blocked path shares with the path the
robot would have taken without the obstacle (the divergence waypoint included), with the base heading along the
*unobstructed* path (at the divergence waypoint: toward where it would have continued) and the JoyLo reset trunk;
seeing the obstacle only after the detour is not admissible. With several obstacles each one is checked against the
shortest path before it was placed. ``divergence_waypoint_index`` / ``view_waypoint_index`` (indices into the recorded
dense path) are recorded in the summary and in the sidecar ``paths[]`` entry.

TRO: ``pose_targets[agent]`` = the goal head-camera sensor pose (§8: 0.05 m / 0.10 rad, ``link`` = the sensor key),
measured at the goal configuration. Sidecar: ``stances.init`` and ``stances.goal`` (reset arms recorded as
``arms: []``), ``goal`` (base, trunk, arms ``same`` | ``untucked``), ``paths[init -> goal]`` (+ the obstacle view /
divergence facts), ``path_bucket_m``, ``language.objects`` (the obstacle).
"""

from __future__ import annotations

import logging
import math
import os
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

from ispatialgym.sampling import geometry as G
from ispatialgym.sampling import knowledge as K
from ispatialgym.sampling import scene_variants as SV
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.tasks.base import Draft
from ispatialgym.sampling.tro_writer import pose_target
from ispatialgym.sampling.v2 import navigation as NAV
from ispatialgym.sampling.v2.core import SamplerV2, register
from ispatialgym.sampling.v2.stances import PathRecord, Stance, path_length

from ispatialgym.activity.problem import AGENT_INSTANCE  # noqa: E402

log = logging.getLogger("ispatialgym.sampling.v2")

SUBTASK_STANDARD = "camera_relocalization-standard"
SUBTASK_OBSTRUCTED = "camera_relocalization-obstructed"
#: JoyLo reset arms (zero) = the kit's ``tucked`` mode (§3.3 exception for camera relocalization)
RESET_ARMS = "tucked"
#: §3.3: translate 1.0 self-collides in cuRobo; 0.95 is the deepest trunk everywhere
MAX_TRANSLATE = 0.95
#: §6.1 structural categories (substring match on the scene object's category): never counted as visible objects
STRUCTURAL_SUBSTRINGS = ("walls", "floors", "ceilings", "roof", "baseboard", "window", "door", "light", "lamp", "chandelier", "curtain", "drape", "electric_switch")
#: minimum init-goal base distance (a goal on the init cell is no relocalization)
MIN_EUCLIDEAN_M = 0.3
#: search budgets (not rule numbers): free points drawn per attempt, goal candidates rendered, obstacle cells tried
N_FREE_POINTS = 60
N_GOAL_TRIES = 12
N_YAW_TRIES = 3
OBSTACLE_MAX_CELLS = 8
#: obstacle displacement tolerances at the goal settle (§6.2 "displacement is checked at the goal settle")
OBSTACLE_MAX_TILT_RAD = 0.05
#: footprint growth of ``blocked_footprints`` (the 2 cm the chair variant's ``freed_footprints`` use, §3.1)
FOOTPRINT_MARGIN_M = 0.02


# ----------------------------------------------------------------------------------------------------------------
# pure helpers (tested without a simulator)
# ----------------------------------------------------------------------------------------------------------------


def normalise_buckets(buckets: Optional[Sequence[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """``generation.path_buckets`` (``[{"max_m": 3, "n": 10}, ...]``) sorted by ``max_m`` and validated (positive,
    strictly increasing upper bounds, ``n`` >= 0)."""
    if not buckets:
        raise ValueError("generation.path_buckets is empty: camera relocalization needs its buckets")
    out = sorted(({"max_m": float(b["max_m"]), "n": int(b.get("n", 0))} for b in buckets), key=lambda b: b["max_m"])
    prev = 0.0
    for b in out:
        if b["max_m"] <= prev:
            raise ValueError(f"path_buckets must have strictly increasing positive max_m, got {buckets}")
        if b["n"] < 0:
            raise ValueError(f"path_buckets n must be >= 0, got {buckets}")
        prev = b["max_m"]
    return out


def bucket_bounds(buckets: Sequence[Dict[str, Any]]) -> List[Tuple[float, float]]:
    """Half-open ranges ``(lo, hi]`` of the normalised buckets: ``(0, 3], (3, 5], (5, 10]``."""
    bounds, lo = [], 0.0
    for b in buckets:
        bounds.append((lo, float(b["max_m"])))
        lo = float(b["max_m"])
    return bounds


def bucket_index(length_m: float, buckets: Sequence[Dict[str, Any]]) -> int:
    """Index of the bucket whose ``(lo, hi]`` contains ``length_m``; ``-1`` when none does (0 or beyond the last)."""
    for i, (lo, hi) in enumerate(bucket_bounds(buckets)):
        if lo < length_m <= hi:
            return i
    return -1


def bucket_of(length_m: float, buckets: Sequence[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """The bucket (dict) of a path length, None when the length is outside every bucket."""
    i = bucket_index(length_m, buckets)
    return None if i < 0 else buckets[i]


def bucket_deficits(counts: Dict[float, int], buckets: Sequence[Dict[str, Any]]) -> List[int]:
    """``n − count`` per bucket (by ``max_m``), what ``multiply_isg --path-buckets`` still has to fill."""
    return [int(b["n"]) - int(counts.get(float(b["max_m"]), 0)) for b in buckets]


def choose_bucket(rng, buckets: Sequence[Dict[str, Any]], counts: Dict[float, int], requested_max_m: Optional[float] = None) -> int:
    """The bucket of the next instance: the requested one (by ``max_m``) when given, else uniformly among the buckets
    with a deficit, else uniformly among all."""
    if requested_max_m is not None:
        for i, b in enumerate(buckets):
            if abs(float(b["max_m"]) - float(requested_max_m)) < 1e-9:
                return i
        raise ValueError(f"requested bucket max_m={requested_max_m} is not one of {[b['max_m'] for b in buckets]}")
    open_ = [i for i, d in enumerate(bucket_deficits(counts, buckets)) if d > 0]
    pool = open_ or list(range(len(buckets)))
    return int(pool[int(rng.integers(len(pool)))])


def path_yaws(points: Sequence[Sequence[float]], final_yaw: float) -> List[float]:
    """Heading along a polyline at every point: the direction of the following segment, the goal yaw last
    (zero-length steps keep the previous heading)."""
    yaws: List[float] = []
    n = len(points)
    for i in range(n):
        if i == n - 1:
            yaws.append(float(final_yaw))
            continue
        dx, dy = float(points[i + 1][0]) - float(points[i][0]), float(points[i + 1][1]) - float(points[i][1])
        if math.hypot(dx, dy) < 1e-6:
            yaws.append(yaws[-1] if yaws else float(final_yaw))
        else:
            yaws.append(math.atan2(dy, dx))
    return yaws


def path_cells(rng, n_points: int, max_cells: int) -> List[int]:
    """Indices of the middle 30..70 % of a path (every interior index on short paths), shuffled, at most ``max_cells``."""
    if n_points < 3:
        return []
    lo, hi = int(math.floor(0.3 * (n_points - 1))), int(math.ceil(0.7 * (n_points - 1)))
    idx = list(range(max(lo, 1), min(hi, n_points - 2) + 1)) or list(range(1, n_points - 1))
    perm = rng.permutation(len(idx))
    return [idx[int(i)] for i in perm[:max_cells]]


def detour_failure(geo_blocked: Optional[float], geo_free: float, bucket_hi: float, min_detour_m: float) -> Optional[str]:
    """Why a blocked-map path is inadmissible (§6.2): ``disconnected`` / ``short_detour`` / ``too_long`` (leaves the
    bucket), None when it is."""
    if geo_blocked is None:
        return "disconnected"
    if float(geo_blocked) - float(geo_free) < float(min_detour_m):
        return "short_detour"
    if float(geo_blocked) > float(bucket_hi):
        return "too_long"
    return None


def divergence_index(path_before: Sequence[Sequence[float]], path_after: Sequence[Sequence[float]], tol: float) -> int:
    """Index of the last waypoint the two paths share (their common prefix: waypoints within ``tol`` of each other):
    the robot on ``path_after`` departs from ``path_before`` right after this waypoint. 0 when they part immediately
    (v1 ``_divergence_index``; ``tol`` = half a map cell)."""
    n = min(len(path_before), len(path_after))
    k = 0
    while k + 1 < n and math.hypot(float(path_before[k + 1][0]) - float(path_after[k + 1][0]), float(path_before[k + 1][1]) - float(path_after[k + 1][1])) <= tol:
        k += 1
    return k


def view_candidates(path: Sequence[Sequence[float]], final_yaw: float, path_before: Optional[Sequence[Sequence[float]]] = None,
                    tol: float = 0.05) -> Tuple[List[int], List[Tuple[float, float]], List[float], Optional[int]]:
    """The waypoints of ``path`` at which an obstacle may count as seen and the base heading there (§6.2): without
    ``path_before`` every waypoint with the heading of ``path``; with ``path_before`` (the shortest path the robot would
    take without the obstacle) only the waypoints **up to the divergence point**, with the headings of ``path_before``
    (at the divergence waypoint: toward where the unobstructed path would have continued). Returns ``(indices, xy
    points, yaws, divergence index or None)``."""
    pts = [(float(p[0]), float(p[1])) for p in path]
    if not pts:
        return [], [], [], None
    if path_before is None:
        return list(range(len(pts))), pts, path_yaws(pts, final_yaw), None
    k = divergence_index(path_before, path, tol)
    before = [(float(p[0]), float(p[1])) for p in path_before]
    yaws = path_yaws(before, final_yaw)[: k + 1]
    return list(range(k + 1)), pts[: k + 1], yaws, k


def first_admissible_view(hits: Sequence[int], divergence: Optional[int]) -> Optional[int]:
    """The first waypoint index in ``hits`` (indices of the blocked path at which the obstacle is in frame) that lies
    on or before the divergence waypoint; None when the obstacle is seen only after the detour (or never)."""
    ok = sorted(int(h) for h in hits if divergence is None or int(h) <= int(divergence))
    return ok[0] if ok else None


def is_structural(category: Optional[str]) -> bool:
    """§6.1 object counting: structural categories never count."""
    if not category:
        return True
    c = category.lower()
    return c in K.STRUCTURE_CATEGORIES or any(s in c for s in STRUCTURAL_SUBSTRINGS)


def heading_between(axis0, axis1) -> float:
    """Angle (rad) between two camera forward axes."""
    a, b = G.t(axis0), G.t(axis1)
    return math.acos(max(-1.0, min(1.0, float(th.dot(a, b) / (th.linalg.norm(a) * th.linalg.norm(b))))))


class ObstructedPathRecord(PathRecord):
    """``paths[]`` entry of the obstructed variant: the blocked-map path plus the §6.2 view facts per obstacle
    (``view_waypoint_index`` — the first waypoint at which the obstacle is in frame before the departure from the
    unobstructed path —, ``divergence_waypoint_index``, ``view_pose2d`` = that waypoint with the unobstructed heading;
    indices into ``waypoints``). The F6 ``Path`` keeps them in ``extra``."""

    def __init__(self, base: PathRecord, view: Optional[Dict[str, Any]] = None):
        super().__init__(base.frm, base.to, base.mode, list(base.waypoints), base.length_m, base.posture, base.carried, base.trunk, base.stride_m, base.n_checked)
        self.view: Dict[str, Any] = dict(view or {})

    def to_sidecar(self) -> Dict[str, Any]:
        d = super().to_sidecar()
        d.update(self.view)
        return d


# ----------------------------------------------------------------------------------------------------------------
# sampler
# ----------------------------------------------------------------------------------------------------------------


@register(SUBTASK_STANDARD, SUBTASK_OBSTRUCTED)
class CameraRelocalizationV2(SamplerV2):
    """One class for both variants; the obstructed rules run when the entry has the ``obstacles`` role."""

    def __init__(self, ctx):
        super().__init__(ctx)
        self.q_reset_rel: Optional[th.Tensor] = None
        self.requested_bucket_m: Optional[float] = None  # the driver may set it (multiply_isg --path-buckets fill)
        self.buckets_written: Dict[float, int] = {}
        self.obst_hull: Dict[str, th.Tensor] = {}
        self.obst_geom: Dict[str, Dict[str, Any]] = {}
        self._scene_aabbs: List[Tuple[str, th.Tensor, th.Tensor]] = []
        self.floor_z = float(self.world.floor_z)

    # ------------------------------------------------------------------ generation numbers

    def g(self, key: str, default=None):
        """``generation.<key>`` else the per-subtask seed of ``generation_defaults`` (the v1 facade lacks the v2 keys)."""
        v = self.gen(key)
        if v is not None:
            return v
        from ispatialgym.sampling import generation_defaults as GD

        d = GD.defaults_for(self.subtask)
        if key in d:
            return d[key]
        if default is not None:
            return default
        raise Reject("spec", f"{self.gp.alias}: generation.{key} missing and no default in generation_defaults")

    @property
    def obstructed(self) -> bool:
        return "obstacles" in self.gp.roles

    @property
    def obstacle_names(self) -> List[str]:
        return [b.name for b in self.bindings.many("obstacles")] if self.bindings.roles.get("obstacles") else []

    def obstacle_params(self) -> Dict[str, Any]:
        return dict(self.g("obstacle"))

    def buckets(self) -> List[Dict[str, Any]]:
        try:
            return normalise_buckets(self.g("path_buckets"))
        except ValueError as e:
            raise Reject("spec", str(e))

    # ------------------------------------------------------------------ frames / configurations

    def room(self) -> Optional[str]:
        req = self.choice.bindings.get("floor")
        if isinstance(req, dict) and req.get("room"):
            return req["room"]
        rooms = self.gp.rooms(self.scene)
        return rooms[0] if rooms else None

    def in_room(self, xy) -> bool:
        room = self.room()
        if room is None:
            return True
        return self.world.scene.seg_map.get_room_instance_by_point(G.t(xy)[:2]) == room

    def init_trunk(self) -> Tuple[float, float]:
        ip = self.g("init_pose")
        return float(ip.get("trunk", 0.5)), 0.0

    def config(self, base2d, translate: float, tilt: float, arms: str = RESET_ARMS) -> th.Tensor:
        return self.kit.q_at_base_pose(self.kit.q_config(arms=arms, translate=translate, tilt=tilt), *base2d)

    def self_collides(self, q: th.Tensor) -> bool:
        """Whether ``q`` collides with itself (world contacts excluded): the §6.1 arm rule at the goal trunk."""
        self.kit._ensure_synced()
        with_self = bool(self.kit.backend.check_collision(q.unsqueeze(0), emb_name="whole_body", self_collision=True, margin=0.0)[0])
        without = bool(self.kit.backend.check_collision(q.unsqueeze(0), emb_name="whole_body", self_collision=False, margin=0.0)[0])
        return with_self and not without

    # ------------------------------------------------------------------ template

    def sample_template(self):
        tr, ti = self.init_trunk()
        self.q_reset_rel = self.kit.q_config(arms=RESET_ARMS, translate=tr, tilt=ti)
        self.template_meta["init_arms"] = "JoyLo reset (arms zero)"
        self.template_meta["init_config"] = {"trunk": [tr, ti], "arms": "zero"}
        self.template_facts["path_buckets"] = self.buckets()
        self.template_facts["bucket_counts_existing"] = {str(k): v for k, v in self.bucket_counts().items()}
        if self.obstructed:
            if not self.obstacle_names:
                raise Reject("spec", f"{self.gp.alias}: the obstructed variant needs the obstacles role bound")
            self.prepare_obstacles()

    # ------------------------------------------------------------------ buckets (§2.3 / §9)

    def bucket_counts(self) -> Dict[float, int]:
        """Instances per bucket (``max_m``) of this template: the sidecars already written plus this run's."""
        counts: Dict[float, int] = dict(self.buckets_written)
        d = os.path.join(self.ctx.out_dir, "instances")
        if os.path.isdir(d):
            import json

            for f in os.listdir(d):
                if not f.endswith(".oracle.json"):
                    continue
                try:
                    with open(os.path.join(d, f)) as fh:
                        b = json.load(fh).get("path_bucket_m")
                except (OSError, ValueError):
                    continue
                if b is not None:
                    counts[float(b)] = counts.get(float(b), 0) + 1
        return counts

    def bucket_of(self, length_m: float) -> Optional[Dict[str, Any]]:
        return bucket_of(length_m, self.buckets())

    # ------------------------------------------------------------------ instance (§6.1)

    def sample_instance(self, rng: np.random.Generator) -> Draft:
        d = Draft()
        buckets = self.buckets()
        requested = self.requested_bucket_m if self.requested_bucket_m is not None else self.gen("requested_bucket_m")
        try:
            bi = choose_bucket(rng, buckets, self.bucket_counts(), requested)
        except ValueError as e:
            raise Reject("spec", str(e))
        lo, hi = bucket_bounds(buckets)[bi]
        max_h = math.radians(float(self.g("max_heading_deg")))
        tr_lo, tr_hi = self.g("goal_translate_range")
        tr_hi = min(float(tr_hi), MAX_TRANSLATE)
        ti_lo, ti_hi = self.g("goal_tilt_range")
        min_objs = int(self.g("min_goal_objects"))
        min_px = int(self.g("min_object_pixels"))
        min_shared = float(self.g("min_shared_visible_fraction"))
        margin = float(self.g("collision_margin_m"))
        tr0, ti0 = self.init_trunk()
        pts = self.kit.random_free_points(rng, N_FREE_POINTS, room_filter=self.in_room)
        if len(pts) < 2:
            raise Reject("no_stance", "not enough free eroded cells in the task room")
        # init: reset pose, collision free
        init = None
        for xy in pts:
            base = (float(xy[0]), float(xy[1]), float(rng.uniform(-math.pi, math.pi)))
            q = self.kit.q_at_base_pose(self.q_reset_rel, *base)
            if bool(self.kit.collision_free(q, margin=margin)[0]):
                init = (base, q)
                break
        if init is None:
            raise Reject("no_stance", "no collision-free init pose")
        base0, q0 = init
        cam0 = self.kit.camera_pose(q0)
        axis0 = self.world.camera_model.optical_axis(cam0)
        # goal candidates: analytic filters (bucket upper bound on the Euclidean distance, heading), collision later
        cands = []
        for xy in pts:
            e = math.hypot(float(xy[0]) - base0[0], float(xy[1]) - base0[1])
            if e < MIN_EUCLIDEAN_M or e > hi:
                continue
            for _ in range(N_YAW_TRIES):
                yaw1 = base0[2] + float(rng.uniform(-max_h, max_h))
                tr1 = float(rng.uniform(float(tr_lo), tr_hi))
                ti1 = float(rng.uniform(float(ti_lo), float(ti_hi)))
                base1 = (float(xy[0]), float(xy[1]), yaw1)
                q1 = self.config(base1, tr1, ti1)
                arms = "same"
                if self.self_collides(q1):
                    q1 = self.config(base1, tr1, ti1, arms="untucked")
                    arms = "untucked"
                cam1 = self.kit.camera_pose(q1)
                heading = heading_between(axis0, self.world.camera_model.optical_axis(cam1))
                if heading > max_h:
                    continue
                cands.append((e, heading, base1, tr1, ti1, q1, cam1, arms))
        if not cands:
            raise Reject("goal_sampling", f"no goal candidate passes the analytic filters (bucket ({lo}, {hi}] m, heading <= {math.degrees(max_h):.0f} deg)")
        order = rng.permutation(len(cands))
        chosen = None
        n_rendered = 0
        fails: Dict[str, int] = {}
        for i in order[:N_GOAL_TRIES]:
            e, heading, base1, tr1, ti1, q1, cam1, arms = cands[int(i)]
            if not bool(self.kit.collision_free(q1, margin=margin)[0]):
                fails["goal_collision"] = fails.get("goal_collision", 0) + 1
                continue
            path, geo = self.kit.geodesic(th.tensor(base0[:2]), th.tensor(base1[:2]))
            if geo is None or path is None or len(path) < 2:
                fails["no_path"] = fails.get("no_path", 0) + 1
                continue
            rec = self.build_path_record(path, base0, base1, (tr0, ti0))
            if rec is None:
                fails["path_collision"] = fails.get("path_collision", 0) + 1
                continue
            if bucket_index(rec.length_m, buckets) != bi:
                fails["bucket"] = fails.get("bucket", 0) + 1
                continue
            # renders: goal first (needs >= min_objs non-structural objects), then init (shared fraction)
            self.kit.apply(q1, base1)
            goal_vis = self.non_structural_visible(min_px)
            n_rendered += 1
            if len(goal_vis) < min_objs:
                fails["goal_objects"] = fails.get("goal_objects", 0) + 1
                continue
            self.kit.apply(q0, base0)
            init_vis = self.non_structural_visible(min_px)
            shared = len(set(goal_vis) & set(init_vis)) / max(len(goal_vis), 1)
            if shared < min_shared:
                fails["shared"] = fails.get("shared", 0) + 1
                continue
            chosen = (e, heading, base1, tr1, ti1, q1, cam1, arms, geo, goal_vis, init_vis, shared, path, rec)
            break
        if chosen is None:
            raise Reject("goal_sampling", f"no goal candidate passed collision / path / bucket / visibility ({n_rendered} rendered; {fails})")
        e, heading, base1, tr1, ti1, q1, cam1, arms, geo, goal_vis, init_vis, shared, path, rec = chosen
        self.kit.apply(q0, base0)
        d.robot_q = q0
        d.base_pose2d = base0
        d.settle_free = []
        d.extras.update({"goal_robot_q": q1, "goal_base_pose2d": base1, "goal_cam": cam1, "goal_trunk": (tr1, ti1), "goal_arms": arms,
                         "bucket": {"index": bi, "lo_m": lo, "hi_m": hi}})
        d.summary.update({"euclidean_m": e, "geodesic_m": float(geo), "path_length_m": rec.length_m, "heading_deg": math.degrees(heading), "goal_base_pose2d": list(base1),
                          "goal_trunk": [tr1, ti1], "goal_arms": arms, "goal_visible_objects": goal_vis, "init_visible_objects": init_vis, "shared_fraction": shared,
                          "path_bucket_m": hi, "bucket": {"lo_m": lo, "hi_m": hi}, "candidate_failures": fails})
        # sidecar: the two stances, the goal block, the path, the bucket
        st_init = Stance("init", base0, tr0, ti0, arms=[])
        st_init.q = q0
        st_init.extras["posture"] = "JoyLo reset (arms zero)"
        st_goal = Stance("goal", base1, tr1, ti1, arms=[] if arms == "same" else "untucked")
        st_goal.q = q1
        self.sidecar.add_stance(st_init)
        self.sidecar.add_stance(st_goal)
        self.sidecar.set_goal(base1, tr1, ti1, arms=arms)
        self.sidecar.add_path(rec)
        self.sidecar.path_bucket_m = hi
        if self.obstructed:
            self.place_obstacles(d, rng, base0, q0, cam0, base1, q1, cam1, path, float(geo), rec, lo, hi)
        return d

    def goal(self, draft: Draft):
        cam1 = draft.extras["goal_cam"]
        draft.pose_targets[AGENT_INSTANCE] = pose_target(position=G.t(cam1[0]).tolist(), orientation=G.t(cam1[1]).tolist(), position_metric="full",
                                                         orientation_metric="full", position_tolerance=self.tol("position", 0.05),
                                                         orientation_tolerance=self.tol("orientation", 0.10), link=self.world.cam_key)

    # ------------------------------------------------------------------ the recorded path (§3.7 on the eroded map)

    @staticmethod
    def dense_points(path: th.Tensor, base0, base1) -> List[th.Tensor]:
        """The waypoints of an eroded-map shortest path as recorded: BEHAVIOR's dense cell-centre path with the first /
        last point replaced by the exact base positions (same count, so view / divergence indices line up)."""
        pts = [G.t(p)[:2].clone() for p in path]
        if len(pts) < 2:
            return [G.t(base0[:2]), G.t(base1[:2])]
        pts[0], pts[-1] = G.t(base0[:2]), G.t(base1[:2])
        return pts

    def build_path_record(self, path: th.Tensor, base0, base1, trunk: Tuple[float, float], frm: str = "init", to: str = "goal") -> Optional[PathRecord]:
        """The ``paths[]`` record of an eroded-map shortest path: the dense path (one waypoint per 0.1 m map cell), waypoint
        yaws = heading of the following segment (init yaw first, goal yaw last), whole-body collision-checked at 0.1 m
        strides at the navigation posture (reset arms, init trunk). None when a stride collides."""
        pts = self.dense_points(path, base0, base1)
        yaws = path_yaws(pts, base1[2])
        yaws[0] = float(base0[2])
        waypoints = [(float(p[0]), float(p[1]), float(y)) for p, y in zip(pts, yaws)]
        length = path_length(waypoints)
        poses = NAV.stride_poses(waypoints, NAV.STRIDE_M)
        ok, bad = NAV.path_collision_free(self.kit, poses, trunk, posture=RESET_ARMS)
        if not ok:
            log.info("%s: path %s -> %s collides at stride %d/%d (%s)", self.gp.alias, frm, to, bad, len(poses), [round(v, 2) for v in poses[bad]])
            return None
        return PathRecord(frm, to, "global", waypoints, length, posture=RESET_ARMS, carried=None, trunk=trunk, stride_m=NAV.STRIDE_M, n_checked=len(poses))

    def map_tol(self) -> float:
        """Divergence tolerance: half a map cell (v1)."""
        return 0.5 * float(self.world.scene.trav_map.map_resolution) + 1e-6

    # ------------------------------------------------------------------ visibility (§6.1 object counting)

    def non_structural_visible(self, min_pixels: int) -> Dict[str, int]:
        px = self.world.visible_pixels(exclude_structure=True)
        out = {}
        for name, n in px.items():
            if n < min_pixels or name == self.robot.name:
                continue
            o = self.world.scene.object_registry("name", name)
            if o is not None and is_structural(getattr(o, "category", None)):
                continue
            out[name] = int(n)
        return out

    # ------------------------------------------------------------------ goal feasibility (agent target)

    def check_goal_feasible(self, draft: Draft):
        """The robot at the goal configuration: the realised sensor pose is the target (the FK prediction is checked
        against it), the obstacles are not displaced, every binary goal atom holds; restore afterwards."""
        state = self.world.snapshot()
        before = {n: tuple(x.clone() for x in self.objects[n].get_position_orientation()) for n in draft.object_poses}
        try:
            self.kit.apply(draft.extras["goal_robot_q"], draft.extras["goal_base_pose2d"])
            self.world.step_physics(2, still=[self.robot])
            cam = self.world.camera_pose()
            tgt = draft.pose_targets[AGENT_INSTANCE]
            err_p = float(th.linalg.norm(G.t(cam[0]) - th.tensor(tgt["position"])))
            err_o = G.quat_angle(G.t(cam[1]), th.tensor(tgt["orientation"]))
            if err_p > 0.5 * self.tol("position", 0.05) or err_o > 0.5 * self.tol("orientation", 0.10):
                raise Reject("goal_fk_mismatch", f"FK camera pose off the simulated one by {err_p * 1000:.1f} mm / {math.degrees(err_o):.1f} deg")
            tgt["position"], tgt["orientation"] = G.t(cam[0]).tolist(), G.t(cam[1]).tolist()  # the measured sensor pose is the target
            drift_max = float(self.obstacle_params()["settle_drift_max_m"]) if self.obstructed else 0.005
            for name, (p0, r0) in before.items():
                p, r = self.objects[name].get_position_orientation()
                if float(th.linalg.norm(p - p0)) > drift_max or G.quat_angle(r, r0) > OBSTACLE_MAX_TILT_RAD:
                    raise Reject("obstacle", f"{name} displaced by the goal configuration ({float(th.linalg.norm(p - p0)) * 1000:.1f} mm)")
            self.configure_targets(draft)
            holds, status = self.binary_goal_holds()
            if not holds:
                failed = [c["bddl"] for c in self.goal_report(status) if not c["satisfied"]]
                raise Reject("goal_infeasible", f"unsatisfied at the goal configuration: {failed}")
            draft.summary["goal_fk_error"] = {"position_m": err_p, "orientation_rad": err_o}
            draft.summary["goal_check"] = {"goal": self.goal_report(status)}
        finally:
            self.world.restore(state)
            self.robot.keep_still()
            self.kit.sync_world()
            self.configure_targets(draft)

    def refine_targets_from_goal(self, draft: Draft):
        pass

    # ------------------------------------------------------------------ obstacles (§6.2)

    def prepare_obstacles(self):
        """Per-template geometry of the obstacles (collision hull in the canonical +z-up pose), the §6.2 size
        admissibility, and the AABBs of the non-structural scene objects an obstacle may not intersect."""
        P = self.obstacle_params()
        z_floor = float(self.obj("floor").aabb[1][2])
        if abs(z_floor - float(self.world.floor_z)) < 0.2:
            self.floor_z = z_floor
        fp_lo, fp_hi = P["footprint_range_m"]
        h_lo, h_hi = P["height_range_m"]
        for name in self.obstacle_names:
            obj = self.objects[name]
            hull = self.world.local_hull(obj)
            if len(hull) == 0:
                raise Reject("spec", f"{name} ({obj.category}/{obj.model}) has no collision geometry")
            pts = G.local_points_world(hull, (th.zeros(3), G.IDENTITY_QUAT))  # canonical +z-up pose
            lo, hi = pts.min(dim=0).values, pts.max(dim=0).values
            fp = [float(hi[0] - lo[0]), float(hi[1] - lo[1])]
            height = float(hi[2] - lo[2])
            if not (fp_lo <= min(fp) and max(fp) <= fp_hi) or not (h_lo <= height <= h_hi):
                raise Reject("obstacle_inadmissible", f"{obj.category}/{obj.model}: footprint {fp[0]:.2f}x{fp[1]:.2f} m, height {height:.2f} m outside "
                                                      f"[{fp_lo}, {fp_hi}] / [{h_lo}, {h_hi}] (§6.2)")
            self.obst_hull[name] = hull
            self.obst_geom[name] = {"scene_object_name": obj.name, "category": obj.category, "model": obj.model, "bottom_offset_m": -float(lo[2]),
                                    "height_m": height, "footprint_m": fp, "radius_m": float(pts[:, :2].norm(dim=1).max())}
        self._scene_aabbs = []
        for o in self.world.scene.objects:
            if o is self.robot or o.name in self.world.spawned or o.category in K.STRUCTURE_CATEGORIES:
                continue
            lo, hi = o.aabb
            self._scene_aabbs.append((o.name, lo.clone(), hi.clone()))
        self.template_meta["obstacles"] = {"instances": dict(self.obst_geom), "generation": P, "per_instance": {}}
        log.info("%s: obstacles %s (%d scene AABBs to avoid, floor z %.3f)", self.gp.alias,
                 {n: f"{g['category']}/{g['model']} {g['footprint_m'][0]:.2f}x{g['footprint_m'][1]:.2f}x{g['height_m']:.2f} m" for n, g in self.obst_geom.items()},
                 len(self._scene_aabbs), self.floor_z)

    def park_obstacle(self, name: str):
        obj = self.objects[name]
        self.world.park(obj, list(self.world.spawned).index(obj.name))

    def obstacle_pose(self, name: str, xy, yaw: float) -> G.Pose:
        quat = G.quat_from_yaw(yaw)
        z = self.floor_z + self.world.bottom_offset(self.obst_hull[name], quat) + 0.003
        return th.tensor([float(xy[0]), float(xy[1]), z]), quat

    def obstacle_aabb(self, name: str, pose: G.Pose) -> Tuple[th.Tensor, th.Tensor]:
        pts = self.world.hull_at(self.obst_hull[name], pose)
        return pts.min(dim=0).values, pts.max(dim=0).values

    def footprints(self, poses: Dict[str, G.Pose], margin: float) -> Dict[str, List]:
        return SV.blocked_footprints([self.obstacle_aabb(n, p) for n, p in poses.items()], margin)

    def blocked_cells(self, footprints: Dict[str, List]) -> set:
        trav = self.world.scene.trav_map
        out = set()
        for poly in footprints.get("block", []):
            out.update(map(tuple, SV.polygon_cells(trav, poly).tolist()))
        return out

    def intersecting_object(self, lo: th.Tensor, hi: th.Tensor) -> Optional[str]:
        for name, olo, ohi in self._scene_aabbs:
            if bool((hi >= olo).all()) and bool((lo <= ohi).all()):
                return name
        return None

    @contextmanager
    def blocked_map(self, footprints: Dict[str, List]):
        """Plan on a copy of the traversability map with ``footprints`` blocked (``floor_map[0]`` swapped in for the
        duration; the kit's eroded-map cache never sees the obstacle)."""
        trav = self.world.scene.trav_map
        orig = trav.floor_map[0]
        trav.floor_map[0] = orig.clone()
        try:
            SV.block_trav_map(self.world.scene, footprints)
            yield trav.floor_map[0]
        finally:
            trav.floor_map[0] = orig

    def view_indices(self, path: th.Tensor, final_yaw: float, corners: th.Tensor, margin: float, path_before: Optional[th.Tensor] = None) -> List[int]:
        """Waypoints of ``path`` at which the whole box ``corners`` is inside the head-camera image with ``margin``, the
        robot at the reset trunk / arms and the base facing along the path (§6.2: the oracle navigates first).

        With ``path_before`` (the shortest path the robot would take without the obstacle) only the waypoints **up to
        the divergence point** count (owner decision 2026-09-15): the obstacle must be seen before the robot's path
        departs from the unobstructed path, otherwise a policy could learn the detour without ever seeing its cause
        (e.g. an obstacle that only enters the frame at the last waypoint, when the base turns to the goal yaw). The
        heading at those waypoints is the heading of ``path_before`` (at the divergence waypoint: toward where the
        unobstructed path would have continued)."""
        idx, pts, yaws, _ = view_candidates(path, final_yaw, path_before, self.map_tol())
        if not idx:
            return []
        qs = th.stack([self.kit.q_at_base_pose(self.q_reset_rel, float(p[0]), float(p[1]), yaw) for p, yaw in zip(pts, yaws)])
        cams = self.kit.camera_poses_batch(qs)
        return [i for i, cam in zip(idx, cams) if self.frustum_ok(cam, corners, margin)]

    def view_record(self, poses: Dict[str, G.Pose], path_b: th.Tensor, final_yaw: float, margin: float, ref_paths: Dict[str, th.Tensor]) -> Dict[str, Any]:
        """Per obstacle the waypoints of ``path_b`` at which it is inside the head-camera frame **before the path departs
        from the obstacle's reference path** (the shortest path before that obstacle was placed; at least one, else
        Reject), the divergence index and the view pose (waypoint + unobstructed heading)."""
        view, div, pose2d = {}, {}, {}
        for name, pose in poses.items():
            ref = ref_paths[name]
            hits = self.view_indices(path_b, final_yaw, G.aabb_corners(*self.obstacle_aabb(name, pose)), margin, path_before=ref)
            k = divergence_index(ref, path_b, self.map_tol())
            div[name] = int(k)
            first = first_admissible_view(hits, k)
            if first is None:
                raise Reject("obstacle", f"{name} is not inside the head camera frame at any waypoint before the path departs from the unobstructed path "
                                         f"(divergence at waypoint {k} of {len(path_b)})")
            view[name] = [int(h) for h in hits]
            yaw = path_yaws([G.t(p)[:2] for p in ref], final_yaw)[first]
            pose2d[name] = [float(path_b[first][0]), float(path_b[first][1]), float(yaw)]
        return {"view_waypoint_index": {n: h[0] for n, h in view.items()}, "view_waypoints": view, "divergence_waypoint_index": div, "view_pose2d": pose2d}

    @staticmethod
    def sidecar_view(rec: Dict[str, Any]) -> Dict[str, Any]:
        """The view facts written into the sidecar ``paths[]`` entry (indices into the recorded dense path)."""
        return {"view_waypoint_index": dict(rec["view_waypoint_index"]), "divergence_waypoint_index": dict(rec["divergence_waypoint_index"]),
                "view_pose2d": {n: list(p) for n, p in rec["view_pose2d"].items()}, "view_heading": "unobstructed path", "view_trunk": "JoyLo reset"}

    def corridor_widths(self, blocked_map: th.Tensor, xy: th.Tensor, direction: th.Tensor, max_m: float = 3.0) -> List[float]:
        """Free floor run (m) on the blocked un-eroded map on either side of the obstacle, perpendicular to the path
        direction at its cell: from the obstacle centre outward, past its own footprint, up to the first blocked cell."""
        trav = self.world.scene.trav_map
        res = float(trav.map_resolution)
        if float(th.linalg.norm(direction)) < 1e-6:
            direction = th.tensor([1.0, 0.0])
        perp = th.tensor([-float(direction[1]), float(direction[0])])
        perp = perp / th.linalg.norm(perp)
        out = []
        for sign in (1.0, -1.0):
            run, started = 0.0, False
            for k in range(1, int(max_m / res) + 1):
                ij = trav.world_to_map(G.t(xy)[:2] + perp * (sign * k * res))
                i, j = int(ij[0]), int(ij[1])
                free = 0 <= i < blocked_map.shape[0] and 0 <= j < blocked_map.shape[1] and bool(blocked_map[i, j] == 255)
                if not started:
                    started = free
                    run = res if free else 0.0
                    continue
                if not free:
                    break
                run += res
            out.append(round(run, 3))
        return out

    def lookalike_of_visible(self, name: str, visible: Sequence[str]) -> Optional[str]:
        """§6.2 / §3.12: the obstacle must not be a look-alike of an object the init or goal view counts."""
        obj = self.objects[name]
        for other in visible:
            o = self.world.scene.object_registry("name", other)
            if o is None or o is obj:
                continue
            if self.gp.lookalikes.lookalikes((obj.category, obj.model), (o.category, o.model)):
                return other
        return None

    def try_obstacle_cell(self, name: str, xy, yaw: float, direction: th.Tensor, placed: Dict[str, G.Pose], base0, q0, base1, q1, geo_before: float,
                          bucket_hi: float, P: Dict[str, Any], trunk: Tuple[float, float], path_before: Optional[th.Tensor] = None):
        """One placement candidate: (record, None) on success (the obstacle stands settled in the world), (None,
        failure tag) otherwise (parked again). Geometric filters, settle, one A* on the blocked map, corridor, the view
        before the departure from ``path_before`` (the shortest path without this obstacle), the recorded path's
        whole-body check, init / goal collision checks with the obstacle."""
        xy = G.t(xy)[:2]
        geom = self.obst_geom[name]
        margin = float(P.get("footprint_margin_m", FOOTPRINT_MARGIN_M))
        if not self.in_room(xy):
            return None, "room"
        keep_out = geom["radius_m"] + margin + self.kit.BASE_ROOM_RADIUS + 0.10
        if any(float(th.linalg.norm(xy - th.tensor(b[:2]))) < keep_out for b in (base0, base1)):
            return None, "near_robot"
        pose = self.obstacle_pose(name, xy, yaw)
        lo, hi = self.obstacle_aabb(name, pose)
        fp = SV.aabb_footprint(lo, hi, margin)
        if not SV.footprint_free(self.world.scene.trav_map, fp):
            return None, "floor"
        if self.intersecting_object(lo, hi) is not None:
            return None, "aabb"
        if any(G.polygons_overlap(th.tensor(fp), th.tensor(SV.aabb_footprint(*self.obstacle_aabb(n, p), margin))) for n, p in placed.items()):
            return None, "aabb"
        obj = self.objects[name]
        self.world.place(obj, xy, pose[1], self.floor_z, local_hull=self.obst_hull[name])
        err = self.world.settle([obj], steps=int(P["settle_steps"]))
        p1, r1 = obj.get_position_orientation()
        drift, tilt, dz = float(th.linalg.norm(p1[:2] - pose[0][:2])), G.quat_angle(r1, pose[1]), float(p1[2] - pose[0][2])
        if err or drift > float(P["settle_drift_max_m"]) or tilt > OBSTACLE_MAX_TILT_RAD or abs(dz) > 0.02:
            self.park_obstacle(name)
            return None, "unstable"
        pose = (p1.clone(), r1.clone())
        fps = self.footprints({**placed, name: pose}, margin)
        rec_path = None
        with self.blocked_map(fps) as blocked:
            path_b, geo_b = self.kit.geodesic(th.tensor(base0[:2]), th.tensor(base1[:2]))
            corridor = self.corridor_widths(blocked, pose[0][:2], direction)
            why = detour_failure(geo_b, geo_before, bucket_hi, float(P["min_detour_m"]))
            hits: List[int] = []
            if why is None and max(corridor) < float(P["min_corridor_width_m"]):
                why = "corridor"
            if why is None:
                # the obstacle must be seen before the robot departs from the path it would have taken without it
                hits = self.view_indices(path_b, base1[2], G.aabb_corners(*self.obstacle_aabb(name, pose)), float(P["in_frame_margin"]), path_before=path_before)
                if not hits:
                    why = "view"
            if why is None:
                self.kit.sync_world()  # the obstacle enters the collision world
                rec_path = self.build_path_record(path_b, base0, base1, trunk)
                if rec_path is None:
                    why = "path_collision"
                elif rec_path.length_m > bucket_hi:
                    why = "too_long"
        if why is None:
            qs = th.stack([q0, q1])
            free = self.kit.collision_free(qs, margin=float(self.g("collision_margin_m"))).tolist()
            if not all(free):
                why = "collision_init" if not free[0] else "collision_goal"
        if why is not None:
            self.park_obstacle(name)
            self.kit.sync_world()
            return None, why
        return {"pose": pose, "path": path_b, "geodesic_m": float(geo_b), "corridor_m": corridor, "view_indices": hits, "yaw": yaw, "settle_drift_m": drift,
                "settle_tilt_rad": tilt, "path_record": rec_path, "footprints": fps}, None

    def place_obstacles(self, d: Draft, rng, base0, q0, cam0: G.Pose, base1, q1, cam1: G.Pose, path_free: th.Tensor, geo_free: float, free_rec: PathRecord,
                        lo: float, hi: float):
        """Stand every obstacle on the current shortest path (each one must divert it by ``min_detour_m``), then
        record the blocked-map path (replacing the free one in the sidecar), the footprints and the view indices."""
        P = self.obstacle_params()
        names = self.obstacle_names
        trunk = self.init_trunk()
        visible = list(d.summary.get("goal_visible_objects", {})) + list(d.summary.get("init_visible_objects", {}))
        for name in names:
            other = self.lookalike_of_visible(name, visible)
            if other is not None:
                raise Reject("obstacle_lookalike", f"{name} is a look-alike of {other}, which the init / goal view counts (§3.12)")
        poses: Dict[str, G.Pose] = {}
        placements: Dict[str, Dict[str, Any]] = {}
        ref_paths: Dict[str, th.Tensor] = {}  # per obstacle: the shortest path before it was placed (the view rule's reference)
        cur_path, cur_geo = path_free, geo_free
        last: Optional[Dict[str, Any]] = None
        fails: Dict[str, int] = {}
        n_tried = 0
        for name in names:
            placed = None
            for idx in path_cells(rng, len(cur_path), OBSTACLE_MAX_CELLS):
                n_tried += 1
                xy = cur_path[idx]
                yaw = float(rng.uniform(-math.pi, math.pi))
                direction = cur_path[min(idx + 1, len(cur_path) - 1)] - cur_path[max(idx - 1, 0)]
                rec, why = self.try_obstacle_cell(name, xy, yaw, direction, poses, base0, q0, base1, q1, cur_geo, hi, P, trunk, path_before=cur_path)
                if rec is None:
                    fails[why] = fails.get(why, 0) + 1
                    continue
                placed = rec
                break
            if placed is None:
                raise Reject("obstacle", f"no cell of the shortest path admits {name} ({n_tried} cells tried: {fails}; free path {geo_free:.2f} m, {len(cur_path)} waypoints)")
            poses[name] = placed["pose"]
            ref_paths[name] = cur_path
            corners = G.aabb_corners(*self.obstacle_aabb(name, placed["pose"]))
            placements[name] = {"position": [round(float(v), 4) for v in placed["pose"][0]], "orientation": [round(float(v), 5) for v in placed["pose"][1]],
                                "yaw": placed["yaw"], "corridor_m": placed["corridor_m"], "settle_drift_m": round(placed["settle_drift_m"], 4),
                                "settle_tilt_rad": round(placed["settle_tilt_rad"], 4), "detour_m": round(placed["geodesic_m"] - cur_geo, 3),
                                "in_view_at_init": self.frustum_ok(cam0, corners, float(P["in_frame_margin"])),
                                "in_view_at_goal": self.frustum_ok(cam1, corners, float(P["in_frame_margin"]))}
            cur_path, cur_geo, last = placed["path"], placed["geodesic_m"], placed
        # the final path (all obstacles blocked) replaces the free-map path in the sidecar; the bucket is re-decided on it, and
        # every obstacle must be seen before the departure from *its* reference path along this final path
        rec_path: PathRecord = last["path_record"]
        if bucket_index(rec_path.length_m, self.buckets()) != d.extras["bucket"]["index"]:
            raise Reject("obstacle", f"blocked-map path {rec_path.length_m:.2f} m leaves the bucket ({lo}, {hi}]")
        view = self.view_record(poses, cur_path, base1[2], float(P["in_frame_margin"]), ref_paths)
        for name in names:
            placements[name].update({"view_waypoint_index": view["view_waypoint_index"][name], "divergence_waypoint_index": view["divergence_waypoint_index"][name]})
        rec_path = ObstructedPathRecord(rec_path, self.sidecar_view(view))
        self.sidecar.paths[:] = [p for p in self.sidecar.paths if not (p.frm == "init" and p.to == "goal")]
        self.sidecar.add_path(rec_path)
        fps = last["footprints"]
        for name, pose in poses.items():
            d.object_poses[name] = pose
        d.settle_free = list(names)
        d.extras["blocked_footprints"] = fps
        d.extras["obstacle_planned_poses"] = {n: (p[0].clone(), p[1].clone()) for n, p in poses.items()}
        d.extras["obstacle_path"] = (cur_path, cur_geo)
        d.extras["obstacle_ref_paths"] = ref_paths
        d.summary.update({"geodesic_blocked_m": cur_geo, "path_length_m": rec_path.length_m, "free_path_length_m": free_rec.length_m,
                          "obstacles": {"placements": placements, "cells_tried": n_tried, "cell_failures": fails, "free_geodesic_m": float(geo_free),
                                        "blocked_geodesic_m": float(cur_geo), "detour_m": float(cur_geo) - float(geo_free), "n_waypoints_blocked": int(len(cur_path)),
                                        "path_free": [[round(float(v), 3) for v in p] for p in path_free], **view}})
        self.sidecar.goal["blocked_footprints"] = fps
        for name in names:
            o = self.objects[name]
            self.sidecar.describe(name, o.category.replace("_", " "), descriptor=f"the {o.category.replace('_', ' ')} on the way")
        log.info("%s: %d obstacle(s) placed after %d cells (%s): free path %.2f m -> blocked %.2f m (recorded %.2f m), seen at waypoint %s (divergence %s) of %d",
                 self.gp.alias, len(names), n_tried, fails, geo_free, cur_geo, rec_path.length_m, view["view_waypoint_index"], view["divergence_waypoint_index"], len(cur_path))

    def after_settle(self, draft: Draft):
        """Obstructed: the settled obstacle must not have moved (≤ drift / tilt); the footprints, path and view indices
        follow the settled poses (re-planned only when the settled footprint blocks other map cells)."""
        if not draft.object_poses:
            return
        P = self.obstacle_params()
        margin = float(P.get("footprint_margin_m", FOOTPRINT_MARGIN_M))
        planned = draft.extras["obstacle_planned_poses"]
        for name, (p0, r0) in planned.items():
            p, r = draft.object_poses[name]
            drift, tilt = float(th.linalg.norm(G.t(p)[:2] - p0[:2])), G.quat_angle(r, r0)
            if drift > float(P["settle_drift_max_m"]) or tilt > OBSTACLE_MAX_TILT_RAD:
                raise Reject("obstacle", f"{name} moved {drift * 1000:.1f} mm / {tilt:.3f} rad during the settle")
        fps = self.footprints(draft.object_poses, margin)
        base1 = draft.extras["goal_base_pose2d"]
        ref_paths = draft.extras["obstacle_ref_paths"]
        if self.blocked_cells(fps) != self.blocked_cells(draft.extras["blocked_footprints"]):
            base0 = draft.base_pose2d
            hi = draft.extras["bucket"]["hi_m"]
            with self.blocked_map(fps):
                path_b, geo_b = self.kit.geodesic(th.tensor(base0[:2]), th.tensor(base1[:2]))
                why = detour_failure(geo_b, float(draft.summary["geodesic_m"]), hi, float(P["min_detour_m"]))
                if why:
                    raise Reject("obstacle", f"blocked-map path after the settle {why}")
                self.kit.sync_world()
                rec = self.build_path_record(path_b, base0, base1, self.init_trunk())
            if rec is None or bucket_index(rec.length_m, self.buckets()) != draft.extras["bucket"]["index"]:
                raise Reject("obstacle", "re-planned path after the settle collides or leaves the bucket")
            draft.extras["obstacle_path"] = (path_b, float(geo_b))
            draft.summary.update({"geodesic_blocked_m": float(geo_b), "path_length_m": rec.length_m})
            draft.summary["obstacles"]["replanned_after_settle"] = True
        else:
            path_b = draft.extras["obstacle_path"][0]
            rec = next(p for p in self.sidecar.paths if p.frm == "init" and p.to == "goal")
            draft.summary["obstacles"]["replanned_after_settle"] = False
        # the view facts follow the settled poses (and the re-planned path): each obstacle before the departure from its
        # reference path, along the final path
        view = self.view_record(draft.object_poses, path_b, base1[2], float(P["in_frame_margin"]), ref_paths)
        for name in draft.object_poses:
            draft.summary["obstacles"]["placements"][name].update({"view_waypoint_index": view["view_waypoint_index"][name],
                                                                   "divergence_waypoint_index": view["divergence_waypoint_index"][name]})
        draft.summary["obstacles"].update(view)
        self.sidecar.paths[:] = [p for p in self.sidecar.paths if not (p.frm == "init" and p.to == "goal")]
        self.sidecar.add_path(ObstructedPathRecord(rec, self.sidecar_view(view)))
        draft.extras["blocked_footprints"] = fps
        self.sidecar.goal["blocked_footprints"] = fps

    # ------------------------------------------------------------------ writing

    def tro_meta_block(self, draft: Draft, instance_id: str, seed: int) -> Dict[str, Any]:
        meta = super().tro_meta_block(draft, instance_id, seed)
        fps = draft.extras.get("blocked_footprints")
        if fps:
            sv = dict(meta["scene"].get("scene_variant") or {})
            sv["blocked_footprints"] = fps
            meta["scene"]["scene_variant"] = sv
        return meta

    def write_instance(self, draft: Draft, index: int, seed: int) -> str:
        path = super().write_instance(draft, index, seed)
        b = float(draft.summary["path_bucket_m"])
        self.buckets_written[b] = self.buckets_written.get(b, 0) + 1
        self.template_meta.setdefault("bucket_counts", {})[str(b)] = self.template_meta.get("bucket_counts", {}).get(str(b), 0) + 1
        if draft.extras.get("blocked_footprints"):
            obs = draft.summary["obstacles"]
            self.template_meta["obstacles"]["per_instance"][self.instance_id(index)] = {
                "blocked_footprints": draft.extras["blocked_footprints"], "free_geodesic_m": obs["free_geodesic_m"], "blocked_geodesic_m": obs["blocked_geodesic_m"],
                "view_waypoint_index": obs["view_waypoint_index"], "divergence_waypoint_index": obs["divergence_waypoint_index"],
                "positions": {n: [round(float(v), 4) for v in p[0]] for n, p in draft.object_poses.items()}}
        return path

    def record_constraints(self, draft: Draft) -> List[Dict[str, Any]]:
        """The confirmed goal trunk as a soft ``stance`` prior of the room (scope: scene)."""
        tr, ti = draft.extras["goal_trunk"]
        return [{"scope": {"scene": self.scene}, "type": "stance", "edges": [], "stand_off": [0.0, 0.0], "lateral": [0.0, 0.0],
                 "trunk": {"translate": [tr, tr], "tilt": [ti, ti]}, "path_bucket_m": draft.summary["path_bucket_m"], "hard": False}]

    def on_reject(self, err: Reject):
        super().on_reject(err)  # parks the obstacles, detaches planner objects
        if self.obstacle_names:
            self.kit.sync_world()  # the next attempt's collision checks must not see a phantom obstacle
