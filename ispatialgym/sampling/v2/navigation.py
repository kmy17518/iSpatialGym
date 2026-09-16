"""Navigation feasibility (spec §3.7), stance transitions (§3.8) and parking (§3.3) of the v2 sampler.

Pure at import time (no simulator): every function takes the ``RobotKit`` / ``SimWorld`` it needs, so the
enumeration helpers (``transition_standoffs``) are unit-tested without Isaac Sim.

§3.7 — stances lie 0.35–1.1 m from a face, inside BEHAVIOR's eroded band, so:

1. if the goal stance lies in the same connected region of the **locally un-eroded** map (the un-eroded floor map of
   the loaded floor, optionally restricted to the task rooms) as the start stance, the shortest grid path on that map is
   taken and validated with whole-body collision checks at the **navigation posture** (untucked arms, carried object
   attached, trunk as recorded) at 0.1 m strides (``mode = "local"``);
2. otherwise the eroded map is planned between the nearest eroded-free cells of the two stances and each stance is
   connected to the global path by a straight local segment, collision-checked the same way (``mode = "global"``).

§3.8 — the trunk change between two consecutive stances happens at the smallest transition stand-off
``s_T ∈ {s, s + 0.1, …, s + 0.5}`` (along -inward from the arriving stance) at which the swept trunk motion (5 samples
between the two trunk values, arms untucked, carried object attached) is whole-body collision-free.

§3.3 — parking: tucked at a free cell inside the task room's floor polygon, >= 1 m from every spawned object and support.
"""

from __future__ import annotations

import heapq
import logging
import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import torch as th

from ispatialgym.sampling.v2.stances import PathRecord, Stance, Transition, path_length

if TYPE_CHECKING:  # pragma: no cover
    from ispatialgym.sampling.robot_utils import RobotKit

log = logging.getLogger("ispatialgym.sampling.v2")

STRIDE_M = 0.1
ATTACH_FIT_SEED = 20260915
TRANSITION_STEP = 0.1
TRANSITION_EXTRA_MAX = 0.5
TRANSITION_SAMPLES = 5
PARK_MIN_DIST = 1.0


class NavigationError(Exception):
    """A path / transition could not be established (the caller turns it into ``Reject(<reason>)``)."""

    def __init__(self, reason: str, message: str = ""):
        super().__init__(f"[{reason}] {message}")
        self.reason = reason


# ----------------------------------------------------------------------------------------------------------------
# pure helpers (tested without a simulator)
# ----------------------------------------------------------------------------------------------------------------


def transition_standoffs(s: float, step: float = TRANSITION_STEP, extra_max: float = TRANSITION_EXTRA_MAX) -> List[float]:
    """§3.8 candidate set ``{s, s + 0.1, …, s + 0.5}`` (smallest first)."""
    n = int(round(extra_max / step))
    return [round(float(s) + k * step, 6) for k in range(n + 1)]


def trunk_sweep(trunk_from: Tuple[float, float], trunk_to: Tuple[float, float], n: int = TRANSITION_SAMPLES) -> List[Tuple[float, float]]:
    """``n`` trunk samples strictly between (and including) the two trunk values, linear in (translate, tilt)."""
    (t0, k0), (t1, k1) = trunk_from, trunk_to
    if n <= 1:
        return [(float(t1), float(k1))]
    return [(t0 + (t1 - t0) * i / (n - 1), k0 + (k1 - k0) * i / (n - 1)) for i in range(n)]


def stride_poses(waypoints: Sequence[Sequence[float]], stride: float = STRIDE_M) -> List[Tuple[float, float, float]]:
    """Base poses every ``stride`` metres along a polyline of (x, y[, yaw]) waypoints; the yaw is interpolated linearly
    (shortest way) between consecutive waypoint yaws, or taken along the segment when a waypoint has none."""
    out: List[Tuple[float, float, float]] = []
    wps = [tuple(float(v) for v in w) for w in waypoints]
    if not wps:
        return out
    for a, b in zip(wps[:-1], wps[1:]):
        seg = math.hypot(b[0] - a[0], b[1] - a[1])
        yaw_a = a[2] if len(a) > 2 else math.atan2(b[1] - a[1], b[0] - a[0])
        yaw_b = b[2] if len(b) > 2 else yaw_a
        dyaw = (yaw_b - yaw_a + math.pi) % (2 * math.pi) - math.pi
        n = max(1, int(math.ceil(seg / stride)))
        for i in range(n):
            s = i / n
            out.append((a[0] + (b[0] - a[0]) * s, a[1] + (b[1] - a[1]) * s, yaw_a + dyaw * s))
    last = wps[-1]
    out.append((last[0], last[1], last[2] if len(last) > 2 else out[-1][2] if out else 0.0))
    return out


def grid_shortest_path(free: th.Tensor, start: Tuple[int, int], goal: Tuple[int, int]) -> Optional[List[Tuple[int, int]]]:
    """A* on a boolean grid (8-connected, no corner cutting through blocked cells); None when disconnected."""
    h, w = free.shape
    if not (0 <= start[0] < h and 0 <= start[1] < w and 0 <= goal[0] < h and 0 <= goal[1] < w):
        return None
    if not bool(free[start]) or not bool(free[goal]):
        return None
    fr = free.numpy() if hasattr(free, "numpy") else free

    def hcost(a):
        return math.hypot(a[0] - goal[0], a[1] - goal[1])

    openq = [(hcost(start), 0.0, start)]
    came: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
    dist = {start: 0.0}
    nbrs = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0), (-1, -1, math.sqrt(2)), (-1, 1, math.sqrt(2)), (1, -1, math.sqrt(2)), (1, 1, math.sqrt(2))]
    while openq:
        _, d, cur = heapq.heappop(openq)
        if cur == goal:
            path = []
            while cur is not None:
                path.append(cur)
                cur = came[cur]
            return path[::-1]
        if d > dist.get(cur, float("inf")):
            continue
        for di, dj, c in nbrs:
            nb = (cur[0] + di, cur[1] + dj)
            if not (0 <= nb[0] < h and 0 <= nb[1] < w) or not fr[nb[0], nb[1]]:
                continue
            if di and dj and not (fr[cur[0] + di, cur[1]] and fr[cur[0], cur[1] + dj]):
                continue
            nd = d + c
            if nd < dist.get(nb, float("inf")):
                dist[nb] = nd
                came[nb] = cur
                heapq.heappush(openq, (nd + hcost(nb), nd, nb))
    return None


def simplify_path(cells: Sequence[Tuple[int, int]], free: th.Tensor) -> List[Tuple[int, int]]:
    """Greedy line-of-sight shortcutting of a grid path (Bresenham visibility on ``free``)."""
    if len(cells) <= 2:
        return list(cells)
    fr = free.numpy() if hasattr(free, "numpy") else free

    def visible(a, b):
        x0, y0 = a
        x1, y1 = b
        dx, dy = abs(x1 - x0), abs(y1 - y0)
        sx, sy = (1 if x1 > x0 else -1), (1 if y1 > y0 else -1)
        err = dx - dy
        x, y = x0, y0
        while True:
            if not fr[x, y]:
                return False
            if (x, y) == (x1, y1):
                return True
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

    out = [cells[0]]
    i = 0
    while i < len(cells) - 1:
        j = len(cells) - 1
        while j > i + 1 and not visible(cells[i], cells[j]):
            j -= 1
        out.append(cells[j])
        i = j
    return out


# ----------------------------------------------------------------------------------------------------------------
# maps
# ----------------------------------------------------------------------------------------------------------------


def local_map(kit: "RobotKit", rooms: Optional[Sequence[str]] = None, floor: int = 0, blocked: Sequence[Sequence[Sequence[float]]] = ()) -> th.Tensor:
    """The **locally un-eroded** map of §3.7: the un-eroded floor map of the loaded floor (255 = free), restricted to the
    task ``rooms`` (+ a one-cell dilation so doorways connect) and minus ``blocked`` footprints (polygons, world xy)."""
    trav = kit.world.scene.trav_map
    free = trav.floor_map[floor] == 255
    if rooms:
        seg = kit.world.scene.seg_map
        ids = [seg.room_ins_name_to_ins_id[r] for r in rooms if r in seg.room_ins_name_to_ins_id]
        if ids and seg.room_ins_map.shape == free.shape:
            mask = th.zeros_like(free)
            for i in ids:
                mask |= seg.room_ins_map == i
            k = 3
            mask = th.nn.functional.max_pool2d(mask.float()[None, None], k, stride=1, padding=k // 2)[0, 0] > 0.5
            free = free & mask
    if blocked:
        from ispatialgym.sampling.scene_variants import polygon_cells

        for poly in blocked:
            cells = polygon_cells(trav, poly, floor)
            if len(cells):
                free[cells[:, 0], cells[:, 1]] = False
    return free


def eroded_map(kit: "RobotKit", floor: int = 0) -> th.Tensor:
    return kit.eroded_trav_map(floor) == 255


def _to_map(kit: "RobotKit", xy) -> Tuple[int, int]:
    ij = kit.world.scene.trav_map.world_to_map(th.tensor([float(xy[0]), float(xy[1])]))
    return int(ij[0]), int(ij[1])


def _to_world(kit: "RobotKit", ij: Tuple[int, int]) -> th.Tensor:
    return kit.world.scene.trav_map.map_to_world(th.tensor([int(ij[0]), int(ij[1])]))


def nearest_free_cell(free: th.Tensor, ij: Tuple[int, int], max_radius: int = 15) -> Optional[Tuple[int, int]]:
    h, w = free.shape
    if 0 <= ij[0] < h and 0 <= ij[1] < w and bool(free[ij]):
        return ij
    for r in range(1, max_radius + 1):
        best = None
        for di in range(-r, r + 1):
            for dj in (-r, r) if abs(di) != r else range(-r, r + 1):
                a, b = ij[0] + di, ij[1] + dj
                if 0 <= a < h and 0 <= b < w and bool(free[a, b]):
                    d = di * di + dj * dj
                    if best is None or d < best[0]:
                        best = (d, (a, b))
        if best is not None:
            return best[1]
    return None


def same_region(free: th.Tensor, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    """Whether two cells are connected on ``free`` (flood fill, 4-connected)."""
    h, w = free.shape
    if not (bool(free[a]) and bool(free[b])):
        return False
    fr = free.numpy() if hasattr(free, "numpy") else free
    seen = {a}
    stack = [a]
    while stack:
        cur = stack.pop()
        if cur == b:
            return True
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nb = (cur[0] + di, cur[1] + dj)
            if 0 <= nb[0] < h and 0 <= nb[1] < w and nb not in seen and fr[nb[0], nb[1]]:
                seen.add(nb)
                stack.append(nb)
    return False


# ----------------------------------------------------------------------------------------------------------------
# whole-body checks along a path
# ----------------------------------------------------------------------------------------------------------------


def path_collision_free(kit: "RobotKit", poses: Sequence[Tuple[float, float, float]], trunk: Tuple[float, float], posture: str = "untucked",
                        margin: float = 0.01, q_arms: Optional[Dict[str, th.Tensor]] = None) -> Tuple[bool, int]:
    """Whole-body collision check (cuRobo, ``margin``) of the base poses at the navigation posture; the carried object
    must already be attached to the planner (``kit.backend.attach_object``). Returns (all free, first failing index)."""
    if not poses:
        return True, -1
    q0 = kit.q_config(arms=posture, translate=trunk[0], tilt=trunk[1], right_arm=(q_arms or {}).get("right"), left_arm=(q_arms or {}).get("left"))
    qs = th.stack([kit.q_at_base_pose(q0, x, y, yaw) for x, y, yaw in poses])
    free = kit.collision_free(qs, margin=margin)
    if bool(free.all()):
        return True, -1
    return False, int(th.nonzero(~free)[0])


def attached_checks_enabled() -> bool:
    """``ISG_ATTACHED_CHECKS=off`` (development switch, recorded in every summary as ``checks.attached_object``): the
    carried object is *not* attached for the checks. Exists because the robot model exempts the attached object from
    self-collision only against the finger links (README "Backend facts"), so with the current backend a grasped object
    deep in the gripper fails every attached check; the switch lets the rest of the pipeline be exercised until the
    model is fixed. It is not a sampler fallback: the setting is global, logged and written into the artefacts."""
    import os

    return os.environ.get("ISG_ATTACHED_CHECKS", "on").lower() not in ("off", "0", "false")


class attached:
    """Context manager: the carried object attached to the planner for the checks inside (§3.7 / §3.8 / §3.5).

    ``R1ProCuRobo.attach_object`` fits the object's collision meshes in the eef frame from the *current* relative pose
    of object and eef. While sampling the object rests on its support and the hands are untucked, so the recorded grasp
    transform ``eef_in_object`` (T_obj→eef of the drawn grasp) is applied first: the object is moved (gravity off) to
    ``eef_pose ∘ inv(T_obj→eef)``, attached, and put back exactly where it was. Without ``eef_in_object`` the current
    relative pose is used (the object is already in the hand)."""

    def __init__(self, kit: "RobotKit", arm: Optional[str], obj: Any, eef_in_object: Optional[Tuple[th.Tensor, th.Tensor]] = None):
        self.kit, self.arm, self.obj, self.eef_in_object = kit, arm, obj, eef_in_object
        if not attached_checks_enabled():
            self.obj = None
        self._saved = None

    def __enter__(self):
        if self.obj is None or self.arm is None:
            return self
        if self.eef_in_object is not None:
            import omnigibson as og

            from ispatialgym.sampling import geometry as G

            p, q = self.obj.get_position_orientation()
            self._saved = (p.clone(), q.clone(), bool(getattr(self.obj, "_gravity_enabled", True)))
            eef = self.kit.robot.eef_links[self.arm].get_position_orientation()
            obj_pose = G.compose((eef[0].float(), eef[1].float()), G.invert((G.t(self.eef_in_object[0]), G.t(self.eef_in_object[1]))))
            self.obj.disable_gravity()
            self.obj.set_position_orientation(obj_pose[0], obj_pose[1])
            self.obj.keep_still()
            og.sim.step_physics()
            self.kit.robot.keep_still()
            self.obj.keep_still()
        # the sphere fit of ``attach_object`` samples the mesh at random (5–7 spheres for a small object, a different set
        # per call): seed torch around it so every check of one attempt sees the same attached geometry
        cpu_state = th.get_rng_state()
        cuda_state = th.cuda.get_rng_state_all() if th.cuda.is_available() else None
        th.manual_seed(ATTACH_FIT_SEED)
        try:
            self.kit.backend.attach_object(self.arm, self.obj)
        finally:
            th.set_rng_state(cpu_state)
            if cuda_state is not None:
                th.cuda.set_rng_state_all(cuda_state)
        return self

    def __exit__(self, *exc):
        if self.obj is None or self.arm is None:
            return False
        self.kit.backend.detach_object(self.arm)
        if self._saved is not None:
            import omnigibson as og

            p, q, grav = self._saved
            self.obj.set_position_orientation(p, q)
            if grav:
                self.obj.enable_gravity()
            self.obj.keep_still()
            og.sim.step_physics()
            self.kit.robot.keep_still()
            self.obj.keep_still()
            self._saved = None
        return False


# ----------------------------------------------------------------------------------------------------------------
# §3.7 paths
# ----------------------------------------------------------------------------------------------------------------


def plan_path(kit: "RobotKit", stance_from: Stance, stance_to: Stance, posture: str = "untucked", carried: Optional[str] = None,
              carried_obj: Any = None, carried_arm: str = "right", rooms: Optional[Sequence[str]] = None, blocked: Sequence[Any] = (),
              max_length_m: Optional[float] = None, stride: float = STRIDE_M, trunk: Optional[Tuple[float, float]] = None) -> PathRecord:
    """§3.7: the base path from ``stance_from`` to ``stance_to`` (world (x, y, yaw) waypoints), ``mode`` ``local`` or
    ``global``, validated with whole-body checks every ``stride`` m at the navigation posture (``trunk`` defaults to
    the arriving stance's trunk) with ``carried_obj`` attached to ``carried_arm``. Raises ``NavigationError`` with
    reason ``no_path`` / ``path_collision`` / ``path_too_long``."""
    trunk = (stance_to.translate, stance_to.tilt) if trunk is None else trunk
    a_xy, b_xy = stance_from.base_xy, stance_to.base_xy
    if float(th.linalg.norm(a_xy - b_xy)) < 1e-3 and abs(stance_from.yaw - stance_to.yaw) < 1e-3:
        return PathRecord(stance_from.name, stance_to.name, "local", [tuple(stance_to.base_pose2d)], 0.0, posture, carried, trunk, stride, 0)
    free_local = local_map(kit, rooms, blocked=blocked)
    a, b = _to_map(kit, a_xy), _to_map(kit, b_xy)
    a_free, b_free = nearest_free_cell(free_local, a, 3), nearest_free_cell(free_local, b, 3)
    waypoints: Optional[List[Tuple[float, float, float]]] = None
    mode = "local"
    if a_free is not None and b_free is not None and same_region(free_local, a_free, b_free):
        cells = grid_shortest_path(free_local, a_free, b_free)
        if cells is not None:
            cells = simplify_path(cells, free_local)
            pts = [_to_world(kit, c) for c in cells[1:-1]]
            waypoints = [tuple(stance_from.base_pose2d)] + [(float(p[0]), float(p[1]), float("nan")) for p in pts] + [tuple(stance_to.base_pose2d)]
    if waypoints is None:
        mode = "global"
        free_g = eroded_map(kit)
        ga, gb = nearest_free_cell(free_g, a, 20), nearest_free_cell(free_g, b, 20)
        if ga is None or gb is None:
            raise NavigationError("no_path", f"no eroded-free cell near {stance_from.name} / {stance_to.name}")
        cells = grid_shortest_path(free_g, ga, gb)
        if cells is None:
            raise NavigationError("no_path", f"{stance_from.name} -> {stance_to.name}: disconnected on the eroded map")
        cells = simplify_path(cells, free_g)
        pts = [_to_world(kit, c) for c in cells]
        waypoints = [tuple(stance_from.base_pose2d)] + [(float(p[0]), float(p[1]), float("nan")) for p in pts] + [tuple(stance_to.base_pose2d)]
    # yaws of the intermediate waypoints: heading of the following segment
    filled: List[Tuple[float, float, float]] = []
    for i, w in enumerate(waypoints):
        yaw = w[2]
        if yaw != yaw:  # nan
            nxt = waypoints[min(i + 1, len(waypoints) - 1)]
            yaw = math.atan2(nxt[1] - w[1], nxt[0] - w[0]) if (nxt[0] != w[0] or nxt[1] != w[1]) else filled[-1][2]
        filled.append((float(w[0]), float(w[1]), float(yaw)))
    length = path_length(filled)
    if max_length_m is not None and length > max_length_m:
        raise NavigationError("path_too_long", f"{length:.2f} m > {max_length_m:.2f} m")
    poses = stride_poses(filled, stride)
    with attached(kit, carried_arm if carried_obj is not None else None, carried_obj):
        ok, bad = path_collision_free(kit, poses, trunk, posture)
    if not ok:
        raise NavigationError("path_collision", f"{mode} path {stance_from.name} -> {stance_to.name} collides at stride {bad}/{len(poses)} ({poses[bad]})")
    return PathRecord(stance_from.name, stance_to.name, mode, filled, length, posture, carried, trunk, stride, len(poses))


# ----------------------------------------------------------------------------------------------------------------
# §3.8 transitions
# ----------------------------------------------------------------------------------------------------------------


def transition(kit: "RobotKit", stance_from: Stance, stance_to: Stance, carried_obj: Any = None, carried_arm: str = "right",
               posture: str = "untucked", margin: float = 0.01, n_samples: int = TRANSITION_SAMPLES) -> Optional[Transition]:
    """§3.8: the smallest ``s_T ∈ transition_standoffs(s)`` (``s`` = the arriving stance's stand-off, driven back along
    -inward) at which the swept trunk motion (``n_samples`` trunks between the two stances' values, arms untucked,
    carried object attached) is whole-body collision-free. Returns the ``Transition`` or None (caller: ``Reject
    ("no_transition")``). Stances with equal trunks need no transition (returns one with ``stand_off = s``)."""
    tf, tt = (stance_from.translate, stance_from.tilt), (stance_to.translate, stance_to.tilt)
    s = float(stance_to.stand_off if stance_to.stand_off is not None else 0.0)
    if abs(tf[0] - tt[0]) < 1e-6 and abs(tf[1] - tt[1]) < 1e-6:
        return Transition(stance_from.name, stance_to.name, s, trunk_from=tf, trunk_to=tt, n_samples=0)
    inward = stance_to.inward
    x0, y0, yaw = stance_to.base_pose2d
    sweep = trunk_sweep(tf, tt, n_samples)
    with attached(kit, carried_arm if carried_obj is not None else None, carried_obj):
        for s_T in transition_standoffs(s):
            back = s_T - s
            bx, by = x0 - float(inward[0]) * back, y0 - float(inward[1]) * back
            if not kit.trav_free((bx, by)):
                continue
            qs = th.stack([kit.q_at_base_pose(kit.q_config(arms=posture, translate=tr, tilt=ti), bx, by, yaw) for tr, ti in sweep])
            if bool(kit.collision_free(qs, margin=margin).all()):
                return Transition(stance_from.name, stance_to.name, s_T, trunk_from=tf, trunk_to=tt, n_samples=n_samples)
    return None


# ----------------------------------------------------------------------------------------------------------------
# §3.3 parking
# ----------------------------------------------------------------------------------------------------------------


def park_pose(kit: "RobotKit", rooms: Sequence[str], keep_away: Sequence[Any] = (), min_dist: float = PARK_MIN_DIST, floor: int = 0) -> th.Tensor:
    """§3.3: a free (eroded) cell inside the task rooms' floor, >= ``min_dist`` from every AABB in ``keep_away``
    (spawned objects and supports), nearest to the rooms' centre among the admissible cells."""
    trav = kit.world.scene.trav_map
    free = kit.eroded_trav_map(floor) == 255
    seg = kit.world.scene.seg_map
    cells = th.nonzero(free)
    if len(cells) == 0:
        raise NavigationError("no_park", "no free traversable cell")
    if rooms and seg.room_ins_map.shape == free.shape:
        ids = [seg.room_ins_name_to_ins_id[r] for r in rooms if r in seg.room_ins_name_to_ins_id]
        if ids:
            in_room = th.zeros(len(cells), dtype=th.bool)
            for i in ids:
                in_room |= seg.room_ins_map[cells[:, 0], cells[:, 1]] == i
            if bool(in_room.any()):
                cells = cells[in_room]
    pts = trav.map_to_world(cells.float()) if len(cells) else th.zeros(0, 2)
    ok = th.ones(len(pts), dtype=th.bool)
    for obj in keep_away:
        try:
            lo, hi = obj.aabb
        except Exception:
            continue
        lo2, hi2 = lo[:2], hi[:2]
        d = th.maximum(th.maximum(lo2 - pts, pts - hi2), th.zeros_like(pts))
        ok &= th.linalg.norm(d, dim=1) >= min_dist
    if not bool(ok.any()):
        raise NavigationError("no_park", f"no cell >= {min_dist} m from the task objects inside {list(rooms)}")
    pts = pts[ok]
    centre = pts.mean(dim=0)
    return pts[int(th.argmin(th.linalg.norm(pts - centre, dim=1)))]
