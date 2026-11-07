"""Interior compartments of the INTERIOR reach-map kind (i7-lean 2026-11-07, owner decision; prompt I item 7).

An :class:`InteriorSpec` states one compartment of a support's interior (a bookcase cube) as boxes in the reach map's frame (the base on
the floor, +x ahead, +y the robot's left, z up from the floor), the compartment's front edge at the map's edge distance ``d_e``:

* the compartment's FRONT plane (its walls' front faces: the support raster's open edge the stance stand-off is measured from) at
  ``x = d_e``, its back at ``d_e + depth_m``;
* ``board`` - the SUPPORT board, its top at ``height_m``, from ``board_setback_m`` behind the front plane to the back, ``long_m`` wide,
  ``board_thickness_m`` thick: the one POLICED support (the fingers' 2 mm allowance, ``contact_policy`` rules 2-3);
* ``above`` - the board above (headroom): its underside ``headroom_m`` above the board top, ``top_thickness_m`` thick, as deep and wide;
* ``wall_left`` / ``wall_right`` - the side walls, inner faces ``width_m`` apart, ``wall_thickness_m`` thick, from the board to the
  board above, front plane to back;
* ``back`` - the back panel (``back_thickness_m``; None = open: the compartment is open at its front and back, ``open_sides``);
* ``extra_boxes`` (i7-lean 2026-11-07, the fridge) - further NORMAL obstacles as oriented boxes ``(name, centre, half, yaw)`` in the
  compartment frame (x from the front plane into the compartment, y from the compartment's lateral centre, z from the board top, yaw
  about z): an OPEN DOOR at the spec's ``joints`` (the support's joint values the spec was measured at, e.g. ``j_left_door`` 2.3);
* ``edge_distances`` - the stand-offs (base to the front plane) its maps are computed at, ``centre_y_m`` - where a STABLE rest's map puts
  the compartment's lateral centre in the base frame (0: a stance square to the opening's centre; a LEAN rest's compartment follows
  the cell instead);

every box but the board is a NORMAL obstacle (every robot part clear: ``contact_policy.check_obstacles`` / ``spheres_clear``). The geometry
fields have one :meth:`InteriorSpec.digest`, recorded in the map's identity (meta ``interior``, ``reach_maps.map_key``) and checked by
``ReachMap.find(kind="interior", interior=)``. Laterally the compartment follows the map cell (``reach_maps.compute_map``): the object at
the cell leans against the wall on its lean side (the lean rest's ``board_face_y_m``), so a cell is "the book at this offset from the
robot, leaning on its board".

The registry (:data:`REGISTRY`, ``ispatialgym/assets/interiors/interiors.json``) names the specs and the supports they describe
(``supports``: scene + support instance; lookup only, not geometry) with their provenance (``source``: measured template facts, the
asset's metadata, what is assumed). The sampler reads the spec of the support it works on (:func:`spec_for_support`) and checks the live
interior raster against it (:func:`live_mismatch`: board height, headroom, open sides).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ispatialgym import REPO_ROOT

SCHEMA = "isg-interior/1"
REGISTRY = os.path.join(REPO_ROOT, "ispatialgym", "assets", "interiors", "interiors.json")
#: the live board height may differ this much from the spec's and still be served by its maps (the anchors' +-1 cm range)
HEIGHT_TOL_M = 0.01
#: the live headroom (median over the raster's board cells) vs the spec's
HEADROOM_TOL_M = 0.02
#: ``SimWorld.support_interior`` measures a cell's headroom from 5 mm above the board (its up-ray start): board top to the underside above
#: = the raster's headroom + this
RASTER_HEADROOM_OFFSET_M = 0.005
GEOMETRY_FIELDS = ("height_m", "depth_m", "width_m", "headroom_m", "board_thickness_m", "top_thickness_m", "wall_thickness_m", "long_m",
                   "board_setback_m", "back_thickness_m", "centre_y_m")
#: a spec's joint values match the live support's within this (rad / m)
JOINT_TOL = 0.02
FRAME = ("reach-map frame (base on the floor: +x ahead, +y left, z up from the floor); the compartment's front edge at x = d_e, its "
         "lateral centre at the cell's y + the lean rest's board_face_y_m - side x width / 2 (the object leans on its side wall)")


@dataclass(frozen=True)
class InteriorSpec:
    name: str
    height_m: float
    depth_m: float
    width_m: float
    headroom_m: float
    board_thickness_m: float = 0.04
    top_thickness_m: float = 0.04
    wall_thickness_m: float = 0.028
    long_m: float = 1.2
    board_setback_m: float = 0.0
    back_thickness_m: Optional[float] = None
    centre_y_m: float = 0.0
    edge_distances: Tuple[float, ...] = (0.35, 0.45, 0.55)
    joints: Tuple[Tuple[str, float], ...] = ()
    extra_boxes: Tuple[Tuple[str, Tuple[float, float, float], Tuple[float, float, float], float], ...] = ()
    supports: Tuple[Tuple[str, str], ...] = ()
    source: Dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def geometry(self) -> Dict[str, Any]:
        g = {k: (None if getattr(self, k) is None else round(float(getattr(self, k)), 4)) for k in GEOMETRY_FIELDS}
        return {"schema": SCHEMA, **g, "open_sides": self.open_sides, "edge_distances": [round(float(v), 4) for v in self.edge_distances],
                "joints": {k: round(float(v), 4) for k, v in self.joints},
                "extra_boxes": [{"name": n, "centre": [round(float(v), 4) for v in c], "half": [round(float(v), 4) for v in h], "yaw": round(float(y), 4)}
                                for n, c, h, y in self.extra_boxes]}

    @property
    def joint_values(self) -> Dict[str, float]:
        return {k: float(v) for k, v in self.joints}

    @property
    def open_sides(self) -> List[str]:
        return ["front"] + (["back"] if self.back_thickness_m is None else [])

    def digest(self) -> str:
        from ispatialgym.grasping import schema as S

        return S.digest(self.geometry(), 12)

    def to_json(self) -> Dict[str, Any]:
        return {**self.geometry(), "name": self.name, "digest": self.digest(), "supports": [list(s) for s in self.supports], "source": dict(self.source),
                "frame": FRAME, "boxes": "board (policed support), above, wall_left, wall_right" + ("" if self.back_thickness_m is None else ", back")}

    @staticmethod
    def from_json(d: Dict[str, Any], name: Optional[str] = None) -> "InteriorSpec":
        kw = {k: d[k] for k in GEOMETRY_FIELDS if k in d and d[k] is not None or k == "back_thickness_m" and k in d}
        if "edge_distances" in d:
            kw["edge_distances"] = tuple(float(v) for v in d["edge_distances"])
        kw["joints"] = tuple(sorted((str(k), float(v)) for k, v in (d.get("joints") or {}).items()))
        kw["extra_boxes"] = tuple((str(b["name"]), tuple(float(v) for v in b["centre"]), tuple(float(v) for v in b["half"]), float(b.get("yaw", 0.0)))
                                  for b in (d.get("extra_boxes") or []))
        return InteriorSpec(name=str(name or d.get("name") or "interior"), supports=tuple(tuple(s) for s in d.get("supports", [])),
                            source=dict(d.get("source") or {}), **kw)

    # -------------------------------------------------------------------------------------------- boxes (compartment frame)

    def boxes(self, h: float, d_e: float) -> List[Tuple[str, Any]]:
        """``[(name, contact_policy.Support box)]`` with the compartment centred at y = 0, its front edge at x = ``d_e`` and the board top
        at z = ``h``; the board first (the policed support)."""
        from ispatialgym.grasping import contact_policy as CP

        x = float(d_e) + 0.5 * float(self.depth_m)
        hd = 0.5 * float(self.depth_m)
        xb = float(d_e) + 0.5 * (float(self.board_setback_m) + float(self.depth_m))  # the boards start board_setback_m behind the front
        hb = 0.5 * (float(self.depth_m) - float(self.board_setback_m))
        hr = float(self.headroom_m)
        out = [("board", CP.Support.box([xb, 0.0, h - 0.5 * self.board_thickness_m], [hb, 0.5 * self.long_m, 0.5 * self.board_thickness_m], name="interior_board")),
               ("above", CP.Support.box([xb, 0.0, h + hr + 0.5 * self.top_thickness_m], [hb, 0.5 * self.long_m, 0.5 * self.top_thickness_m], name="interior_above"))]
        for nm, sg in (("wall_left", 1.0), ("wall_right", -1.0)):
            out.append((nm, CP.Support.box([x, sg * (0.5 * self.width_m + 0.5 * self.wall_thickness_m), h + 0.5 * hr],
                                           [hd, 0.5 * self.wall_thickness_m, 0.5 * hr], name=f"interior_{nm}")))
        if self.back_thickness_m is not None:
            out.append(("back", CP.Support.box([float(d_e) + float(self.depth_m) + 0.5 * self.back_thickness_m, 0.0, h + 0.5 * hr],
                                               [0.5 * self.back_thickness_m, 0.5 * self.width_m + self.wall_thickness_m, 0.5 * hr], name="interior_back")))
        for nm, c, hf, yaw in self.extra_boxes:  # the open door(s): oriented, at the spec's joint values
            out.append((nm, CP.Support.box([float(d_e) + float(c[0]), float(c[1]), float(h) + float(c[2])], [float(v) for v in hf], yaw=float(yaw),
                                           name=f"interior_{nm}")))
        return out

    def policed(self, h: float, d_e: float):
        return self.boxes(h, d_e)[0][1]

    def board_x_range(self, d_e: float) -> Tuple[float, float]:
        """The x span of the support board (where an object can stand): ``[d_e + board_setback_m, d_e + depth_m]``."""
        return float(d_e) + float(self.board_setback_m), float(d_e) + float(self.depth_m)

    def obstacles(self, h: float, d_e: float) -> Tuple[Any, ...]:
        return tuple(b for _, b in self.boxes(h, d_e)[1:])


def compartment_offsets(cell_y, board_face_y_m: float, side_sign: float, width_m: float):
    """The compartment's lateral centre per cell (the lean-side wall's inner face at ``cell_y + board_face_y_m``)."""
    return cell_y + float(board_face_y_m) - float(side_sign) * 0.5 * float(width_m)


def fits(width_m: float, board_face_y_m: float, side_sign: float, far_extent_y: float, margin: float = 0.0) -> bool:
    """The object (its root-relative extent toward the far wall ``far_extent_y``) stands between the walls."""
    return float(side_sign) * (float(board_face_y_m) - float(far_extent_y)) <= float(width_m) - float(margin)


# ------------------------------------------------------------------------------------------------ the registry


def _read_registry(p: str) -> Dict[str, InteriorSpec]:
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        d = json.load(f)
    if d.get("schema") != SCHEMA:
        raise ValueError(f"{p}: schema {d.get('schema')!r}, expected {SCHEMA}")
    return {name: InteriorSpec.from_json(s, name) for name, s in (d.get("interiors") or {}).items()}


def load_registry(path: Optional[str] = None) -> Dict[str, InteriorSpec]:
    """The registry: ``path``, else the repo's :data:`REGISTRY` plus ``ISPATIALGYM_INTERIORS`` (a campaign's registry file, e.g. the
    measure CLI's output; its entries add to the repo's and replace those of the same name)."""
    if path is not None:
        return _read_registry(path)
    out = _read_registry(REGISTRY)
    extra = os.environ.get("ISPATIALGYM_INTERIORS")
    if extra:
        out.update(_read_registry(extra))
    return out


def spec_named(name_or_path: str) -> InteriorSpec:
    """A registry spec by name (:func:`load_registry`), ``<registry.json>::<name>``, a registry file holding one spec, or a spec json file
    (its fields + ``name``)."""
    if "::" in name_or_path:
        path, nm = name_or_path.split("::", 1)
        reg = _read_registry(path)
        if nm not in reg:
            raise KeyError(f"no interior spec {nm!r} in {path} (has {sorted(reg)})")
        return reg[nm]
    if os.path.isfile(name_or_path):
        with open(name_or_path) as f:
            d = json.load(f)
        if "interiors" in d:
            reg = _read_registry(name_or_path)
            if len(reg) != 1:
                raise KeyError(f"{name_or_path} holds {sorted(reg)}: name one with <path>::<name>")
            return next(iter(reg.values()))
        return InteriorSpec.from_json(d)
    reg = load_registry()
    if name_or_path not in reg:
        raise KeyError(f"no interior spec {name_or_path!r} in {REGISTRY} (has {sorted(reg)})")
    return reg[name_or_path]


def joints_match(spec_joints: Dict[str, float], live: Optional[Dict[str, float]], tol: float = JOINT_TOL) -> bool:
    """The spec's joint values and the live support's (``partner_joint_init`` applied) name the same joints at the same values."""
    live = {str(k): float(v) for k, v in (live or {}).items()}
    return set(spec_joints) == set(live) and all(abs(spec_joints[k] - live[k]) <= tol for k in spec_joints)


def spec_for_support(scene: Optional[str], support: Optional[str], height: float, tol: float = HEIGHT_TOL_M,
                     registry: Optional[Dict[str, InteriorSpec]] = None, joints: Optional[Dict[str, float]] = None) -> Optional[InteriorSpec]:
    """The spec describing ``support`` of ``scene`` whose board is within ``tol`` of ``height`` (m above the floor) at the support's
    joint values ``joints`` (none = its scene state; a spec measured with a door open never serves the closed door, nor the reverse), else
    None."""
    reg = load_registry() if registry is None else registry
    best = None
    for spec in reg.values():
        if not any(sc == scene and nm == support for sc, nm in spec.supports):
            continue
        if not joints_match(spec.joint_values, joints):
            continue
        d = abs(float(spec.height_m) - float(height))
        if d <= tol and (best is None or d < best[0]):
            best = (d, spec)
    return None if best is None else best[1]


def live_mismatch(spec: InteriorSpec, top: Any, floor_z: float = 0.0, joints: Optional[Dict[str, float]] = None) -> Optional[str]:
    """Why the live interior raster ``top`` (``SimWorld.support_interior``) is not ``spec``'s compartment, or None: the board height
    (:data:`HEIGHT_TOL_M`), the median headroom of its board cells (:data:`HEADROOM_TOL_M`, when the raster has one) and the open sides
    (a raster with one open edge has a back panel, two: open front and back) and the joints (i7-lean: the door's value)."""
    if not joints_match(spec.joint_values, joints):
        return f"joints {dict(joints or {})} vs spec {spec.name} {spec.joint_values}"
    h = float(getattr(top, "z")) - float(floor_z)
    if abs(h - float(spec.height_m)) > HEIGHT_TOL_M:
        return f"board at {h:.4f} m, spec {spec.name} at {float(spec.height_m):.4f} (> {HEIGHT_TOL_M})"
    hr = getattr(top, "headroom", None)
    mask = getattr(top, "mask", None)
    if hr is not None and mask is not None:
        try:
            import torch as th

            vals = th.as_tensor(hr)[th.as_tensor(mask)]
            if int(vals.numel()):
                med = float(vals.median()) + RASTER_HEADROOM_OFFSET_M
                if abs(med - float(spec.headroom_m)) > HEADROOM_TOL_M:
                    return f"headroom median {med:.3f} m, spec {spec.name} {float(spec.headroom_m):.3f} (> {HEADROOM_TOL_M})"
        except Exception:  # noqa: BLE001 - a raster stand-in without tensors: the height check only
            pass
    oe = getattr(top, "open_edges", None)
    if oe is not None and len(list(oe)) and (len(list(oe)) >= 2) != (spec.back_thickness_m is None):
        return f"open sides {list(oe)} vs spec {spec.name} {spec.open_sides}"
    return None


def summary(specs: Sequence[InteriorSpec]) -> List[Dict[str, Any]]:
    return [{"name": s.name, "digest": s.digest(), **s.geometry()} for s in specs]


# ------------------------------------------------------------------------------------------------ measuring a spec in its scene (Kit)

#: the counted AABB overhang of a support beyond its raster edge (``iors_large_inside.overhang``): the front plane is that far out
OVERHANG_MIN_M = 0.05


def compartment_frame(mid, inward) -> Tuple[Any, Any, Any]:
    """``(origin xy, inward xy, tangent xy)`` of a raster edge (tangent = the robot's left facing inward, ``stances.EdgeFrame``)."""
    import torch as th

    m = th.as_tensor(mid, dtype=th.float32)[:2]
    n = th.as_tensor(inward, dtype=th.float32)[:2]
    n = n / th.linalg.norm(n)
    return m, n, th.tensor([-float(n[1]), float(n[0])])


def oriented_box(points, yaw: float) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """``(centre, half)`` of the box of ``points`` (N, 3; the compartment frame) whose axes are turned ``yaw`` about z: the points
    rotated by -yaw, their AABB, its centre rotated back."""
    import numpy as np

    P = np.asarray(points, dtype=float)
    c, s = np.cos(yaw), np.sin(yaw)
    Rt = np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])  # world -> box axes
    Q = P @ Rt.T
    lo, hi = Q.min(axis=0), Q.max(axis=0)
    ctr = Rt.T @ (0.5 * (lo + hi))
    return tuple(float(v) for v in ctr), tuple(float(v) for v in 0.5 * (hi - lo))


def link_yaw_in(link_quat_xyzw, tangent) -> float:
    """The yaw of a link's x axis in a compartment frame whose x is ``inward`` (= -tangent rotated): measured from inward."""
    import math

    import numpy as np

    from ispatialgym.grasping import schema as S

    ax = S.quat_to_mat(np.asarray(link_quat_xyzw, dtype=float))[:, 0]
    t = np.asarray(tangent, dtype=float)
    n = np.array([t[1], -t[0]])  # inward from the tangent (tangent = (-n_y, n_x))
    return math.atan2(float(ax[:2] @ t), float(ax[:2] @ n))


def measure(world, support, z_surface: float, *, name: str, scene: str, edge: Optional[str] = None, joints: Optional[Dict[str, float]] = None,
            moved_links: Sequence[str] = (), edge_distances: Sequence[float] = (0.35, 0.45, 0.55), z_margin: float = 0.03, thickness_m: float = 0.02,
            static_aabb=None) -> InteriorSpec:
    """The :class:`InteriorSpec` of ``support``'s compartment at ``z_surface`` (m above the floor) in the live ``world`` (``SimWorld``; the
    support's joints already set to ``joints``, e.g. by ``reach_bands.apply_support_joints``). Measured, in the compartment frame of the
    open raster edge ``edge`` (default the first open edge; origin = its midpoint moved out to the front plane, the support's static AABB
    overhang beyond the raster edge when >= :data:`OVERHANG_MIN_M`; x inward, y the robot's left facing inward, z from the board top):
    the board (``SimWorld.support_interior``: z, its cells' depth span = ``board_setback_m`` .. ``depth_m``), the headroom (the raster's
    median + :data:`RASTER_HEADROOM_OFFSET_M`), the side walls (horizontal rays from the cells' lateral centre at mid depth and mid
    headroom: inner faces -> ``width_m``, the compartment centre -> ``centre_y_m`` relative to the raster edge's midpoint), the back panel
    (a ray inward at mid height that meets the support within the depth + 0.1 m; open when it does not or the opposite raster edge is
    open). Every OTHER link in ``moved_links`` (the open doors: the links ``partner_joint_init`` moved) is an extra NORMAL obstacle: its
    collision meshes' points (``mesh.points_in_parent_frame`` through ``link.scaled_transform``: the articulation at the joint values)
    in the compartment frame, boxed along the link's own x axis turned into that frame (a door hinged on a vertical axis stays an oriented
    box). Thicknesses not measured (boards, walls, back) are ``thickness_m``."""
    import math

    import torch as th

    from ispatialgym.sampling import object_joints as OJ

    floor_z = float(world.floor_z)
    itop = world.support_interior(support, floor_z + float(z_surface), z_margin=z_margin)
    edges = {n: (mid, normal, length) for n, mid, normal, length in itop.edges()}
    open_e = list(getattr(itop, "open_edges", []) or [])
    en = edge or (open_e[0] if open_e else None)
    if en is None or en not in edges:
        raise RuntimeError(f"{support.name}: no open raster edge at {float(itop.z):.3f} (open {open_e}, asked {edge!r})")
    mid0, outward, _length = edges[en]
    mid, inward, tangent = compartment_frame(mid0, -th.as_tensor(outward, dtype=th.float32)[:2])
    lo, hi = static_aabb if static_aabb is not None else support.aabb
    corners = th.tensor([[float(a), float(b)] for a in (float(lo[0]), float(hi[0])) for b in (float(lo[1]), float(hi[1]))])
    over = float((corners @ (-inward)).max()) - float(mid @ (-inward))
    front = over if over >= OVERHANG_MIN_M else 0.0
    origin = mid - inward * front
    cells = itop.cells(free_only=False)
    d = (cells - origin) @ inward
    l = (cells - origin) @ tangent
    res = float(itop.resolution)
    hr = float(th.as_tensor(itop.headroom)[th.as_tensor(itop.mask)].median()) + RASTER_HEADROOM_OFFSET_M
    zb = float(itop.z)
    z_mid = zb + 0.5 * hr
    paths = tuple(str(lk.prim_path) for lk in support.links.values())

    def ray(p0_xy, dir_xy, reach: float, z: float) -> Optional[float]:
        s0 = [float(p0_xy[0]), float(p0_xy[1]), float(z)]
        e0 = [float(p0_xy[0] + dir_xy[0] * reach), float(p0_xy[1] + dir_xy[1] * reach), float(z)]
        h = world.raycast(th.tensor([s0]), th.tensor([e0]))[0]
        if not h.get("hit") or float(h.get("distance", 0.0)) < 1e-3 or not str(h.get("rigidBody", "")).startswith(paths):
            return None
        return float(h["distance"])

    d_mid = 0.5 * (float(d.min()) + float(d.max()))
    lc0 = 0.5 * (float(l.min()) + float(l.max()))
    p_c = origin + inward * d_mid + tangent * lc0
    left, right = ray(p_c, tangent, 1.5, z_mid), ray(p_c, -tangent, 1.5, z_mid)
    if left is not None and right is not None:
        width, lc = left + right, lc0 + 0.5 * (left - right)
    else:
        width, lc = float(l.max() - l.min()) + res, lc0
    back = ray(origin + tangent * lc + inward * (front + 0.01), inward, float(d.max()) + 0.10 + res, z_mid)
    opposite = {"+u": "-u", "-u": "+u", "+v": "-v", "-v": "+v"}.get(en)
    closed = back is not None and opposite not in open_e
    depth = (front + 0.01 + back) if closed else float(d.max()) + 0.5 * res
    extra = []
    for link_name in moved_links:
        link = support.links[link_name]
        if not (link.collision_meshes or {}):
            continue  # a meta link that moved with the door (its lights): no collision geometry, no obstacle
        pts = []
        try:
            from omnigibson.utils import transform_utils as T

            tf = link.scaled_transform
            for mesh in (link.collision_meshes or {}).values():
                pp = mesh.points_in_parent_frame
                if pp is not None and len(pp):
                    pts.append(T.transform_points(pp, tf).float())
        except Exception:  # noqa: BLE001 - a mesh without readable points: the link's AABB corners
            pts = []
        if not pts:
            a, b = (th.as_tensor(v, dtype=th.float32) for v in link.aabb)
            pts = [th.tensor([[float(x), float(y), float(z)] for x in (a[0], b[0]) for y in (a[1], b[1]) for z in (a[2], b[2])])]
        W = th.cat(pts)
        Pc = th.stack([(W[:, :2] - origin) @ inward, (W[:, :2] - origin) @ tangent - lc, W[:, 2] - zb], dim=1).numpy()
        yaw = link_yaw_in(link.get_position_orientation()[1].numpy(), tangent.numpy())
        ctr, half = oriented_box(Pc, yaw)
        extra.append((f"door_{link_name}", ctr, half, yaw))
    joints_set = OJ.joint_dict(support) if joints else {}
    src = {"measured": f"interiors.measure: {scene}/{support.name} raster edge {en} at {zb:.4f} (requested {float(z_surface):.3f} +- {z_margin}), "
                       f"{int(itop.mask.sum())} board cells, open edges {open_e}, front plane {front:.4f} m out (static AABB overhang), "
                       f"wall rays {left} / {right}, back ray {back}, moved links {list(moved_links)}",
           "joints_settled": {k: round(float(joints_set.get(k, v)), 4) for k, v in (joints or {}).items()},
           "assumed": f"board / board above / walls / back thickness {thickness_m} m; long_m = the walls' outer span"}
    return InteriorSpec(name=name, height_m=round(zb - floor_z, 4), depth_m=round(depth, 4), width_m=round(width, 4), headroom_m=round(hr, 4),
                        board_thickness_m=thickness_m, top_thickness_m=thickness_m, wall_thickness_m=thickness_m, long_m=round(width + 2 * thickness_m, 4),
                        board_setback_m=round(max(0.0, float(d.min()) - 0.5 * res), 4), back_thickness_m=thickness_m if closed else None,
                        centre_y_m=round(lc - float((mid - origin) @ tangent), 4) if not math.isnan(lc) else 0.0,
                        edge_distances=tuple(float(v) for v in edge_distances),
                        joints=tuple(sorted((str(k), round(float(v), 4)) for k, v in (joints or {}).items())), extra_boxes=tuple(extra),
                        supports=((scene, support.name),), source=src)


def write_registry(path: str, specs: Sequence[InteriorSpec]) -> None:
    """Add / replace ``specs`` in the registry file ``path`` (created when missing)."""
    d = {"schema": SCHEMA, "interiors": {}}
    if os.path.exists(path):
        with open(path) as f:
            d = json.load(f)
    for sp in specs:
        js = sp.to_json()
        d.setdefault("interiors", {})[sp.name] = {k: v for k, v in js.items() if k not in ("name", "digest", "frame", "boxes", "schema")}
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(d, f, indent=1)


def main(argv=None):
    """``python -m ispatialgym.sampling.interiors measure --scene S --support NAME --surface-z Z [--surface-tol T] [--joints '{"j": v}']
    [--rooms R ...] [--edge +u] [--stand-offs 0.35 0.45 0.55 0.65] --name NAME --out REGISTRY.json`` (an OmniGibson scene load: a Kit job)."""
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("measure")
    m.add_argument("--scene", required=True)
    m.add_argument("--support", required=True)
    m.add_argument("--surface-z", type=float, required=True)
    m.add_argument("--surface-tol", type=float, default=0.03)
    m.add_argument("--joints", default=None, help="JSON {joint: value} (the task's partner_joint_init)")
    m.add_argument("--rooms", nargs="*", default=None)
    m.add_argument("--edge", default=None)
    m.add_argument("--stand-offs", nargs="+", type=float, default=[0.35, 0.45, 0.55])
    m.add_argument("--thickness", type=float, default=0.02)
    m.add_argument("--name", required=True)
    m.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    from ispatialgym.sampling import object_joints as OJ
    from ispatialgym.sampling.sim_utils import SimWorld

    world = SimWorld(a.scene, a.rooms)
    support = world.obj(a.support)
    joints, moved = {}, []
    if a.joints:
        joints, clamped = OJ.parse_joint_spec(json.loads(a.joints), list(support.joints), OJ.joint_limits(support))
        before = OJ.link_poses(support)
        OJ.set_joints(support, joints)
        world.step_physics(30, still=[world.robot])
        world.full_step()
        support.keep_still()
        moved = OJ.moved_links(before, OJ.link_poses(support))
    root = getattr(support, "root_link_name", "base_link")
    static = OJ.static_aabb(support, moved) if moved else None
    spec = measure(world, support, a.surface_z, name=a.name, scene=a.scene, edge=a.edge, joints=joints, moved_links=[n for n in moved if n != root],
                   edge_distances=a.stand_offs, z_margin=a.surface_tol, thickness_m=a.thickness, static_aabb=static)
    write_registry(a.out, [spec])
    print("INTERIOR_SPEC " + json.dumps(spec.to_json()), flush=True)
    import omnigibson as og

    og.shutdown()


if __name__ == "__main__":
    main()
