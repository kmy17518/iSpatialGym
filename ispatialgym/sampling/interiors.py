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
                   "board_setback_m", "back_thickness_m")
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
    supports: Tuple[Tuple[str, str], ...] = ()
    source: Dict[str, Any] = field(default_factory=dict, compare=False, hash=False)

    def geometry(self) -> Dict[str, Any]:
        g = {k: (None if getattr(self, k) is None else round(float(getattr(self, k)), 4)) for k in GEOMETRY_FIELDS}
        return {"schema": SCHEMA, **g, "open_sides": self.open_sides}

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
        kw = {k: d[k] for k in GEOMETRY_FIELDS if k in d}
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


def load_registry(path: Optional[str] = None) -> Dict[str, InteriorSpec]:
    p = path or os.environ.get("ISPATIALGYM_INTERIORS") or REGISTRY
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        d = json.load(f)
    if d.get("schema") != SCHEMA:
        raise ValueError(f"{p}: schema {d.get('schema')!r}, expected {SCHEMA}")
    return {name: InteriorSpec.from_json(s, name) for name, s in (d.get("interiors") or {}).items()}


def spec_named(name_or_path: str) -> InteriorSpec:
    """A registry spec by name, or a spec json file (one spec: its fields + ``name``)."""
    if os.path.isfile(name_or_path):
        with open(name_or_path) as f:
            return InteriorSpec.from_json(json.load(f))
    reg = load_registry()
    if name_or_path not in reg:
        raise KeyError(f"no interior spec {name_or_path!r} in {REGISTRY} (has {sorted(reg)})")
    return reg[name_or_path]


def spec_for_support(scene: Optional[str], support: Optional[str], height: float, tol: float = HEIGHT_TOL_M,
                     registry: Optional[Dict[str, InteriorSpec]] = None) -> Optional[InteriorSpec]:
    """The spec describing ``support`` of ``scene`` whose board is within ``tol`` of ``height`` (m above the floor), else None."""
    reg = load_registry() if registry is None else registry
    best = None
    for spec in reg.values():
        if not any(sc == scene and nm == support for sc, nm in spec.supports):
            continue
        d = abs(float(spec.height_m) - float(height))
        if d <= tol and (best is None or d < best[0]):
            best = (d, spec)
    return None if best is None else best[1]


def live_mismatch(spec: InteriorSpec, top: Any, floor_z: float = 0.0) -> Optional[str]:
    """Why the live interior raster ``top`` (``SimWorld.support_interior``) is not ``spec``'s compartment, or None: the board height
    (:data:`HEIGHT_TOL_M`), the median headroom of its board cells (:data:`HEADROOM_TOL_M`, when the raster has one) and the open sides
    (a raster with one open edge has a back panel, two: open front and back)."""
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
