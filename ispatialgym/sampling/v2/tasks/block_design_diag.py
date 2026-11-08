"""Block design board-placement diagnostics (i7-blockdesign 2026-11-08, ``ISPATIALGYM_BD_DIAG``; logging only, no verdict changes).

``ISPATIALGYM_BD_DIAG=1``: once per process the manipulation candidates (shift, stand-off, lateral, trunk, region depth / lateral, flip
cells, score) and, per board position :meth:`BlockDesign._place_board` tests, the geometry (edge depth, corner containment), the cells
kept and every probe try's verdict - with the board / the board + support left out of the world when the probe fails, to separate
collision from reach. ``=2`` adds, once per candidate tried, a sweep of board positions (depth x lateral of the manipulation base) with
the on-top / cells / layout / probe verdict of each - the map of where the board could go from that stance.
"""

from __future__ import annotations

import logging
import os
from typing import List, Optional

import torch as th

log = logging.getLogger("ispatialgym.sampling.v2")

SWEEP_DEPTHS = tuple(round(0.42 + 0.03 * i, 2) for i in range(9))  # 0.42 .. 0.66
SWEEP_LATERALS = tuple(round(-0.18 + 0.04 * i, 2) for i in range(8))  # -0.18 .. +0.10


def level() -> int:
    try:
        return int(os.environ.get("ISPATIALGYM_BD_DIAG", "0") or 0)
    except ValueError:
        return 0


def _st(c) -> str:
    so = c.stand_off if c.stand_off is not None else float("nan")
    b = c.base_xy
    return (f"so{float(so):.2f}/lat{float(c.lateral):+.2f}/sh{float(c.extras.get('lateral_shift_m', 0.0)):+.2f}/t{float(c.translate):.2f}/{float(c.tilt):+.2f}"
            f" base({float(b[0]):.2f},{float(b[1]):.2f}) yaw{float(c.yaw):+.2f}")


def candidates(s, order: List[int], scores: Optional[dict] = None) -> None:
    """Once per process: every manipulation candidate in the order tried."""
    if level() < 1 or getattr(s, "_bd_diag_cands_done", False):
        return
    s._bd_diag_cands_done = True
    init = s.init_stance
    log.info("BD_DIAG init %s; %d candidates, tried %d", _st(init), len(order), int(s.gen("manipulation_candidates_tried", 6)))
    for rank, k in enumerate(order):
        c = s.manip_candidates[k]
        fi = c.extras.get("focus_info", {}) or {}
        cells = fi.get("cells")
        cells = [] if cells is None else list(cells)
        cb = " ".join(f"({float(c.world_to_base(xy)[0]):.2f}|{float(c.world_to_base(xy)[1]):+.2f})" for xy in cells)
        log.info("BD_DIAG cand rank %d k %d %s focus depth %.2f score %s cells %d flip %d: %s", rank, k, _st(c), float(fi.get("depth", float("nan"))),
                 f"{scores[k]:.3f}" if scores and k in scores else "-", len(cells), s._n_flip_cells(c, init), cb)


def corners(s, manip, depth: float, lateral: float, half: float, right, far) -> str:
    xy = manip.base_xy + manip.forward * float(depth) + manip.left * float(lateral)
    out = []
    for sy in (-1.0, 1.0):
        for sx in (-1.0, 1.0):
            p = xy + right * sx * half + far * sy * half
            out.append(("near" if sy < 0 else "far") + ("R" if sx > 0 else "L") + ("+" if s.top.contains(p, 0.01) else "-"))
    so = float(manip.stand_off) if manip.stand_off is not None else float("nan")
    return f"edge depth {float(depth) - so:+.3f} (half {half:.4f}) " + " ".join(out)


def probe(s, k, tag: str, manip, pose, right, toward, table_yaw, support, board_obj) -> None:
    """The probe's tries at one position, then (when it fails) with the board and with board + support out of the world."""
    if level() < 1:
        return
    det: List[str] = []
    ok = s._board_probe(manip, pose, right, toward, table_yaw, support, detail=det)
    msg = f"BD_DIAG probe k {k} {tag}: {'PASS' if ok else 'FAIL'} [{'; '.join(det)}]"
    if not ok:
        d2: List[str] = []
        ok2 = s._board_probe(manip, pose, right, toward, table_yaw, support, detail=d2, ignore=[board_obj])
        msg += f" | board ignored: {'PASS' if ok2 else 'FAIL'} [{'; '.join(d2[-4:])}]"
        if not ok2 and support is not None:
            d3: List[str] = []
            ok3 = s._board_probe(manip, pose, right, toward, table_yaw, support, detail=d3, ignore=[board_obj, support])
            msg += f" | board+support ignored: {'PASS' if ok3 else 'FAIL'} [{'; '.join(d3[-4:])}]"
    log.info("%s", msg[:3000])


def layout(s, k, tag: str, kept, manip, init, primary_dir: str) -> None:
    if level() < 1:
        return
    parts = []
    for c in kept:
        bc = manip.world_to_base(c)
        parts.append(f"({float(bc[0]):.2f}|{float(bc[1]):+.2f} init{float(init.world_to_base(c)[1]):+.2f} {'E' if s._flip_cell_ok(c, manip, init, primary_dir) else '-'})")
    log.info("BD_DIAG layout k %d %s: kept %d: %s", k, tag, len(kept), " ".join(parts))


def sweep(s, k, manip, right, toward, table_yaw, cells, flip_eligible, n_blocks, support, layout_ok, name, q, half, half_b) -> None:
    """``=2``: once per candidate, every (depth, lateral) of the sweep grid: on the top, cells kept / flip-eligible, layout, probe."""
    if level() < 2:
        return
    done = getattr(s, "_bd_diag_swept", None)
    if done is None:
        done = s._bd_diag_swept = set()
    if k in done:
        return
    done.add(k)
    far = -toward
    n_flip_needed = s._n_flips() if s._flips_expected() else 0
    rows = []
    for depth in SWEEP_DEPTHS:
        row = []
        for lateral in SWEEP_LATERALS:
            if not s._board_on_top(manip, depth, lateral, half, right, far):
                row.append("off")
                continue
            xy = manip.base_xy + manip.forward * float(depth) + manip.left * float(lateral)
            kept = [c for c in (cells or []) if abs(float(th.dot(c - xy, right))) > half_b or abs(float(th.dot(c - xy, toward))) > half_b]
            n_elig = sum(1 for c in kept if flip_eligible is not None and flip_eligible(c))
            cell_ok = len(kept) >= n_blocks and n_elig >= n_flip_needed
            lay = bool(layout_ok(kept)) if (cell_ok and layout_ok is not None) else None
            pose = (th.tensor([float(xy[0]), float(xy[1]), float(s.top.z) + 0.001]), q)
            s._teleport({name: pose})
            pr = s._board_probe(manip, pose, right, toward, table_yaw, support)
            row.append(f"{len(kept)}/{n_elig}{'L' if lay else ('l' if lay is False else '.')}{'P' if pr else 'p'}")
        rows.append(f"d{depth:.2f}: " + " ".join(f"{lat:+.2f}={v}" for lat, v in zip(SWEEP_LATERALS, row)))
    log.info("BD_DIAG sweep k %d %s (cells kept/flip-eligible, L layout ok / l no flip set / . not tried, P probe pass / p fail):\n  %s", k, _st(manip), "\n  ".join(rows))
