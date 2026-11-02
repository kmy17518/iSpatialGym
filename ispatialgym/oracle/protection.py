"""Protected objects during transits (owner 2026-09-26, overnight plan item 2).

Teacup 0925-222212 / 235559: on the way to the cup the hand passed low over the saucer and shoved it ~20 cm against the
cup before the grasp - the planner keeps the finger SPHERES >= 1 cm off, the real fingers stick out past them, and the
saucer is 1.5 cm tall. Raising the planner's activation distance reached the sampler's IK (one ``MotionPlannerCfg``) and
broke bar soap (wave 5), so nothing global changes; instead two checks switch on only near a protected object:

* **plan-time** (``Motions`` protect context): a planned transit - the free-space move to a pre-grasp, pre-place or
  pre-contact, the carry included - is checked with the moving arm's spheres (and any carried object's) against the
  protected objects' boxes inflated by :data:`INFLATION_M`; a plan that clears is executed unchanged, one that does not
  is re-planned via the point above its goal;
* **run-time** (``qa.Monitors`` transit span): during a transit any NEW robot contact with a protected object, or a
  protected object moving more than :data:`MOVE_TOL_M`, is a verdict - the executor rolls back and re-plans.

**Protected** = the objects the task's goal refers to (positive atoms of the BDDL goal: teacup -> the saucer,
disembedding -> the destination tray, block design -> the other blocks and the board) that the task itself placed in
the scene (not furniture: a scene object's name is ``<category>_<model>_<k>``), plus the action's own target until its
intended contact begins. Distractors (only in negated atoms), source containers and furniture keep today's handling.
A per-variant override lives in ``isg_customs.json``: ``"protection": {"exclude": [bddl names], "include": [...]}`` with a
``_protection_note`` (note: any key added to a variant entry changes its spec hash, hence its sampling seeds).
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch as th

log = logging.getLogger(__name__)

#: the gap between the planner's finger spheres and the real fingers (teacup 0925-235559: spheres 1 cm off, fingers in the saucer)
INFLATION_M = 0.02
#: a protected object that moves more than this during a transit is a run-time verdict
MOVE_TOL_M = 0.01
#: the endpoint-relative threshold stays this much under the endpoints' own clearance (their neighbourhood never fires)
ENDPOINT_SLACK_M = 0.002
#: QA verdict categories of the run-time check (the executor forces the via-above re-plan on the retry)
VERDICTS = ("protected_contact", "protected_moved")

_QUANTIFIERS = {"forall", "exists", "forpairs", "fornpairs", "forn"}


def _sexpr(text: str) -> Any:
    """A tiny s-expression reader for compact BDDL literals: ``(under ?a ?b)`` -> ``["under", "?a", "?b"]``."""
    tokens = text.replace("(", " ( ").replace(")", " ) ").split()
    stack: List[List[Any]] = [[]]
    for tok in tokens:
        if tok == "(":
            stack.append([])
        elif tok == ")":
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(tok)
    out = stack[0]
    return out[0] if len(out) == 1 else out


def positive_goal_instances(goal: Iterable[Any]) -> List[str]:
    """BDDL instance names (``saucer.n.02_1``) the goal refers to in POSITIVE atoms, in order, without duplicates:
    ``(not ...)`` literals are skipped (disembedding's distractors appear only there), ``and`` / ``or`` are followed,
    quantified literals are skipped. Literals may be parsed lists or compact strings."""
    out: List[str] = []

    def visit(lit: Any) -> None:
        if isinstance(lit, str):
            lit = _sexpr(lit) if lit.strip().startswith("(") else lit
        if not isinstance(lit, (list, tuple)) or not lit:
            return
        head = str(lit[0]).lower()
        if head == "not" or head in _QUANTIFIERS:
            return
        if head in ("and", "or"):
            for sub in lit[1:]:
                visit(sub)
            return
        for arg in lit[1:]:
            if isinstance(arg, (list, tuple)):
                visit(arg)
                continue
            name = str(arg).lstrip("?")
            if ".n." in name and name not in out:
                out.append(name)

    for lit in goal or ():
        visit(lit)
    return out


def is_task_object(obj: Any) -> bool:
    """Whether ``obj`` was placed in the scene by the task (``target_teacup_wopjex_0``, ``destination_tray_hvlfig_0``,
    ``targets_toy_dice_isgblk_0``) rather than being scene furniture (``breakfast_table_rhjoby_0``, ``door_crtrpa_0``,
    ``walls_urngpl_0``): the sampler names what it spawns ``<role>_<category>_<model>_<k>``, the scene keeps
    ``<category>_<model>_<k>``."""
    name = str(getattr(obj, "name", "") or "")
    cat = str(getattr(obj, "category", "") or "")
    if not name or not cat or cat in ("floors", "walls", "ceilings", "agent"):
        return False
    model = str(getattr(obj, "model", "") or "")
    if model and name.startswith(f"{cat}_{model}_"):
        return False
    if not model and name.startswith(f"{cat}_"):
        return False
    return True


def variant_override(variant: Optional[str], customs_path: Optional[str] = None) -> Dict[str, Any]:
    """``isg_customs.json`` -> ``[variant]["protection"]`` (``{"exclude": [...], "include": [...]}``) or ``{}``."""
    if not variant:
        return {}
    path = customs_path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "task_specs", "isg_customs.json")
    try:
        with open(path) as f:
            entry = json.load(f).get(variant) or {}
    except Exception as e:  # noqa: BLE001
        log.debug("protection: customs unreadable (%r)", e)
        return {}
    ov = entry.get("protection") if isinstance(entry, dict) else None
    return dict(ov) if isinstance(ov, dict) else {}


def goal_protected(goal: Iterable[Any], physical: Dict[str, Any], override: Optional[Dict[str, Any]] = None) -> List[Tuple[str, Any]]:
    """``[(bddl name, scene object)]`` of the protected objects: task-placed objects in the goal's positive atoms, minus
    ``override["exclude"]``, plus ``override["include"]`` (bddl names)."""
    ov = override or {}
    exclude = {str(x) for x in (ov.get("exclude") or ())}
    include = [str(x) for x in (ov.get("include") or ())]
    out: List[Tuple[str, Any]] = []
    for inst in positive_goal_instances(goal):
        obj = physical.get(inst)
        if obj is None or inst in exclude or not is_task_object(obj):
            continue
        out.append((inst, obj))
    for inst in include:
        obj = physical.get(inst)
        if obj is not None and not any(o is obj for _, o in out):
            out.append((inst, obj))
    return out


def object_box(obj: Any) -> Optional[Tuple[th.Tensor, th.Tensor, th.Tensor]]:
    """World box of ``obj`` as ``(centre (3,), rotation (3,3) whose columns are the box axes, half extents (3,))``: the
    base-aligned bounding box of its collision geometry, the AABB when that is unavailable; None without either."""
    from ispatialgym.robot import conventions as C

    try:
        c, q, ext, _ = obj.get_base_aligned_bbox(visual=False)
        return th.as_tensor(c, dtype=th.float32).reshape(3), C.quat_to_mat(th.as_tensor(q, dtype=th.float32).reshape(4)), 0.5 * th.as_tensor(ext, dtype=th.float32).reshape(3)
    except Exception:  # noqa: BLE001
        pass
    try:
        lo, hi = obj.aabb
        lo, hi = th.as_tensor(lo, dtype=th.float32), th.as_tensor(hi, dtype=th.float32)
        return 0.5 * (lo + hi), th.eye(3), 0.5 * (hi - lo)
    except Exception:  # noqa: BLE001
        return None


def sphere_box_clearance(spheres: th.Tensor, box: Tuple[th.Tensor, th.Tensor, th.Tensor]) -> Tuple[th.Tensor, th.Tensor]:
    """``spheres`` (N, S, 4) world x y z r -> per configuration the smallest signed distance between a sphere surface and
    the box (N,) and the index of that sphere (N,). Spheres with r <= 0 (disabled) are ignored (+inf)."""
    c, R, half = box
    sp = spheres.float().cpu()
    p = (sp[..., :3] - c.reshape(1, 1, 3)) @ R  # box frame
    q = p.abs() - half.reshape(1, 1, 3)
    outside = th.clamp(q, min=0.0).norm(dim=-1)
    inside = th.clamp(q.max(dim=-1).values, max=0.0)
    d = outside + inside - sp[..., 3]
    d = th.where(sp[..., 3] > 0, d, th.full_like(d, float("inf")))
    val, idx = d.min(dim=-1)
    return val, idx


def transit_violations(clear: th.Tensor, inflation: float = INFLATION_M, slack: float = ENDPOINT_SLACK_M) -> Optional[Dict[str, Any]]:
    """``clear`` (N,) clearance of the moving spheres to one protected box along a planned transit (N >= 2). The threshold
    is the inflation, lowered to the endpoints' own clearance (minus ``slack``) when the transit starts or ends closer
    than that - a pre-contact 3 cm off a painting, a brace already on the partner: the transit may approach as close as
    its endpoints are, no closer. Returns ``{"index", "clearance_m", "threshold_m"}`` of the worst interior configuration
    below the threshold, None when the transit clears."""
    n = int(clear.shape[0])
    if n < 3:
        return None
    c0, c1 = float(clear[0]), float(clear[-1])
    thr = min(float(inflation), c0 - float(slack), c1 - float(slack))
    interior = clear[1:-1]
    k = int(th.argmin(interior))
    worst = float(interior[k])
    if worst >= thr:
        return None
    return {"index": k + 1, "clearance_m": round(worst, 4), "threshold_m": round(thr, 4), "start_m": round(c0, 4), "end_m": round(c1, 4)}


def moving_sphere_mask(link_of_sphere: Sequence[str], arms: Sequence[str]) -> th.Tensor:
    """Spheres of the moving arm(s): the arm's links, gripper and fingers (``right_*``) and the object attached to its hand
    (``attached_object_right_eef_link``) - the static arm (a brace on the partner) and the body never set the threshold."""
    arms = tuple(arms)
    return th.tensor([any(l.startswith(f"{a}_") or l == f"attached_object_{a}_eef_link" for a in arms) for l in link_of_sphere], dtype=th.bool)


def check_trajectory(spheres: th.Tensor, link_of_sphere: Sequence[str], arms: Sequence[str], objects: Sequence[Tuple[str, Any]],
                     inflation: float = INFLATION_M) -> List[Dict[str, Any]]:
    """Every protected object the planned transit (``spheres`` (N, S, 4) world, one row per configuration) comes closer to
    than the endpoint-relative threshold: ``[{"object", "clearance_m", "threshold_m", "index", "link", ...}]``."""
    mask = moving_sphere_mask(link_of_sphere, arms)
    if not bool(mask.any()):
        return []
    sp = spheres[:, mask]
    links = [l for l, m in zip(link_of_sphere, mask.tolist()) if m]
    out: List[Dict[str, Any]] = []
    for name, obj in objects:
        box = object_box(obj)
        if box is None:
            continue
        val, idx = sphere_box_clearance(sp, box)
        v = transit_violations(val, inflation)
        if v is None:
            continue
        v["object"] = str(getattr(obj, "name", name))
        v["link"] = links[int(idx[v["index"]])] if links else ""
        v["n"] = int(val.shape[0])
        out.append(v)
    return out
