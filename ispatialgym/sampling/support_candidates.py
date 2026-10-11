"""Support candidates of the representative picks (owner 2026-10-11, the 55 representative tasks).

Every pick of ``roles/representatives.csv`` lists up to five support rows in order (``triplet`` = their index, the note
``init=<ontop|inside>; top_z=<m above the floor>``). ``ground_isg`` grounds one combination per row, labelled ``<pick>~T<triplet>``:
the rows of one support category share an alias and union their scene blocks (``bindings.support`` lists several names,
``bindings.support_surfaces`` one ``{name, init, top_z, row}`` entry per row, ``row`` = the label), an ``init=inside`` row gets its own
alias (support token ``<category>_inside``). Each row is one **candidate**, sampled on its own: ``sample_isg --scene S --support NAME
--support-row ROW`` (``GroundedProblem.override_support``). The owner's order per pick: (cand1, k=0) ... (cand5, k=0), then
(cand1, k=1) ... (cand5, k=1) with the eval seed ``--seed-split eval --seed-index k``, stopping at the first candidate whose instance
passes demo + replay (:func:`submission_order`).

:func:`pick_candidates` maps every combination of a pick to its alias and scene block: the customs ``support_surfaces`` entry whose
``row`` is the combination label (exact), else - walls, floor supports, rows without ``top_z``, picks without a support role - the
alias ``ground_isg`` gives the combination (its own combination code on the same icl, nothing written). It then checks the customs
block the way the sampler will use it (``override_support``, a non-empty whitelist per sampleable role). ``status`` is ``ok`` or why
the candidate cannot run: ``no_iddl``, ``not_grounded: <the grounder's skip reason>``, ``no_alias``, ``superseded_by:<alias>``,
``no_scene_block``, ``not_bound: <reason>``, ``empty_whitelist:<roles>``, ``duplicate_of:c<n>`` (the same block as an earlier
candidate of the pick: one job covers both).

Run (eval85: ``$E/pick_candidates.py``)::

    python -m ispatialgym.sampling.support_candidates [--launch55 | --subtask ID ... | --pick PICK ...] [--customs FILE] [--icl ROOT]
        [--order [--seeds 0,1] [--split eval]] [--only-ok] [--json FILE]

Output: tab-separated rows with a header, one per candidate (``--order``: one per (candidate, seed) in submission order, with a
suggested run directory ``<pick>/c<cand>_k<k>``); ``args`` = the ``sample_isg`` arguments of the candidate (and seed). A summary goes
to stderr.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shlex
import sys
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ispatialgym.sampling.v2.problem import DEFAULT_CUSTOMS, GroundedProblem, GroundingError, support_row_tag

#: the 55-task launch (owner 2026-10-11 04:58 / 13:25): every representative task but camera relocalization, object scaling and mental
#: rotation (sampled first) and the large-scale IORS tasks
LAUNCH_55_SUBTASKS = (
    "disembedding",
    "alignment-axial",
    "alignment-world_vertical",
    "configuration_matching-block_design",
    "configuration_matching-attachment",
    "configuration_matching-articulation_open_small_scale",
    "configuration_matching-articulation_close_small_scale",
    "configuration_matching-articulation_open_large_scale",
    "configuration_matching-articulation_close_large_scale",
    "inter_object_relation_satisfaction-ontop_small_scale",
    "inter_object_relation_satisfaction-under_small_scale",
    "inter_object_relation_satisfaction-inside",
    "inter_object_relation_satisfaction-nextto",
    "inter_object_relation_satisfaction-touching",
)
SLOT = "representative"
DEFAULT_SEEDS = (0, 1)

TABLE_COLUMNS = ("pick", "cand", "label", "alias", "scene", "support", "row", "init", "top_z", "triplet", "room", "kind", "status", "args")
ORDER_COLUMNS = ("pick", "step", "cand", "k", "label", "alias", "scene", "support", "row", "init", "top_z", "triplet", "status", "run", "args")


@dataclass
class Candidate:
    """One combination of a pick: the support row ``cand`` (1-based, sheet order) and the alias / scene block it runs in."""

    pick: str
    cand: int
    label: str  # the grounder's combination label: <pick>~T<triplet>, the pick itself when it has one combination
    subtask: str
    alias: Optional[str] = None
    scene: Optional[str] = None
    support: Optional[str] = None  # sample_isg --support: scene object / wall name, "floor"; None for a pick without a support role
    row: Optional[str] = None  # sample_isg --support-row: the support_surfaces row label, a wall's region id
    init: Optional[str] = None
    top_z: Optional[float] = None
    triplet: Optional[str] = None
    room: Optional[str] = None  # the support's room(s) per the selection (a wall region's room, a floor support's room)
    kind: str = "none"  # surface | wall | floor | object | none (no support role)
    fallbacks: List[str] = field(default_factory=list)  # the alias's fallback aliases (customs ``fallbacks``), rank order
    status: str = "ok"

    @property
    def ok(self) -> bool:
        return self.status == "ok"

    @property
    def tag(self) -> Optional[str]:
        """The support tag of the candidate's template ids (``<S>__support-<tag>__...``; :meth:`GroundedProblem.override_support`)."""
        if not self.support:
            return None
        return f"{self.support}-{support_row_tag(self.row)}" if self.row else self.support

    def sample_args(self, k: Optional[int] = None, split: str = "eval") -> List[str]:
        """``sample_isg`` arguments of the candidate (with the split seed ``k`` when given)."""
        out = ["--alias", self.alias or "?"]
        if self.scene:
            out += ["--scene", self.scene]
        if self.support:
            out += ["--support", self.support]
        if self.row:
            out += ["--support-row", self.row]
        if k is not None:
            out += ["--seed-split", split, "--seed-index", str(int(k))]
        return out

    def run_dir(self, k: int) -> str:
        """Suggested run directory of one (candidate, seed) job: ``<pick, # -> _>/c<cand>_k<k>``."""
        return f"{pick_slug(self.pick)}/c{self.cand}_k{int(k)}"


def pick_slug(pick: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", pick)


# ----------------------------------------------------------------------------------------------------------------
# the grounder's combinations of a pick
# ----------------------------------------------------------------------------------------------------------------


def pick_variants(icl, pick: str, slot: str = SLOT) -> List[Tuple[str, Dict[str, Dict[str, str]]]]:
    """``[(label, {role: sheet row})]`` of one pick and slot, in the order ``ground_isg`` builds them
    (``Grounder._combinations_from_picks``): a role the slot lists several times (the support candidates) gives one combination per row,
    labelled ``<pick>~T<triplet>`` (``<pick>~<n>`` without triplets), the pick itself when there is one combination."""
    roles = icl.picks.get(pick) or {}
    if slot != SLOT and not any(slot in slots for slots in roles.values()):
        return []
    choices: Dict[str, List[Dict[str, str]]] = {}
    for role, slots in roles.items():
        rows = slots.get(slot) or slots.get(SLOT) or []
        if rows:
            choices[role] = list(rows)
    multi = [r for r, rs in choices.items() if len(rs) > 1]
    variants: List[Dict[str, Dict[str, str]]] = [{}]
    for role, rs in choices.items():
        variants = [dict(v, **{role: r}) for v in variants for r in (rs if role in multi else rs[:1])]
    out = []
    for i, rows in enumerate(variants):
        if len(variants) == 1:
            label = pick
        else:
            tags = [f"T{rows[r]['triplet']}" for r in multi if rows[r].get("triplet")]
            label = f"{pick}~{'~'.join(tags) if len(tags) == len(multi) else i + 1}"
        out.append((label, rows))
    return out


def _surface_index(customs: Dict[str, Any]) -> Dict[Tuple[str, str], List[str]]:
    """``(scene, row label) -> [aliases]`` over the live primary aliases' ``bindings.support_surfaces`` entries."""
    out: Dict[Tuple[str, str], List[str]] = {}
    for alias, e in customs.items():
        if alias.startswith("_") or not isinstance(e, dict) or e.get("fallback_of") or e.get("superseded_by"):
            continue
        for sc, blk in (e.get("scenes") or {}).items():
            for s in (blk.get("bindings") or {}).get("support_surfaces") or []:
                if isinstance(s, dict) and s.get("row"):
                    out.setdefault((sc, s["row"]), []).append(alias)
    return out


def _note_rooms(row: Dict[str, str]) -> Optional[str]:
    m = re.search(r"\brooms=([A-Za-z0-9_,]+)", row.get("note") or "")
    return m.group(1) if m else None


def _check(c: Candidate, customs: Dict[str, Any]) -> None:
    """``c.status`` from the customs entry: the alias exists and is live, the scene block exists and binds the candidate as
    ``sample_isg --support`` will narrow it, every sampleable role keeps a model."""
    entry = customs.get(c.alias) if c.alias else None
    if not isinstance(entry, dict):
        c.status = "no_alias"
        return
    if entry.get("superseded_by"):
        c.status = f"superseded_by:{entry['superseded_by']}"
        return
    c.fallbacks = [a for a in entry.get("fallbacks") or [] if isinstance(customs.get(a), dict) and not customs[a].get("superseded_by")]
    if c.scene is None:
        return  # a pick without a scene object (held objects, camera relocalization) samples every block of its alias
    if c.scene not in (entry.get("scenes") or {}):
        c.status = "no_scene_block"
        return
    gp = GroundedProblem(c.alias, copy.deepcopy(entry), subtask=c.subtask)
    if c.support:
        try:
            rec = gp.override_support(c.scene, c.support, row=c.row)
        except GroundingError as e:
            c.status = f"not_bound: {e}"
            return
        if rec.get("init") is not None:
            c.init, c.top_z = rec["init"], rec["top_z"]
    empty = [r for r in gp.sampleable_roles() if not any(gp.whitelist(c.scene, r).values())]
    if empty:
        c.status = f"empty_whitelist:{','.join(empty)}"


def pick_candidates(customs: Dict[str, Any], icl=None, iddls: Optional[Dict[str, Any]] = None, subtasks: Optional[Sequence[str]] = None,
                    picks: Optional[Sequence[str]] = None) -> Dict[str, List[Candidate]]:
    """``{pick: [Candidate]}`` (sheet order) of the picks of ``subtasks`` / ``picks`` (default: every pick of the sheet), from the
    customs data (``{alias: entry}``), the icl (``ground_isg.ICL``; its ``roles/representatives.csv`` is the sheet) and the iddls."""
    from ispatialgym.sampling import ground_isg as GI
    from ispatialgym.sampling import iddl as I
    from ispatialgym.sampling.customs import Customs

    icl = icl if icl is not None else GI.ICL()
    iddls = iddls if iddls is not None else I.load_all(GI.DEFAULT_IDDL)
    wanted = [p for p in icl.picks if (not subtasks or p.split("#")[0] in subtasks) and (not picks or p in picks)]
    grounder = GI.Grounder(iddls, icl, Customs(), options=GI.Options(dry_run=True, representative_test=False, fallbacks=False))
    combos: Dict[str, Any] = {}
    skipped: Dict[str, str] = {}
    for st in dict.fromkeys(p.split("#")[0] for p in wanted):
        if st not in iddls:
            continue
        for comb in grounder.combinations(st):
            combos[comb.pick] = comb
        for label, reason in grounder.report.skipped.get(st, []):
            skipped[label.rsplit("/", 1)[0]] = reason
    surfaces = _surface_index(customs)
    out: Dict[str, List[Candidate]] = {}
    for pick in wanted:
        st = pick.split("#")[0]
        cands: List[Candidate] = []
        for i, (label, rows) in enumerate(pick_variants(icl, pick), start=1):
            c = Candidate(pick, i, label, st)
            srow = rows.get("support")
            anchor = srow or next((r for r in rows.values() if r.get("kind") == "scene" and r.get("scene")), None)
            if anchor is not None:
                c.scene = anchor.get("scene") or None
            if srow is not None:
                c.triplet = srow.get("triplet") or None
                c.room = _note_rooms(srow)
            cands.append(c)
            if st not in iddls:
                c.status = "no_iddl"
                continue
            comb = combos.get(label)
            if comb is None:
                c.status = f"not_grounded: {skipped.get(label, 'no combination')}"
                continue
            iddl = iddls[st]
            c.alias = iddl.fill_alias(comb.alias_values(icl, iddl))
            rp = comb.roles.get("support")
            if rp is not None and srow is not None:
                c.room = c.room or (rp.pick_row or {}).get("room") or None
                name = srow.get("name") or srow.get("model") or ""
                if rp.is_floor:
                    c.kind, c.support, c.room = "floor", "floor", comb.floor_room.get(c.scene) or c.room
                elif "/" in name:
                    c.kind, c.support, c.row = "wall", srow.get("model") or name.split("/")[1], name
                    c.room = name.split("/")[0]
                elif rp.surface is not None:
                    c.kind, c.support, c.row = "surface", name, label
                    c.init, c.top_z = rp.surface.get("init"), rp.surface.get("top_z")
                    hits = surfaces.get((c.scene, label)) or []
                    if len(hits) == 1 or (hits and c.alias not in hits):
                        c.alias = hits[0]
                else:
                    c.kind, c.support = "object", name
            _check(c, customs)
        seen: Dict[Tuple[Any, ...], int] = {}
        for c in cands:
            key = (c.alias, c.scene, c.support, c.row)
            if c.ok and key in seen:
                c.status = f"duplicate_of:c{seen[key]}"
            elif c.ok:
                seen[key] = c.cand
        out[pick] = cands
    return out


def submission_order(cands: Dict[str, List[Candidate]], seeds: Sequence[int] = DEFAULT_SEEDS) -> List[Tuple[Candidate, int, int]]:
    """``[(candidate, k, step)]``: per pick, every candidate with ``seeds[0]`` in candidate order, then every candidate with
    ``seeds[1]``, ...; ``step`` counts 1.. within the pick."""
    out = []
    for pick_cands in cands.values():
        step = 0
        for k in seeds:
            for c in pick_cands:
                step += 1
                out.append((c, int(k), step))
    return out


# ----------------------------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------------------------


def _fmt(v: Any) -> str:
    if v is None or v == "" or v == []:
        return "-"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def table_rows(cands: Dict[str, List[Candidate]], only_ok: bool = False) -> List[Dict[str, str]]:
    out = []
    for pick_cands in cands.values():
        for c in pick_cands:
            if only_ok and not c.ok:
                continue
            out.append({"pick": c.pick, "cand": str(c.cand), "label": c.label, "alias": _fmt(c.alias), "scene": _fmt(c.scene),
                        "support": _fmt(c.support), "row": _fmt(c.row), "init": _fmt(c.init), "top_z": _fmt(c.top_z), "triplet": _fmt(c.triplet),
                        "room": _fmt(c.room), "kind": c.kind, "status": c.status, "args": shlex.join(c.sample_args()) if c.alias else "-"})
    return out


def order_rows(cands: Dict[str, List[Candidate]], seeds: Sequence[int] = DEFAULT_SEEDS, split: str = "eval",
               only_ok: bool = False) -> List[Dict[str, str]]:
    out = []
    for c, k, step in submission_order(cands, seeds):
        if only_ok and not c.ok:
            continue
        out.append({"pick": c.pick, "step": str(step), "cand": str(c.cand), "k": str(k), "label": c.label, "alias": _fmt(c.alias),
                    "scene": _fmt(c.scene), "support": _fmt(c.support), "row": _fmt(c.row), "init": _fmt(c.init), "top_z": _fmt(c.top_z),
                    "triplet": _fmt(c.triplet), "status": c.status, "run": c.run_dir(k),
                    "args": shlex.join(c.sample_args(k, split)) if c.alias else "-"})
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="pick_candidates", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--launch55", action="store_true", help=f"the picks of the 55-task launch ({len(LAUNCH_55_SUBTASKS)} subtasks)")
    sel.add_argument("--subtask", action="append", default=None, help="the picks of this subtask (repeatable)")
    sel.add_argument("--pick", action="append", default=None, help="this pick, e.g. 'disembedding#05' (repeatable)")
    ap.add_argument("--customs", default=os.environ.get("ISG_CUSTOMS") or DEFAULT_CUSTOMS, help="grounded customs (default: $ISG_CUSTOMS, else the assets' isg_customs.json)")
    ap.add_argument("--icl", default=None, help="icl root; the sheet is <icl>/roles/representatives.csv (default: the assets' icl)")
    ap.add_argument("--iddl", default=None, help="iddl root (default: the assets' iddl)")
    ap.add_argument("--order", action="store_true", help="the submission order: (cand1, k0) ... (candN, k0), (cand1, k1) ... per pick")
    ap.add_argument("--seeds", default=",".join(str(k) for k in DEFAULT_SEEDS), help="--order: the seed indices, in pass order (default 0,1)")
    ap.add_argument("--split", default="eval", help="--order: the seed split (default eval)")
    ap.add_argument("--only-ok", action="store_true", help="only the candidates that can run (status ok)")
    ap.add_argument("--json", default=None, help="also write the candidates (and the order) as JSON to this file")
    args = ap.parse_args(argv)

    from ispatialgym.sampling import ground_isg as GI
    from ispatialgym.sampling import iddl as I

    with open(args.customs) as f:
        customs = json.load(f)
    icl = GI.ICL(args.icl) if args.icl else GI.ICL()
    iddls = I.load_all(args.iddl or GI.DEFAULT_IDDL)
    subtasks = list(LAUNCH_55_SUBTASKS) if args.launch55 else args.subtask
    unknown = [p for p in args.pick or [] if p not in icl.picks]
    if unknown:
        print(f"unknown pick(s) {unknown}", file=sys.stderr)
        return 2
    cands = pick_candidates(customs, icl, iddls, subtasks=subtasks, picks=args.pick)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    rows, cols = (order_rows(cands, seeds, args.split, args.only_ok), ORDER_COLUMNS) if args.order else (table_rows(cands, args.only_ok), TABLE_COLUMNS)
    print("\t".join(cols))
    for r in rows:
        print("\t".join(r[c] for c in cols))
    if args.json:
        data = {"customs": os.path.abspath(args.customs), "icl": icl.root,
                "picks": {p: [{**asdict(c), "tag": c.tag, "args": c.sample_args() if c.alias else None} for c in cs] for p, cs in cands.items()},
                "order": [{"pick": c.pick, "step": step, "cand": c.cand, "k": k, "status": c.status, "run": c.run_dir(k),
                           "args": c.sample_args(k, args.split) if c.alias else None} for c, k, step in submission_order(cands, seeds)]}
        with open(args.json, "w") as f:
            json.dump(data, f, indent=1)
    all_c = [c for cs in cands.values() for c in cs]
    bad = [c for c in all_c if not c.ok]
    no_run = [p for p, cs in cands.items() if not any(c.ok for c in cs)]
    print(f"# {len(cands)} picks, {len(all_c)} candidates: {len(all_c) - len(bad)} ok, {len(bad)} not runnable; picks without a runnable "
          f"candidate: {len(no_run)}{' (' + ', '.join(no_run) + ')' if no_run else ''}  [customs {args.customs}]", file=sys.stderr)
    for c in bad:
        print(f"#   {c.label}: {c.status}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
