"""Contact maps that learn from real runs (owner A6, 2026-11-08, i7-contactfu) - the ONE overlay mechanism of the expanding tilt map
(:mod:`ispatialgym.sampling.v2.map_tilt`, i7-plateqtf): the same event schema (``isg-map-overlay/1``), per-process / per-job shard files,
the same offline merge into ``<root>/<map key>/v<N>.json`` and the same rule that the frozen map is never edited.

A contact map (:mod:`contact_maps`) is built from the library entries a physics validation found VALID (stratified: 2-3 per
(yaw x end x spin) group, :class:`validate_contacts.GroupQuota`). It then learns from the demos:

* the sampler notes, per instance, which library entry its push / brace executes (:func:`note_used`: the map's LINEAGE key - the key of
  the frozen map the overlay belongs to -, the kind, the entry's yaw slot and class label, whether the map held it);
* after the demo, :func:`harvest` turns each SUCCESSFUL demo's noted entries into ``ok`` events of the lineage's overlay, one immutable shard
  per success under the job directory (``<job>/map_overlay/<lineage>/harvest__<instance>.jsonl``: one writer, no race; a harvest that
  finds the file skips it);
* ``map_tilt merge`` (offline) folds the shards into a new overlay version (``<overlay root>/<lineage>/v<N>.json``);
* ``contact_maps compute --overlay-root <root>`` rebuilds the map: the VALID entries plus every entry the merged overlay CONFIRMED (a real
  oracle success) - an entry joins only after a real oracle success (here) or a later physics validation (the library's verdict). The
  new map's identity carries ``overlay = {lineage, version}`` (:func:`contact_maps.map_key`), so a job pins the version it read.

Event entry ids: ``contact|<yaw slot deg>|<class>`` (:func:`entry_id`) - the library entry of that (yaw slot, class) in the lineage's
library (the rebuild resolves it and checks the library digest the event recorded).
"""

from __future__ import annotations

import glob
import json
import math
import os
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ispatialgym.sampling.v2 import map_tilt as MT

PREFIX = "contact"
SHARD_DIR = "map_overlay"  # the fast flow's job dir subdirectory (job_fast.sh: ISPATIALGYM_MAP_OVERLAY_SHARDS=$JOBDIR/map_overlay)


def entry_id(yaw_deg: float, cls: str) -> str:
    return f"{PREFIX}|{float(yaw_deg):g}|{cls}"


def parse_entry_id(eid: str) -> Optional[Tuple[float, str]]:
    parts = str(eid).split("|", 2)
    if len(parts) != 3 or parts[0] != PREFIX:
        return None
    return float(parts[1]), parts[2]


def lineage_key(meta: Dict[str, Any]) -> str:
    """The overlay key of a map: the frozen map it grew from (``meta.overlay.lineage``), else its own key."""
    return str((meta.get("overlay") or {}).get("lineage") or meta.get("key"))


def note_used(summary: Dict[str, Any], cm, kind: str, cls: Optional[str], lookup: Optional[Dict[str, Any]], role: str, **facts) -> Optional[Dict[str, Any]]:
    """Record in a draft's ``summary["contact_entries"]`` the library entry an instance executes: the map ``cm`` (``ContactMap``; None:
    nothing to learn into), the executed class label ``cls`` and the map lookup at the executing stance (``TT.contact_map_check``: the yaw
    slot and its error). The harvest reads it after the demo."""
    if cm is None or not cls or not lookup or lookup.get("yaw_slot_deg") is None:
        return None
    slot = float(lookup["yaw_slot_deg"])
    rows = cm.class_rows(int(min(range(len(cm.yaws)), key=lambda i: abs(math.degrees(float(cm.yaws[i])) - slot))))
    in_map = any(r is not None and r.get("class") == cls for r in rows)
    rec = {"kind": kind, "role": role, "lineage": lineage_key(cm.meta), "map_key": cm.meta.get("key"), "map": os.path.basename(cm.path),
           "overlay_version": (cm.meta.get("overlay") or {}).get("version", 0), "library_digest": (cm.meta.get("library") or {}).get("digest"),
           "yaw_slot_deg": lookup.get("yaw_slot_deg"), "yaw_err_deg": lookup.get("yaw_err_deg"), "class": cls, "in_map": bool(in_map),
           "map_feasible_here": cls in (lookup.get("feasible") or []), **facts}
    summary.setdefault("contact_entries", []).append(rec)
    return rec


def _summary_of(tro_path: str) -> Optional[Dict[str, Any]]:
    p = tro_path[: -len("-tro_state.json")] + ".summary.json" if tro_path.endswith("-tro_state.json") else None
    if not p or not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def harvest(jobdir: str, *, max_yaw_err_deg: float = 22.5) -> Dict[str, int]:
    """One job directory's SUCCESSFUL demos -> ``ok`` events (module doc): per success one shard file per lineage. Entries whose yaw is
    more than ``max_yaw_err_deg`` from the slot are not learned (the executed pose is not that entry's). Returns counts."""
    out = {"successes": 0, "events": 0, "skipped_existing": 0, "no_summary": 0, "far_yaw": 0}
    res = os.path.join(jobdir, "results.jsonl")
    if not os.path.exists(res):
        return out
    with open(res) as f:
        rows = [json.loads(ln) for ln in f if ln.strip()]
    for r in rows:
        if not (r.get("success") or r.get("status") == "success"):
            continue
        out["successes"] += 1
        summ = _summary_of(str(r.get("tro") or ""))
        if summ is None:
            out["no_summary"] += 1
            continue
        by_lineage: Dict[str, List[Dict[str, Any]]] = {}
        for c in summ.get("contact_entries") or []:
            if c.get("yaw_err_deg") is not None and float(c["yaw_err_deg"]) > max_yaw_err_deg:
                out["far_yaw"] += 1
                continue
            by_lineage.setdefault(str(c["lineage"]), []).append(c)
        for lin, cs in by_lineage.items():
            path = os.path.join(jobdir, SHARD_DIR, lin, f"harvest__{r['instance']}.jsonl")
            if os.path.exists(path):
                out["skipped_existing"] += 1
                continue
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = path + f".tmp{os.getpid()}"
            with open(tmp, "w") as f:
                for c in cs:
                    line = {"schema": MT.SCHEMA, "map_key": lin, "base_version": int(c.get("overlay_version") or 0),
                            "entry": entry_id(float(c["yaw_slot_deg"]), str(c["class"])), "outcome": "ok", "source": "oracle_success",
                            "instance": r["instance"], "job_id": r.get("job_id"), "kind": c.get("kind"), "role": c.get("role"),
                            "in_map": c.get("in_map"), "library_digest": c.get("library_digest"), "yaw_err_deg": c.get("yaw_err_deg")}
                    f.write(json.dumps(line, sort_keys=True) + "\n")
                    out["events"] += 1
            os.rename(tmp, path)
    return out


def confirmed(root: Optional[str], lineage: str, version: Optional[int] = None) -> Tuple[Dict[Tuple[float, str], Dict[str, int]], int]:
    """``({(yaw slot deg, class): record}, version)`` of the contact entries a merged overlay version CONFIRMED (an ``ok`` event)."""
    entries, v = MT.load_merged(root, lineage, version)
    out = {}
    for eid, rec in entries.items():
        k = parse_entry_id(eid)
        if k is not None and MT.is_confirmed(rec):
            out[(round(k[0], 4), k[1])] = dict(rec)
    return out, v


def join_entries(lib_entries: Sequence[Dict[str, Any]], selected: Sequence[Dict[str, Any]], conf: Dict[Tuple[float, str], Dict[str, int]],
                 families: Iterable[str]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """The map's entries with the overlay: ``selected`` (the validated ones) plus every library entry of ``families`` whose (yaw slot, class)
    the overlay confirmed. Returns ``(entries, joined facts)``."""
    fams = set(families)
    have = {(round(float(e.get("yaw_deg", 0.0)), 4), str(e["class"])) for e in selected}
    out, joined = list(selected), []
    for e in lib_entries:
        k = (round(float(e.get("yaw_deg", 0.0)), 4), str(e["class"]))
        if e.get("family") in fams and k in conf and k not in have:
            out.append(e)
            have.add(k)
            joined.append({"yaw_deg": k[0], "class": k[1], "verdict_before": (e.get("validation") or {}).get("verdict"), "ok": int(conf[k].get("ok", 0))})
    return out, joined


def main(argv=None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description="contact-map overlay: harvest successful demos into per-job shards (merge: python -m ispatialgym.sampling.v2.map_tilt merge)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("harvest", help="successful demos of job dirs -> ok events (<job>/map_overlay/<lineage>/harvest__<instance>.jsonl)")
    h.add_argument("jobdirs", nargs="+", help="job dirs or globs ($OFH/out/<tag>/*/g*)")
    a = ap.parse_args(argv)
    tot: Dict[str, int] = {}
    for d in sorted({p for g in a.jobdirs for p in (glob.glob(g) or [g]) if os.path.isdir(p)}):
        r = harvest(d)
        for k, v in r.items():
            tot[k] = tot.get(k, 0) + v
    print(json.dumps(tot))


if __name__ == "__main__":
    main()
