#!/usr/bin/env python3
"""i7-stancememo 5x.3 (owner 2026-11-08): the adaptive per-attempt time cap, counted on the logs (CPU) before any run.

Rule (owner): per alias, an attempt is capped at max(300 s, 5 x the median written-attempt time of that alias), from the existing runs.
The switch (``ISPATIALGYM_ATTEMPT_CAP``, ``ispatialgym/sampling/v2/attempt_cap.py``) cuts an attempt while it is still SAMPLING (the
search: reset + sample); the post-sample stages are short and bounded and run to their end. So an attempt counts as cut here when its
reset + sample seconds exceed the cap; attempts whose TOTAL exceeds the cap but whose search ended before it are listed apart.

Input: the i7-diag audit's per-attempt rows ($OFH/diag_20261108/early_reject/attempts_r4.jsonl, one row per attempt of every job with a
sample.log newer than 2026-11-06 12:00; i7-diag d1f06313 scripts/oracle_fix/audit_early_reject.py scan).

    python3 time_cap_count.py table  <attempts.jsonl> <out.tsv> [primary|all]     # the per-alias medians the switch reads
    python3 time_cap_count.py count  <attempts.jsonl> [primary|all] [alias ...]   # what the cap would have cut, per alias and in total

The median pool: ``primary`` (default) = the audit's primary set (every int1_* job + per alias the latest tag whose jobs all COMPLETED),
the current state per alias - its median is the audit's (upper median, ``xs[len // 2]``); an alias with no written attempt there falls
back to all its runs. ``all`` = every written attempt in the file (incl. diagnostic runs and older trunks). The count runs over the
attempts of the same pool. ``table`` writes ``alias_key  n_written  median_written_s  cap_s  pool`` (cap = max(300, 5 x median));
aliases without any written attempt get no row (the switch leaves them uncapped). Counted per alias over every attempt in the file: cut attempts (rejected / written), the
seconds past the cap, how many cut REJECTED attempts belong to an instance that a later attempt of the same job wrote, and the
written attempts the cap would have cut (lost demos)."""
import collections
import json
import sys

FLOOR_S, MULT = 300.0, 5.0


def load(path, pool="primary"):
    atts, jobs = [], []
    for line in open(path):
        r = json.loads(line)
        if r.get("kind") == "attempt" and r.get("t") is not None and r.get("inst", -1) >= 0:
            atts.append(r)
        elif r.get("kind") == "job":
            jobs.append(r)
    if pool == "all":
        return atts, atts
    sel = primary(jobs)
    return atts, [a for a in atts if (a["tag"], a["key"]) in sel]


def primary(jobs):
    """The i7-diag audit's primary set (audit_early_reject.primary): every int1_* job, plus per alias key the latest tag whose jobs all
    COMPLETED with stats."""
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for j in jobs:
        by[j["key"]][j["tag"]].append(j)
    sel = set()
    for k, tags in by.items():
        best = None
        for t, js in tags.items():
            if t.startswith("int1_"):
                sel.add((t, k))
            if all(j["state"] == "COMPLETED" and j["attempts"] is not None for j in js):
                e = max(j["end"] for j in js)
                if best is None or e > best[0]:
                    best = (e, t)
        if best:
            sel.add((best[1], k))
    return sel


def med(xs):
    xs = sorted(xs)
    return xs[len(xs) // 2] if xs else None


def medians(atts, pool_atts=None):
    """{alias key: (n written, median written s, pool)}: from ``pool_atts`` (the primary set), else from ``atts`` (all runs)."""
    def by_key(rows):
        wt = collections.defaultdict(list)
        for a in rows:
            if a["outcome"] == "written":
                wt[a["key"]].append(float(a["t"]))
        return wt
    out = {k: (len(v), med(v), "all") for k, v in by_key(atts).items()}
    if pool_atts is not None and pool_atts is not atts:
        out.update({k: (len(v), med(v), "primary") for k, v in by_key(pool_atts).items()})
    return out


def cap_of(m):
    return max(FLOOR_S, MULT * m)


def search_s(a):
    st = a.get("stages") or {}
    return float(st.get("reset", 0.0)) + float(st.get("sample", 0.0))


def table(path, out, pool="primary"):
    atts, pa = load(path, pool)
    meds = medians(atts, pa)
    with open(out, "w") as f:
        f.write(f"# i7-stancememo 5x.3: per-alias median written-attempt seconds (pool {pool} of {path}); cap = max(300, 5 x median)\n")
        f.write("alias_key\tn_written\tmedian_written_s\tcap_s\tpool\n")
        for k in sorted(meds):
            n, m, p = meds[k]
            f.write(f"{k}\t{n}\t{m:.1f}\t{cap_of(m):.1f}\t{p}\n")
    print(f"{len(meds)} aliases -> {out}")


def count(path, pool="primary", only=()):
    atts_all, atts = load(path, pool)
    meds = medians(atts_all, atts)
    written_inst = collections.defaultdict(set)  # (tag, job, key) -> {(inst, att)} written
    for a in atts:
        if a["outcome"] == "written":
            written_inst[(a["tag"], a["job"], a["key"])].add((a["inst"], a["att"]))
    per = collections.defaultdict(lambda: collections.Counter())
    tot = collections.Counter()
    examples = collections.defaultdict(list)
    for a in atts:
        k = a["key"]
        if only and not any(o in k for o in only):
            continue
        if k not in meds:
            per[k]["no_median"] += 1
            continue
        cap = cap_of(meds[k][1])
        c = per[k]
        c["attempts"] += 1
        c["attempt_s"] += float(a["t"])
        s = search_s(a)
        if s > cap:
            cut_s = float(a["t"]) - cap
            if a["outcome"] == "written":
                c["cut_written"] += 1
                examples[k].append(f"WRITTEN {a['tag']} {a['inst']}/{a['att']} {a['t']:.0f} s (search {s:.0f} s)")
            else:
                c["cut_rejected"] += 1
                c["saved_s"] += cut_s
                later = any(i == a["inst"] and j > a["att"] for i, j in written_inst[(a["tag"], a["job"], k)])
                c["cut_rejected_inst_later_written"] += int(later)
                c[f"cut_reason.{a['reason']}"] += 1
                if len(examples[k]) < 6:
                    examples[k].append(f"{a['tag']} {a['inst']}/{a['att']} {a['reason']} {a['t']:.0f} s (search {s:.0f} s)")
        elif float(a["t"]) > cap:
            c["total_over_cap_search_under"] += 1
            c["total_over_cap_written"] += int(a["outcome"] == "written")
    print("alias\tmedian_written_s (n)\tcap_s\tattempts\tattempt_s\tcut_rejected\tcut_written\tsaved_s\tcut_rej_inst_later_written\t"
          "total>cap_search<cap (written)\tcut reasons")
    for k in sorted(per, key=lambda k: -per[k]["saved_s"]):
        c = per[k]
        if "attempts" not in c:
            continue
        for x in ("attempts", "attempt_s", "cut_rejected", "cut_written", "saved_s", "cut_rejected_inst_later_written", "total_over_cap_search_under",
                  "total_over_cap_written"):
            tot[x] += c[x]
        reasons = ", ".join(f"{r.split('.', 1)[1]} {n}" for r, n in sorted(c.items()) if r.startswith("cut_reason."))
        if c["cut_rejected"] or c["cut_written"] or c["total_over_cap_search_under"]:
            n, m, _ = meds[k]
            print(f"{k.replace('inter_object_relation_satisfaction-', 'iors-')}\t{m:.0f} ({n})\t{cap_of(m):.0f}\t{c['attempts']}\t{c['attempt_s']:.0f}\t"
                  f"{c['cut_rejected']}\t{c['cut_written']}\t{c['saved_s']:.0f}\t{c['cut_rejected_inst_later_written']}\t"
                  f"{c['total_over_cap_search_under']} ({c['total_over_cap_written']})\t{reasons}")
    print(f"TOTAL\t-\t-\t{tot['attempts']}\t{tot['attempt_s']:.0f}\t{tot['cut_rejected']}\t{tot['cut_written']}\t{tot['saved_s']:.0f}\t"
          f"{tot['cut_rejected_inst_later_written']}\t{tot['total_over_cap_search_under']} ({tot['total_over_cap_written']})")
    n_nomed = sum(c["no_median"] for c in per.values())
    print(f"# attempts of aliases without a written attempt (uncapped): {n_nomed}")
    for k, ex in sorted(examples.items()):
        if any(e.startswith("WRITTEN") for e in ex) or only:
            print(f"# {k}: " + "; ".join(ex))


if __name__ == "__main__":
    if sys.argv[1] == "table":
        table(sys.argv[2], sys.argv[3], *(sys.argv[4:5] or ["primary"]))
    elif sys.argv[1] == "count":
        count(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "primary", tuple(sys.argv[4:]))
    else:
        raise SystemExit(__doc__)
