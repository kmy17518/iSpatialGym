"""i7-diag (2026-11-08, owner 5m "early-reject audit"): CPU-only, read-only. Where the sampler's rejected attempts spend their seconds.

scan  (on the cluster: ssh minyeong@svl11 python3 - scan <tagspec> ... < audit_early_reject.py > attempts.jsonl)
      tagspec = <tag> or <tag>:<alias-key glob> (e.g. int1_7, i7i2_j2A:*inside_large_scale_02). One JSON row per attempt
      ("kind": "attempt": instance, attempt, outcome written / rejected / unfinished, stage, reason, the reject string, the
      per-attempt "stages:" seconds (reset / sample / filter / apply / settle / goal / program / write / occlusion / total), the
      start draw seconds, the attempt's kit counters) and one per job ("kind": "job": sacct state / elapsed, job_stats).
aliases (local: python3 audit_early_reject.py aliases attempts.jsonl) per alias of the primary set (every int1_* job + per alias
      the latest tag whose jobs all completed): written, rejected seconds split draw / search / post-sample, top classes.
items (local: python3 audit_early_reject.py items attempts.jsonl) the ranked early-reject list (owner 5m), est. GPU-s saved per
      written demo under each item's stated rule.
seg   (cluster) per attempt counts of sample.log lines matching given regexes.

The sampler logs, per attempt, "instance I attempt A rejected in <stage> after <T>s: [<reason>] <why>" and
"instance I attempt A stages: reset=..s sample=..s ... total=..s" (total == the reject's T == job_stats attempts_s summed);
the start draw ("start_per_attempt: start drawn in ..s") runs inside the attempt's sample stage. A start drawn with no
stages line after it is an attempt the job's end (time limit / cancel) cut: "unfinished", seconds unknown from the log."""
import collections, fnmatch, glob, json, os, re, subprocess, sys

ROOTS = ["/vision/u/minyeong/projects/isg_home/oracle_fix/out", "/vision/u/shiyuc/minyeong/isgf_root/out"]
REJ = re.compile(r"instance (\d+) attempt (\d+) rejected in (\w+) after ([\d.]+)s: \[(\w+)\] ?(.*)$")
STG = re.compile(r"instance (\d+) attempt (\d+) stages: (.*)$")
WRT = re.compile(r"instance (\d+) written \(attempt (\d+), ([\d.]+)s\)")
KIT = re.compile(r"instance (\d+) attempt (\d+) kit: (.*)$")
SD = re.compile(r"start_per_attempt: start drawn in ([\d.]+)s")
# per-attempt line counters (the attempt's lines = after the previous attempt's "stages:" line, up to its own)
COUNT = {"cam": re.compile(r"wrist camera within"), "blame": re.compile(r"CAMERA_BLAME"),
         "cplace": re.compile(r"C place IK \(attached\) fail"),
         "cplace_j2": re.compile(r"C place IK \(attached\) fail.*pose (pre-place|place): ik_joint_limit: (right|left)_arm_joint2 ")}


def read(p):
    try:
        return open(p, errors="replace").read()
    except OSError:
        return ""


def job_dirs(spec):
    tag, _, pat = spec.partition(":")
    out, seen = [], set()
    for r in ROOTS:
        for d in sorted(glob.glob(f"{r}/{tag}/*/g*")):
            key = d.split("/")[-2]
            if pat and not fnmatch.fnmatch(key, pat):
                continue
            rel = d[len(r):]
            if rel in seen or not os.path.exists(d + "/sample.log"):
                continue
            seen.add(rel)
            out.append((tag, key, d))
    return out


def sacct(jid):
    if not jid:
        return "?", 0, ""
    res = subprocess.run(["sacct", "-j", jid, "-X", "-n", "-P", "-o", "JobID,State,ElapsedRaw,End"], capture_output=True,
                         text=True).stdout
    for l in res.splitlines():
        p = l.split("|")
        if len(p) == 4 and p[0] == jid and p[2].isdigit():
            return p[1], int(p[2]), p[3]
    return "?", 0, ""


def parse(tag, key, d):
    lines = read(d + "/sample.log").splitlines()
    att = collections.OrderedDict()
    pend_sd = None
    pend_cnt = collections.Counter()
    for l in lines:
        for name, rx in COUNT.items():
            if rx.search(l):
                pend_cnt[name] += 1
        m = SD.search(l)
        if m:
            pend_sd = float(m.group(1))
            continue
        m = REJ.search(l)
        if m:
            k = (int(m.group(1)), int(m.group(2)))
            a = att.setdefault(k, {"start_draw_s": pend_sd})
            a.update(outcome="rejected", stage=m.group(3), t=float(m.group(4)), reason=m.group(5), why=m.group(6)[:600])
            continue
        m = WRT.search(l)
        if m:
            k = (int(m.group(1)), int(m.group(2)))
            a = att.setdefault(k, {"start_draw_s": pend_sd})
            a.update(outcome="written", stage="write", t=float(m.group(3)), reason="written", why="")
            continue
        m = STG.search(l)
        if m:
            k = (int(m.group(1)), int(m.group(2)))
            a = att.setdefault(k, {"start_draw_s": pend_sd})
            a["stages"] = {s: float(v) for s, v in re.findall(r"(\w+)=([\d.]+)s", m.group(3))}
            a["lines"] = dict(pend_cnt)
            pend_sd = None
            pend_cnt = collections.Counter()
            continue
        m = KIT.search(l)
        if m:
            k = (int(m.group(1)), int(m.group(2)))
            a = att.setdefault(k, {"start_draw_s": pend_sd})
            kc = a.setdefault("kit", {})
            for name, n in re.findall(r"([\w.]+) \((\d+)\)", m.group(3)):
                kc[name] = kc.get(name, 0) + int(n)
            ks = a.setdefault("kit_s", {})  # "ik_batch 69 calls 394.3s": the attempt's seconds per kit item
            for name, n, sec in re.findall(r"([\w.]+) (\d+) calls ([\d.]+)s", m.group(3)):
                ks[name] = round(ks.get(name, 0.0) + float(sec), 2)
    rows = []
    for (i, n), a in att.items():
        if "outcome" not in a:
            a.update(outcome="unfinished", stage="?", t=a.get("stages", {}).get("total"), reason="unfinished", why="")
        rows.append(dict(kind="attempt", tag=tag, key=key, job=d.split("/")[-1], inst=i, att=n, **a))
    if pend_sd is not None:  # a start drawn after the last finished attempt: cut by the job's end
        rows.append(dict(kind="attempt", tag=tag, key=key, job=d.split("/")[-1], inst=-1, att=-1, outcome="unfinished",
                         stage="?", t=None, reason="unfinished", why="start drawn, no stages line before the log ended",
                         start_draw_s=pend_sd))
    s = json.loads(read(d + "/job_stats.json") or "{}")
    jid = read(d + "/jobid").strip()
    st, el, end = sacct(jid)
    tr = re.search(r"trunk_[0-9a-f]+", read(d + "/job.env"))
    rows.append(dict(kind="job", tag=tag, key=key, job=d.split("/")[-1], dir=d, jobid=jid, state=st, elapsed_s=el, end=end,
                     trunk=tr.group(0) if tr else "?", attempts=s.get("attempts"), written=s.get("written"),
                     demos=s.get("demos"), successes=s.get("successes"), attempts_s=s.get("attempts_s"),
                     reject_s=s.get("reject_s"), written_s=s.get("written_s"),
                     world_load_s=(s.get("world_load_s") or {}).get("sum"), sample_wall_s=s.get("sample_wall_s"),
                     demo_wall_s=s.get("demo_wall_s"), rejections=s.get("rejections")))
    return rows


def scan(specs):
    for spec in specs:
        for tag, key, d in job_dirs(spec):
            for r in parse(tag, key, d):
                print(json.dumps(r))


def seg(spec, pats):
    """Per attempt of <tagspec>'s jobs: outcome, seconds, and how many sample.log lines inside the attempt match each regex
    (the attempt's lines = those after the previous attempt's "stages:" line, up to its own)."""
    rx = [re.compile(p) for p in pats]
    for tag, key, d in job_dirs(spec):
        cnt, out = [0] * len(rx), None
        for l in read(d + "/sample.log").splitlines():
            for i, r in enumerate(rx):
                if r.search(l):
                    cnt[i] += 1
            m = REJ.search(l) or WRT.search(l)
            if m:
                out = (m.group(3), m.group(5)) if m.re is REJ else ("write", "written")
            m = STG.search(l)
            if m:
                tot = re.search(r"total=([\d.]+)s", m.group(3))
                print(json.dumps(dict(tag=tag, key=key, job=d.split("/")[-1], inst=int(m.group(1)), att=int(m.group(2)),
                                      t=float(tot.group(1)) if tot else None, stage=(out or ("?", "?"))[0],
                                      reason=(out or ("?", "?"))[1], counts=cnt)))
                cnt, out = [0] * len(rx), None
        if any(cnt):
            print(json.dumps(dict(tag=tag, key=key, job=d.split("/")[-1], inst=-1, att=-1, t=None, stage="?",
                                  reason="unfinished", counts=cnt)))


def load(path):
    rows = [json.loads(l) for l in open(path)]
    return [r for r in rows if r["kind"] == "job"], [r for r in rows if r["kind"] == "attempt"]


def primary(jobs):
    """The audit's primary set: every int1_* job, plus per alias key the latest tag whose jobs all COMPLETED with stats."""
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


def split(a):
    """A rejected attempt's seconds in three groups: draw (reset + start draw), search (the rest of sample),
    post (filter / apply / settle / goal / program / write / occlusion: after the sample stage)."""
    st = a.get("stages") or {}
    sd = a.get("start_draw_s") or 0.0
    draw = st.get("reset", 0.0) + sd
    search = max(0.0, st.get("sample", 0.0) - sd)
    post = sum(v for k, v in st.items() if k not in ("reset", "sample", "total"))
    return draw, search, post


def short(key):
    return key.split("-", 1)[-1]


def alias_table(path):
    """Per alias key of the primary set: written, rejected seconds by group, top rejection classes (seconds per written demo)."""
    jobs, atts = load(path)
    sel = primary(jobs)
    W = collections.Counter()
    G = collections.Counter()
    for j in jobs:
        if (j["tag"], j["key"]) in sel:
            W[j["key"]] += j["written"] or 0
            G[j["key"]] += j["elapsed_s"] or 0
    rej = collections.defaultdict(lambda: [0.0, 0.0, 0.0])
    wr = collections.Counter()
    cls = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0.0]))
    for a in atts:
        if (a["tag"], a["key"]) not in sel or not a.get("t"):
            continue
        if a["outcome"] == "written":
            wr[a["key"]] += a["t"]
            continue
        d, s_, p = split(a)
        r = rej[a["key"]]
        r[0] += d
        r[1] += s_
        r[2] += p
        c = cls[a["key"]][f"{a['stage']}/{a['reason']}"]
        c[0] += 1
        c[1] += a["t"]
    tot = [sum(v[i] for v in rej.values()) for i in range(3)]
    print(f"# primary set: {len(sel)} (tag, alias) pairs; rejected s draw {tot[0]:.0f} / search {tot[1]:.0f} / post {tot[2]:.0f}; "
          f"written-attempt s {sum(wr.values()):.0f}; written {sum(W.values())}; GPU-s {sum(G.values())}")
    print("alias\ttags\twritten\tGPU-h\trejected_s\tdraw_s\tsearch_s\tpost_s\trejected_s_per_written\ttop classes (n, s, s/written)")
    for k in sorted(rej, key=lambda k: -sum(rej[k])):
        r = rej[k]
        w = W[k]
        tags = ",".join(sorted(t for t, kk in sel if kk == k))
        top = "; ".join(f"{c} {n} {t:.0f} {t / w:.0f}" if w else f"{c} {n} {t:.0f} -"
                        for c, (n, t) in sorted(cls[k].items(), key=lambda x: -x[1][1])[:3])
        print(f"{short(k)}\t{tags}\t{w}\t{G[k] / 3600:.2f}\t{sum(r):.0f}\t{r[0]:.0f}\t{r[1]:.0f}\t{r[2]:.0f}\t"
              f"{sum(r) / w if w else float('nan'):.0f}\t{top}")


def _lsq(X, y):
    import numpy as np
    X = np.c_[np.ones(len(y)), np.array(X, float)]
    y = np.array(y, float)
    c = np.linalg.lstsq(X, y, rcond=None)[0]
    p = X @ c
    r2 = 1 - ((y - p) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-9)
    return c, r2


def items(path):
    """The ranked early-reject list: per item the rejections it addresses (n, seconds), the estimated seconds saved under
    the item's rule, and that per written demo of the item's runs (0 written: per the 4 demos each job asked for, '>=')."""
    jobs, atts = load(path)
    P = primary(jobs)
    jw = collections.Counter()
    for j in jobs:
        jw[(j["tag"], j["key"])] += j["written"] or 0
    wt = collections.defaultdict(list)  # written attempt seconds per key in the primary set
    for a in atts:
        if (a["tag"], a["key"]) in P and a["outcome"] == "written" and a.get("t"):
            wt[a["key"]].append(a["t"])

    def med(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2] if xs else None

    def pairs(spec):
        if spec == "P":
            return set(P)
        out = set()
        for sp in spec:
            t, _, kp = sp.partition(":")
            out |= {(jt, jk) for jt, jk in jw if jt == t and fnmatch.fnmatch(jk, kp or "*")}
        return out

    def top_s(rows, n):
        c = collections.Counter()
        for a in rows:
            c[short(a["key"])] += a["t"]
        return c.most_common(n)

    def pick(sp, pred):
        return [a for a in atts if (a["tag"], a["key"]) in sp and a.get("t") and a["outcome"] == "rejected" and pred(a)]

    res = []

    jok = collections.Counter()
    for j in jobs:
        jok[(j["tag"], j["key"])] += j["successes"] or 0

    def add(iid, title, sp, rows, est, rule, fix, status, extra="", label=None):
        w = sum(jw[x] for x in sp)
        ok = sum(jok[x] for x in sp)
        n_jobs = sum(1 for j in jobs if (j["tag"], j["key"]) in sp)
        tot = sum(a["t"] for a in rows)
        post = sum(split(a)[2] for a in rows)
        den, lb = (w, "") if w else (4 * n_jobs, ">=")
        res.append(dict(id=iid, title=title, runs=label or sorted({f"{t}:{short(k)}" for t, k in sp}), ok=ok, n=len(rows), rejected_s=round(tot), post_s=round(post),
                        est_s=round(est), written=w, per_written=f"{lb}{est / den:.0f}" if den else "-",
                        per_written_v=est / den if den else 0, addressable_per_written=f"{lb}{tot / den:.0f}" if den else "-",
                        rule=rule, fix=fix, status=status, extra=extra))

    # 1. IORS large floor aliases: the init layout is fixed per instance (start draw 0 s), so attempts re-search the same B / C
    sp = pairs(["i7diag_floor:*under_large_scale_03", "i7diag_floor:*ontop_large_scale_02"])
    rows = pick(sp, lambda a: a["reason"] == "target_placement" and ("no_transition" in a["why"] or "no push chain" in a["why"]
                                                                     or "carry leg failed" in a["why"]))
    mk = {k: med(a["t"] for a in atts if (a["tag"], a["key"]) in sp and a["key"] == k and a["outcome"] == "written")
          for k in {a["key"] for a in rows}}
    by_inst = collections.defaultdict(list)
    for a in rows:
        by_inst[(a["key"], a["inst"])].append(a)
    rep = [a for v in by_inst.values() for a in sorted(v, key=lambda a: a["att"])[1:]]
    add("FLOOR-memo", "IORS large floor (under_large_03, ontop_large_02): the same B init->manipulation path collision / C 'no push "
        "chain' / C 'carry leg failed' repeated over an instance's attempts", sp, rows, sum(max(0.0, a["t"] - mk[a["key"]]) for a in rep),
        "memo: the repeats after the instance's first such rejection, each down to the alias's median written attempt ("
        + ", ".join(f"{short(k)} {v:.0f} s" for k, v in mk.items()) + ")",
        "per-instance memo (across attempts) of dead B (transition path_collision at the same swept pose) and dead C (d, lambda: no push "
        "chain / carry leg), as generation.c_skip_dead_b does within an attempt; or redraw the floor init layout after 2 such attempts",
        "new", "instances: " + ", ".join(f"{short(k)} {i} x{len(v)}" for (k, i), v in by_inst.items()))

    # 2. adaptive per-attempt time cap
    sp = pairs("P")
    lost, rows, est, top = 0, [], 0.0, collections.Counter()
    for a in atts:
        if (a["tag"], a["key"]) not in sp or not a.get("t"):
            continue
        m = med(wt[a["key"]])
        if m is None:
            continue
        cap = max(300.0, 5 * m)
        if a["t"] > cap:
            if a["outcome"] == "written":
                lost += 1
            else:
                rows.append(a)
                est += a["t"] - cap
                top[short(a["key"])] += a["t"] - cap
    add("CAP", "attempts running far past the alias's written attempts (incl. the time_budget stalls)", sp, rows, est,
        "cap: per-attempt budget max(300 s, 5 x the alias's median written attempt); est = the seconds past the cap",
        "adaptive attempt budget from the alias memory's written-attempt times (the fixed 360 / 2400 / 3000 / 3600 s budgets stay as "
        "ceilings)", "new", f"written attempts the cap would have cut: {lost}; saved s by alias (alias written): " +
        ", ".join(f"{k} {v:.0f} ({sum(jw[x] for x in sp if short(x[1]) == k)})" for k, v in top.most_common(5)),
        label=f"primary set ({len(sp)} tag / alias pairs)")

    # 3. attachment_03 rest_unstable (owner 5n.2: the nudge test right after the rest draw)
    sp = {x for x in P if x[1].endswith("attachment_03")}
    rows = pick(sp, lambda a: a["reason"] == "rest_unstable")
    add("AT03-rest", "attachment_03 rest_unstable at settle_validate", sp, rows,
        sum(a["t"] - split(a)[0] - 10.0 for a in rows), "draw: the nudge test (~10 s) right after the rest draw",
        "5n.2: the existing rest-nudge test right after the rest is drawn (switch)", "DECIDED 5n.2 (i7-cork)")

    # 4. block_design board positions (owner 5m.2: BD_BOARD_EARLY / SEARCH default)
    sp = {x for x in P if x[1].endswith("block_design_01")}
    rows = pick(sp, lambda a: "board_placement" in a["why"])
    add("BD-board", "block_design_01 board_placement (no board position works)", sp, rows,
        sum(max(0.0, a["t"] - 2.5) for a in rows), "measured: BD_BOARD_EARLY 2.5 s per rejected attempt (bd_early 60 att, 149 s)",
        "5m.2: BD_BOARD_EARLY + BD_BOARD_SEARCH default (search: written on the first attempt, bd_fix 4 / 4)", "DECIDED 5m.2")

    # 5. ontop_large_02 approach_untrackable (owner candidate)
    sp = pairs(["i7diag_floor:*ontop_large_scale_02"])
    rows = pick(sp, lambda a: a["reason"] == "approach_untrackable")
    add("ON02-track", "ontop_large_02 approach_untrackable at program_check (carry chain 7, approach 2)", sp, rows,
        sum(split(a)[2] for a in rows), "post: the program_check's own tracked carry / approach chain run in stance C's acceptance; saves "
        "the post-sample stages (the search to that C is spent either way)",
        "track the carry leg (dense poses, from where the previous motion leaves the arm) and the approach in the C acceptance, "
        "so the C loop moves on; memo the (B trunk, C) pairs whose carry chain fails", "owner candidate",
        f"upper bound (whole attempts) {sum(a['t'] for a in rows):.0f} s; the alias's sample stage is 104-1533 s per attempt "
        "(C tries 'skipped - the carry leg failed at this base pose')")

    # 6. touching_03 opryhp attached lift (owner candidate)
    sp = pairs(["t03specs_T6:*"])
    rows = pick(sp, lambda a: a["reason"] == "lift_unreachable")
    add("T03-lift", "touching_03 opryhp_0 goal_check lift_unreachable (the lift with the book attached)", sp, rows,
        sum(split(a)[2] for a in rows), "post: the attached-lift IK at the candidate's acceptance in sample (before apply / settle / goal)",
        "check the attached lift (and (d) the straightening lift, 5k.2) at the draw; the alias writes 0 - the lift geometry, "
        "not the check's place, is the blocker", "owner candidate (5k.2 A/B pending)")

    # 7. inside_large_02 joint-2 at the place (owner candidate): regression of attempt seconds on the joint-2 limit rejects
    fr = ["i7i2_j2A:*", "i7i2_fridgeA:*", "i7i2_fridgeA2:*", "i7i2_fridge_def:*", "i7_fridge4:*", "i7i2_fridgeB:*", "i7i2_fridgeB2:*",
          "i7i2_j2B:*"]
    sp = {x for x in pairs(fr) if x[1].endswith("inside_large_scale_02")}
    ar = [a for a in atts if (a["tag"], a["key"]) in sp and a.get("t")]

    def kj2(a):
        k = a.get("kit") or {}
        return k.get("limit_reject.right_arm_joint2", 0) + k.get("limit_reject.left_arm_joint2", 0)

    def kother(a):
        return sum(v for kk, v in (a.get("kit") or {}).items() if kk.startswith("limit_reject.") and not kk.endswith("joint2"))
    c, r2 = _lsq([[kj2(a), kother(a)] for a in ar], [a["t"] for a in ar])
    n2 = sum(kj2(a) for a in ar)
    nB = sum(kj2(a) for a in ar if a["tag"] == "i7i2_j2B")
    add("IN02-j2", "inside_large_02 arm joint-2 limit at the place / pre-place (exact IK at C)", sp,
        [a for a in ar if kj2(a) and a["outcome"] == "rejected"], c[1] * n2,
        f"regress: attempt s = {c[0]:.0f} + {c[1]:.1f} x joint-2 limit rejects + {c[2]:.1f} x other limit rejects (R2 {r2:.2f}, "
        f"{len(ar)} attempts); est = {c[1]:.1f} s x {n2} joint-2 rejects", "the 50 mrad joint margin in the interior place map's IK "
        "(the map has none: 268 / 349 C place failures, median 3 mrad from the limit with collisions off) + 5t's inside arm side",
        "owner candidate; 5t (arm side, inside only) DECIDED", f"ARM_SIDE arm i7i2_j2B: {nB} joint-2 rejects for 4 written -> "
        f"{c[1] * nB / 4:.0f} s / written remain for the map margin", label="i7_fridge4, i7i2_fridgeA / A2 / B / B2 / _def, "
        "i7i2_j2A, i7i2_j2B (ARM_SIDE)")

    # 8. the sink's wrist-camera narrow failures at the pick (owner candidate)
    sp = {x for x in pairs(["i7i2_sink:*", "i7i2_sinkdiag2:*", "ch_sh:*"]) if x[1].endswith("inside_large_scale_03")}
    ar = [a for a in atts if (a["tag"], a["key"]) in sp and a.get("t")]
    c, r2 = _lsq([[(a.get("lines") or {}).get("cam", 0)] for a in ar], [a["t"] for a in ar])
    ncam = sum((a.get("lines") or {}).get("cam", 0) for a in ar)
    d3 = [a for a in atts if a["tag"] == "i7i2_sinkdiag3" and a.get("t")]
    add("IN03-cam", "inside_large_03 (plate -> sink, bar) B pick: 'wrist camera within 5 mm' (narrow, 2-4 mm clear of bar_egwapq_0)",
        sp, [a for a in ar if (a.get("lines") or {}).get("cam")], c[1] * ncam,
        f"regress: attempt s = {c[0]:.0f} + {c[1]:.1f} x camera-fail lines (R2 {r2:.2f}, {len(ar)} attempts, CAMERA_BLAME off); "
        f"est = {c[1]:.1f} s x {ncam} lines", "i7-camhull's ISPATIALGYM_CAMERA_HULL_FALLBACK (ch_sh shadow: 27 of 27 narrow failures "
        "pass the hull) - a shared gate, owner question", "owner candidate (i7-camhull shadow done)",
        "i7i2_sinkdiag3 (CAMERA_BLAME on): " + ", ".join(f"{a['inst']}/{a['att']} {a['t']:.0f} s cam {(a.get('lines') or {}).get('cam', 0)}"
                                                          for a in d3) + " - the 30 min attempt WROTE; its blame probes add time")

    # 9. under_large_03 gripper_not_in_frame (owner candidate)
    sp = pairs(["i7diag_floor:*under_large_scale_03"])
    rows = pick(sp, lambda a: a["reason"] == "gripper_not_in_frame")
    add("U3-frame", "under_large_03 gripper_not_in_frame (goal_check grasp eef 3, program_check pre-place fingertip 3)", sp, rows,
        sum(split(a)[2] for a in rows), "post: the grasp-eef / pre-place fingertip projection in B / C's acceptance (camera FK only)",
        "project the grasp eef (settled pose) and the pre-place fingertip at the B / C acceptance - the same 5 % rule, no relaxation",
        "owner candidate", f"upper bound (whole attempts) {sum(a['t'] for a in rows):.0f} s")

    # 10. touching lift_unreachable at goal_check (owner 5r.1 lift screen default)
    sp = {x for x in P if re.search(r"touching_0[124]$", x[1])}
    rows = pick(sp, lambda a: a["stage"] == "goal_check" and a["reason"] == "lift_unreachable")
    add("T0x-lift", "touching_01/02/04 goal_check lift_unreachable", sp, rows, 0.61 * sum(a["t"] - split(a)[0] for a in rows),
        "draw: the screen acts at the draw; measured catch 0.61 (c3_fu_a vs base: 18 -> 7 lift rejections)",
        "5r.1: ISPATIALGYM_TOUCH_LIFT_SCREEN default", "DECIDED 5r.1")

    # 11. touching_04 contact_unreachable at goal_check (owner 5r: screen at the draw vs the push map)
    rows = pick(sp, lambda a: a["stage"] == "goal_check" and a["reason"] == "contact_unreachable")
    add("T0x-contact", "touching_01/02/04 goal_check contact_unreachable", sp, rows,
        sum(a["t"] - split(a)[0] for a in rows), "draw: screened against the push map at the draw",
        "5r follow-up: screen contact_unreachable at the draw against the push map (switch, N 4 A/B)", "DECIDED 5r (switch)")

    # 12. repeats_eval (all aliases): the repeat-of-eval test as soon as start + goal are drawn
    sp = pairs("P")
    rows = pick(sp, lambda a: a["reason"] == "repeats_eval")
    add("REPEAT", "instance_filter repeats_eval (all aliases)", sp, rows, sum(split(a)[2] + 0.5 * split(a)[1] for a in rows),
        "the post-sample seconds + half the search (the goal is drawn partway through sample)",
        "the repeats-eval comparison right after the start / goal draw, before the grasp / IK search; for axial_02 5o's free start edge",
        "new (5o.2 for axial_02)", label=f"primary set ({len(sp)} tag / alias pairs)")

    # 13. empty feasible set F at the focus (all aliases): is a memo worth it? (repeat rate of the focus cell + trunk in a job)
    FX = re.compile(r"empty feasible set F at the focus \[([-\d.]+), ([-\d.]+)\] \(trunk ([-\d.]+)/([-\d.]+)")
    rows = pick(sp, lambda a: a["reason"] == "target_placement" and FX.search(a["why"].split("; candidate")[-1]))
    seen, rep = collections.defaultdict(set), []

    def cell(m):
        return round(float(m[0]) / 0.05), round(float(m[1]) / 0.05), m[2], m[3]
    for a in sorted(rows, key=lambda a: (a["tag"], a["key"], a["job"], a["inst"], a["att"])):
        ms = FX.findall(a["why"])
        k = (a["tag"], a["key"], a["job"])
        if cell(ms[-1]) in seen[k]:
            rep.append(a)
        seen[k] |= {cell(m) for m in ms}
    add("FOCUS-F", "target_placement whose last candidate is 'empty feasible set F at the focus' (all aliases)", sp, rows,
        sum(a["t"] - split(a)[0] for a in rep), "memo: only the attempts whose focus cell (5 cm) + trunk already failed in the job",
        "none as an early check: the map / exact disagreement at the focus is new per draw (in-job repeat rate low); a map-side fix "
        "(the exact-IK margin in the maps) is the lever", "new (information)",
        f"repeats {len(rep)} of {len(rows)}; by alias (s): " + ", ".join(f"{k} {v:.0f}" for k, v in top_s(rows, 4)),
        label=f"primary set ({len(sp)} tag / alias pairs)")

    res.sort(key=lambda r: -r["per_written_v"])
    print("| rank | item | runs | n | rejected s | post-sample s | est. saved s | written / ok | est. s / written | addressable s / written | "
          "rule | proposed fix | status |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(res, 1):
        runs = r["runs"] if isinstance(r["runs"], str) else ", ".join(r["runs"])
        print(f"| {i} | {r['id']}: {r['title']} | {runs} | {r['n']} | {r['rejected_s']} | {r['post_s']} | {r['est_s']} | {r['written']} / {r['ok']} | "
              f"{r['per_written']} | {r['addressable_per_written']} | {r['rule']} | {r['fix']} | {r['status']}"
              + (f" ({r['extra']})" if r["extra"] else "") + " |")


if __name__ == "__main__":
    if sys.argv[1] == "scan":
        scan(sys.argv[2:])
    elif sys.argv[1] == "seg":  # seg <tagspec> <regex> [<regex> ...]
        seg(sys.argv[2], sys.argv[3:])
    elif sys.argv[1] == "aliases":  # aliases attempts.jsonl
        alias_table(sys.argv[2])
    elif sys.argv[1] == "items":  # items attempts.jsonl
        items(sys.argv[2])
