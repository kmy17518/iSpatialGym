#!/usr/bin/env python3
"""Slack notification + decision bridge for the first-demo pass (owner request 2026-09-16 17:35, channel #cursor-demos-chat
C0C2D1X2V8E - private, always by id). The coordinator posts through the Slack MCP; this script keeps the two source files
and renders every post so the channel and the repo never disagree.

    python scripts/isg/slack_bridge.py new --subtask S --question "..." --option "A) ..." --option "B) ..." --default "..." [--artefact P ...] [--context "..."]
    python scripts/isg/slack_bridge.py thread --id D-001 --ts 1726530000.123456       # remember the Slack thread of a decision
    python scripts/isg/slack_bridge.py answer --id D-001 --by slack|chat --answer "B - ..." [--commit sha]
    python scripts/isg/slack_bridge.py hold --id D-001 | reopen --id D-001
    python scripts/isg/slack_bridge.py render --id D-001                              # the [DECISION] post text
    python scripts/isg/slack_bridge.py status                                         # regenerate status.md, print the [STATUS hh:mm] post
    python scripts/isg/slack_bridge.py valid                                          # [DEMO-VALID] posts for rows not yet announced
    python scripts/isg/slack_bridge.py mark-valid --subtask S --ts TS                 # after posting
    python scripts/isg/slack_bridge.py parse --text "D-004: B - only after ..."       # classify an owner reply (JSON)
    python scripts/isg/slack_bridge.py note --text "..." [--ts TS]                    # owner free text -> TODO activity log

Files: ``runlogs/pilot_20260915/pending_decisions.json`` (one record per decision: id, time, subtask, question, options,
default, status open|answered|held, answered_by slack|chat, answer, applied_commit, thread_ts; ids D-001, D-002, ... never
reused), ``runlogs/pilot_20260915/status.md`` (regenerated every 30 min), ``runlogs/pilot_20260915/slack_state.json``
(announced [DEMO-VALID] rows, posted message ts). Tags: [DECISION D-0xx], [DEMO-VALID], [STATUS hh:mm], [BLOCKED];
owner replies start with D-, Q:, RUN: or free text; the coordinator never acts on its own posts.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
import first_demo as fd  # noqa: E402

ROOT = os.path.join(REPO, "runlogs", "pilot_20260915")
DECISIONS = os.path.join(ROOT, "pending_decisions.json")
STATUS_MD = os.path.join(ROOT, "status.md")
STATE = os.path.join(ROOT, "slack_state.json")
QUEUE = os.path.join(ROOT, "ready_queue.json")
IDLE_MD = os.path.join(ROOT, "idle_pods.md")
SCHED_LOG = os.path.join(ROOT, "pod_scheduler.log")
POOL_FILE = os.path.join(REPO, "runlogs", "runpod", "smoke", "pods.txt")
BAND_FILE = os.path.join(REPO, "runlogs", "runpod", "rb", "pods.txt")
LOCKS = os.path.join(REPO, "runlogs", "runpod", "smoke", "locks")
CHANNEL = "C0C2D1X2V8E"
RUN_MINUTES = 75  # typical sample-one + demo wall time on a pool pod


def now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M PDT")


def short(st):
    return st.replace("inter_object_relation_satisfaction-", "IORS ").replace("configuration_matching-", "CM ")


def load(path, default):
    return json.load(open(path)) if os.path.exists(path) else default


def save(path, obj):
    json.dump(obj, open(path, "w"), indent=1)


# ------------------------------------------------------------------------------------------------------ decisions
def cmd_new(a):
    d = load(DECISIONS, {"decisions": []})
    n = max([int(x["id"].split("-")[1]) for x in d["decisions"]] + [0]) + 1
    rec = {"id": f"D-{n:03d}", "time": now(), "subtask": a.subtask, "question": a.question, "context": a.context or "", "options": a.option or [],
           "default": a.default or "", "artefacts": a.artefact or [], "status": "open", "answered_by": None, "answer": None, "applied_commit": None,
           "thread_ts": None, "answered_at": None}
    if a.answered:  # decided already (e.g. in the main chat) - recorded for the channel history, posted as answered
        rec.update(status="answered", answered_by=a.answered_by or "chat", answer=a.answered, applied_commit=a.commit, answered_at=a.answered_at or now())
    d["decisions"].append(rec)
    save(DECISIONS, d)
    print(rec["id"])
    print(render(rec))


def find(d, did):
    for x in d["decisions"]:
        if x["id"] == did:
            return x
    raise SystemExit(f"{did} not found")


def render(rec):
    lines = [f"[DECISION {rec['id']}] {short(rec['subtask'])} — {rec['question']}"]
    if rec.get("context"):
        lines += [rec["context"]]
    lines += list(rec.get("options") or [])
    if rec.get("default"):
        lines.append(f"default: {rec['default']}")
    if rec.get("artefacts"):
        lines.append("artefacts: " + ", ".join(f"`{p}`" for p in rec["artefacts"]))
    if rec.get("status") == "answered":
        lines.append(f"answered ({rec.get('answered_by')}, {rec.get('answered_at')}): {rec.get('answer')}" + (f" — applied in {rec['applied_commit']}" if rec.get("applied_commit") else ""))
    elif rec.get("status") == "held":
        lines.append("status: held (owner)")
    return "\n".join(lines)


def cmd_render(a):
    print(render(find(load(DECISIONS, {"decisions": []}), a.id)))


def cmd_thread(a):
    d = load(DECISIONS, {"decisions": []})
    find(d, a.id)["thread_ts"] = a.ts
    save(DECISIONS, d)


def cmd_answer(a):
    d = load(DECISIONS, {"decisions": []})
    r = find(d, a.id)
    if r["status"] == "answered" and r.get("answered_by") == "chat" and a.by == "slack":
        print(f"{a.id} already answered in the main chat ({r['answer']}) - chat overrides Slack; not changed")
        return
    r.update(status="answered", answered_by=a.by, answer=a.answer, applied_commit=a.commit, answered_at=now())
    save(DECISIONS, d)
    print(f"[DECISION {a.id}] answered: {a.answer}" + (f" — applied in {a.commit}" if a.commit else " — being applied"))


def cmd_hold(a):
    d = load(DECISIONS, {"decisions": []})
    find(d, a.id)["status"] = "held"
    save(DECISIONS, d)


def cmd_reopen(a):
    d = load(DECISIONS, {"decisions": []})
    find(d, a.id)["status"] = "open"
    save(DECISIONS, d)


# --------------------------------------------------------------------------------------------------------- status
def pool_runs():
    """pod -> (alias, kind, pid) for every remote_run holding a pool lock (via its open lock fd)."""
    out = {}
    try:
        pids = subprocess.run(["pgrep", "-f", "remote_run.py"], capture_output=True, text=True).stdout.split()
    except Exception:  # noqa: BLE001
        return out
    for pid in pids:
        try:
            fds = os.listdir(f"/proc/{pid}/fd")
        except Exception:  # noqa: BLE001
            continue
        pod = None
        for f in fds:
            try:
                t = os.readlink(f"/proc/{pid}/fd/{f}")
            except Exception:  # noqa: BLE001
                continue
            if t.startswith(LOCKS) and t.endswith(".lock"):
                pod = os.path.basename(t)[:-5]
        if not pod:
            continue
        ppid = subprocess.run(["ps", "-o", "ppid=", "-p", pid], capture_output=True, text=True).stdout.strip()
        cmd = subprocess.run(["ps", "-o", "args=", "-p", ppid], capture_output=True, text=True).stdout.strip() if ppid else ""
        m = re.search(r"--alias (\S+)", cmd) or re.search(r"--tro (\S+)", cmd)
        alias = (m.group(1) if m and "--alias" in m.group(0) else (os.path.basename(m.group(1)).split("__")[0] if m else "?"))
        out[pod] = (alias, "demo-only" if "demo-only" in cmd else "run", int(pid))
    return out


def local_run():
    ps = subprocess.run(["pgrep", "-af", "first_demo.py run .*--local"], capture_output=True, text=True).stdout
    for line in ps.splitlines():
        if "bash" in line.split(None, 2)[1:2] or "pgrep" in line:
            continue
        m = re.search(r"--alias (\S+)", line)
        if m:
            return m.group(1)
    return None


def balance_line():
    try:
        out = subprocess.run([sys.executable, os.path.join(REPO, "scripts", "runpod", "rp.py"), "balance"], capture_output=True, text=True, cwd=REPO, timeout=60).stdout
        for line in out.splitlines():
            if line.startswith("balance"):
                return line.strip()
    except Exception:  # noqa: BLE001
        pass
    return "balance: n/a"


def board_states():
    """subtask -> (state, verdict, artefact, commit) from the board file."""
    out = {}
    if not os.path.exists(fd.BOARD):
        return out
    for line in open(fd.BOARD):
        if not line.startswith("| ") or line.startswith("| subtask") or line.startswith("| time"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 8 and cells[0] in fd.SUBTASKS:
            m = re.search(r"run @ ([0-9a-f]{7})", cells[4])
            out[cells[0]] = {"state": cells[1], "blocking": cells[2], "owner": cells[3], "verdict": cells[5], "artefact": cells[6], "commit": m.group(1) if m else None, "last": cells[7]}
    return out


def last_tick():
    if not os.path.exists(SCHED_LOG):
        return ""
    ticks = [l for l in open(SCHED_LOG) if " tick: " in l]
    return ticks[-1].strip() if ticks else ""


def cmd_status(a):
    fleet = load(os.path.join(REPO, "runlogs", "runpod", "fleet.json"), {"pods": {}})["pods"]
    names = {p: (fleet.get(p) or {}).get("name", "?") for p in fleet}
    states = board_states()
    n_valid = sum(1 for s in states.values() if s["state"] == "valid demo")
    runs = pool_runs()
    pool = [l.strip() for l in open(POOL_FILE) if l.strip()] if os.path.exists(POOL_FILE) else []
    band = [l.strip() for l in open(BAND_FILE) if l.strip()] if os.path.exists(BAND_FILE) else []
    q = load(QUEUE, {"entries": [], "holds": {}})
    ready = [e for e in q["entries"] if e.get("state") == "ready"]
    running = [e for e in q["entries"] if e.get("state") == "running"]
    loc = local_run()
    tick = last_tick()
    m = re.search(r"bands (\d+) \((\d+) idle\)", tick)
    band_busy = (int(m.group(1)) - int(m.group(2))) if m else len(band)
    idle_pool = [p for p in pool if p not in runs]
    customs = json.load(open(fd.CUSTOMS))
    decisions = load(DECISIONS, {"decisions": []})["decisions"]
    open_ids = [d["id"] for d in decisions if d["status"] in ("open", "held")]
    bal = balance_line()
    t = dt.datetime.now()
    # ---- status.md (full)
    lines = [f"# Status {now()}", "", f"**{n_valid} / 21 subtasks with a valid demo.** {bal}", "", "## Runs in flight", "", "| pod | name | role | run | since | ETA |", "|---|---|---|---|---|---|"]
    for pod, (alias, kind, pid) in sorted(runs.items(), key=lambda kv: kv[1][0]):
        since = next((e.get("launched_at", "") for e in running if e.get("alias") == alias), "")
        eta = ""
        if since:
            try:
                eta = (dt.datetime.strptime(since, "%Y-%m-%d %H:%M PDT") + dt.timedelta(minutes=RUN_MINUTES)).strftime("~%H:%M")
            except Exception:  # noqa: BLE001
                eta = ""
        lines.append(f"| {pod} | {names.get(pod, '?')} | pool | {kind} {alias} ({short(fd.subtask_of(alias, customs))}) | {since[-9:] if since else 'agent run'} | {eta} |")
    if loc:
        lines.append(f"| local GPU | 4090 | local | run {loc} | | |")
    if band_busy:
        lines.append(f"| {', '.join(band) if band else '-'} | | band | {band_busy} band job(s) running ({os.path.basename(tick) and tick.split(' tick: ')[-1]}) | | |")
    if not runs and not loc and not band_busy:
        lines.append("| - | - | - | nothing running | | |")
    lines += ["", f"## Idle pods: {len(idle_pool)} pool" + (f", band {m.group(2)}" if m else ""), ""]
    if os.path.exists(IDLE_MD):
        why = [l for l in open(IDLE_MD) if l.startswith("| ") and "| pool |" in l]
        if why:
            lines.append("reason: " + why[0].split("|")[5].strip())
        holds = q.get("holds", {})
        if holds:
            lines.append("holds: " + "; ".join(f"{short(s)}: {h['reason'][:100]}" for s, h in holds.items()))
    lines += ["", f"## Queued runs: {len(ready)}", ""]
    for e in ready:
        lines.append(f"- prio {e.get('priority')} {e['kind']} {e['alias']} — next scheduler tick (≤ 5 min), ETA ~{RUN_MINUTES} min after launch")
    lines += ["", "## Board", "", "| subtask | state | blocking | owner |", "|---|---|---|---|"]
    for st in fd.SUBTASKS:
        s = states.get(st, {})
        lines.append(f"| {short(st)} | {s.get('state', '?')} | {s.get('blocking', '')[:140]} | {s.get('owner', '')} |")
    lines += ["", f"## Open decisions: {', '.join(open_ids) or 'none'}", ""]
    for d in decisions:
        if d["status"] in ("open", "held"):
            lines.append(f"- {d['id']} ({d['status']}) {short(d['subtask'])}: {d['question']} — default: {d['default']}")
    with open(STATUS_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    # ---- compact Slack post (<= 12 lines)
    by_role = {}
    for pod, (alias, kind, pid) in runs.items():
        by_role.setdefault("pool", []).append(f"{short(fd.subtask_of(alias, customs))}@{names.get(pod, pod)[-9:]}")
    post = [f"[STATUS {t.strftime('%H:%M')}] valid {n_valid}/21 · {bal.replace('   ', ' · ')}",
            f"pool {len(pool)} pods: {len(runs)} running, {len(idle_pool)} idle · band {len(band)} pod(s), {band_busy} running · local GPU: {('run ' + short(fd.subtask_of(loc, customs))) if loc else 'free'}"]
    if by_role.get("pool"):
        post.append("running: " + ", ".join(sorted(by_role["pool"]))[:900])
    post.append(f"queued: {len(ready)}" + (" — " + ", ".join(short(e['subtask']) for e in ready)[:300] if ready else "") + f" · holds: {len(q.get('holds', {}))} ({', '.join(short(s) for s in q.get('holds', {}))})"[:400])
    valid_rows = [short(s) for s, v in states.items() if v["state"] == "valid demo"]
    post.append("valid: " + (", ".join(valid_rows) or "-"))
    waiting = [short(s) for s, v in states.items() if v["state"] != "valid demo" and s not in q.get("holds", {}) and s not in {fd.subtask_of(a2, customs) for a2, _, _ in runs.values()}]
    post.append("waiting for a fix / not running: " + (", ".join(waiting)[:600] or "-"))
    if idle_pool and os.path.exists(IDLE_MD):
        why = [l for l in open(IDLE_MD) if l.startswith("| ") and "| pool |" in l]
        if why:
            post.append("idle reason: " + why[0].split("|")[5].strip()[:300])
    post.append(f"open decisions: {', '.join(open_ids) or 'none'}")
    print("\n".join(post[:12]))


# ---------------------------------------------------------------------------------------------------- demo-valid
def cmd_valid(a):
    st = load(STATE, {"announced_valid": {}, "posts": []})
    states = board_states()
    ledger = fd.read_jsonl(fd.LEDGER)
    for s, v in states.items():
        if v["state"] != "valid demo":
            continue
        key = f"{s}@{v.get('commit')}"
        if key in st["announced_valid"]:
            continue
        rec = [r for r in ledger if r["subtask"] == s and (r.get("demo") or {}).get("success") and (r.get("code_commit") or "").startswith(v.get("commit") or "")]
        r = rec[-1] if rec else None
        alias = r["alias"] if r else "?"
        meta = (r.get("demo") or {}).get("meta") if r else v.get("artefact")
        print(f"[DEMO-VALID] {short(s)} — {alias} @{v.get('commit')} — meta: {meta}")
        print(f"::key {key}")


def cmd_mark_valid(a):
    st = load(STATE, {"announced_valid": {}, "posts": []})
    st["announced_valid"][a.key] = {"ts": a.ts, "time": now()}
    save(STATE, st)


# ---------------------------------------------------------------------------------------------------------- parse
def parse_reply(text):
    t = text.strip()
    # own posts: tagged top-level posts and thread replies ([A D-0xx], [RUN ...], [DECISION ...] answered), and anything the
    # Slack MCP stamps with its "Sent using (Cursor)" footer - never acted on
    if re.match(r"^\[(DECISION|DEMO-VALID|STATUS|BLOCKED|A |RUN |Q |NOTE)", t) or re.search(r"Sent using.*\(Cursor\)", t):
        return {"kind": "own_post"}
    m = re.match(r"^(D-\d{3})\s*:\s*(.*)$", t, re.S)
    if m:
        did, rest = m.group(1), m.group(2).strip()
        if rest.lower() == "hold":
            return {"kind": "decision", "id": did, "action": "hold"}
        m2 = re.match(r"^([A-Da-d])\b\s*(?:[—\-–]+\s*(.*))?$", rest, re.S)
        if m2:
            return {"kind": "decision", "id": did, "action": "option", "option": m2.group(1).upper(), "constraint": (m2.group(2) or "").strip()}
        return {"kind": "decision", "id": did, "action": "free_text", "answer": rest}
    if re.match(r"^Q\s*:", t):
        return {"kind": "question", "text": re.sub(r"^Q\s*:\s*", "", t)}
    m = re.match(r"^RUN\s*:\s*(\S+)(.*)$", t, re.S)
    if m:
        args = m.group(2)
        ma = re.search(r"--max-attempts\s+(\d+)", args)
        ms = re.search(r"--scene\s+(\S+)", args)
        return {"kind": "run", "alias": m.group(1), "max_attempts": int(ma.group(1)) if ma else 8, "scene": ms.group(1) if ms else None}
    return {"kind": "note", "text": t}


def cmd_parse(a):
    print(json.dumps(parse_reply(a.text), indent=1))


def cmd_note(a):
    with open(os.path.join(REPO, "TODO.md"), "a") as f:
        f.write(f"- {now()} owner (Slack note{' ts ' + a.ts if a.ts else ''}): {a.text.strip()}\n")
    print("noted")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("new")
    s.add_argument("--subtask", required=True)
    s.add_argument("--question", required=True)
    s.add_argument("--context", default="")
    s.add_argument("--option", action="append")
    s.add_argument("--default", default="")
    s.add_argument("--artefact", action="append")
    s.add_argument("--answered", default=None, help="record as already answered (main chat) with this text")
    s.add_argument("--answered-by", default="chat")
    s.add_argument("--answered-at", default=None)
    s.add_argument("--commit", default=None)
    s.set_defaults(fn=cmd_new)
    s = sub.add_parser("render")
    s.add_argument("--id", required=True)
    s.set_defaults(fn=cmd_render)
    s = sub.add_parser("thread")
    s.add_argument("--id", required=True)
    s.add_argument("--ts", required=True)
    s.set_defaults(fn=cmd_thread)
    s = sub.add_parser("answer")
    s.add_argument("--id", required=True)
    s.add_argument("--by", choices=["slack", "chat"], required=True)
    s.add_argument("--answer", required=True)
    s.add_argument("--commit", default=None)
    s.set_defaults(fn=cmd_answer)
    s = sub.add_parser("hold")
    s.add_argument("--id", required=True)
    s.set_defaults(fn=cmd_hold)
    s = sub.add_parser("reopen")
    s.add_argument("--id", required=True)
    s.set_defaults(fn=cmd_reopen)
    s = sub.add_parser("status")
    s.set_defaults(fn=cmd_status)
    s = sub.add_parser("valid")
    s.set_defaults(fn=cmd_valid)
    s = sub.add_parser("mark-valid")
    s.add_argument("--key", required=True)
    s.add_argument("--ts", required=True)
    s.set_defaults(fn=cmd_mark_valid)
    s = sub.add_parser("parse")
    s.add_argument("--text", required=True)
    s.set_defaults(fn=cmd_parse)
    s = sub.add_parser("note")
    s.add_argument("--text", required=True)
    s.add_argument("--ts", default=None)
    s.set_defaults(fn=cmd_note)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
