#!/usr/bin/env python3
"""No idle GPU (owner request 2026-09-16 16:40): a 5-minute loop that keeps the pool and the band fleet busy.

    python scripts/isg/pod_scheduler.py loop [--interval 300] [--dry-run]     # the guard (nohup it)
    python scripts/isg/pod_scheduler.py tick [--dry-run]                      # one pass
    python scripts/isg/pod_scheduler.py status                                # queue + idle report to stdout
    python scripts/isg/pod_scheduler.py add --alias A [--scene S] [--priority 40] [--kind run|demo-only] [--tro P] [--local] [--note ..]
    python scripts/isg/pod_scheduler.py hold --subtask S --reason "awaiting owner decision: ..."   # no auto entry
    python scripts/isg/pod_scheduler.py release --subtask S

Every pass:
(a) reads the pool locks (``runlogs/runpod/smoke/locks/<pod>.lock``, flock probe - the lock ``remote_run.py`` holds for the
    whole run) and the band pods' queue depth / current job (``/root/samp``; a current.txt job already in done.txt is
    finished, not running);
(b) maintains the **ready queue** ``runlogs/pilot_20260915/ready_queue.json`` ordered by priority: a subtask enters when
    every fix-log entry affecting it is in HEAD and its last run does not contain the newest such entry (validity rule
    of ``first_demo.py``), or when an agent ``add``s it explicitly; subtasks on ``hold`` (owner decision pending) and
    subtasks whose last run already carried every fix (nothing new to try) stay out and are explained in the idle
    report; coverage chunks never enter (paused);
(c) launches one queued run per free pool pod (``first_demo.py run`` / ``demo-only`` - the ledger, board and fix log stay
    current; ``remote_run`` takes the one-job-per-pod lock; the local GPU only runs entries marked ``local`` and one at a
    time), refills idle band pods from the remaining band manifests (``fleet_remaining.py`` + ``dispatch_samp.py``, never
    ``--rebalance``), and when one side has nothing left moves idle pods to the other role (``smoke/pods.txt``,
    ``rb/pods.txt``, ``pods_bands.txt``, ``paused_for_resume.md``; TODO.md activity line with pod ids + jobs moved);
(d) writes ``runlogs/pilot_20260915/idle_pods.md``: pod, role, idle since, why nothing was schedulable.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import datetime as dt
import fcntl
import glob
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(REPO, "scripts", "runpod"))
import first_demo as fd  # noqa: E402
import rp  # noqa: E402

ROOT = os.path.join(REPO, "runlogs", "pilot_20260915")
QUEUE = os.path.join(ROOT, "ready_queue.json")
IDLE_MD = os.path.join(ROOT, "idle_pods.md")
LOG = os.path.join(ROOT, "pod_scheduler.log")
STATE = os.path.join(ROOT, "pod_scheduler_state.json")
FD_LOGS = os.path.join(fd.FD, "logs")
POOL_FILE = os.path.join(REPO, "runlogs", "runpod", "smoke", "pods.txt")
LOCKS = os.path.join(REPO, "runlogs", "runpod", "smoke", "locks")
BAND_FILE = os.path.join(REPO, "runlogs", "runpod", "rb", "pods.txt")
BAND_ROLE_FILE = os.path.join(REPO, "runlogs", "runpod", "pods_bands.txt")
PAUSED = os.path.join(ROOT, "paused_for_resume.md")
TODO = os.path.join(REPO, "TODO.md")
RB = os.path.join(REPO, "runlogs", "reach_bands_20260915")
BAND_MANIFESTS_FILE = os.path.join(RB, "active_manifests.txt")  # one manifest path per line; default below
BAND_MANIFESTS_DEFAULT = ["jobs_q12_rest.json", "jobs_q13_a6_rerun.json", "jobs_under_edge_probe.json"]
FLEET_REMAINING = os.path.join(REPO, "scripts", "reach_bands", "fleet_remaining.py")
DISPATCH = os.path.join(REPO, "scripts", "runpod", "dispatch_samp.py")

# owner priority (16:40): P1 fix-landed re-runs first, then everything the validity rule re-enters (SUBTASKS order)
PRIORITY = {
    "configuration_matching-articulation_open_large_scale": 10, "configuration_matching-articulation_close_large_scale": 11,
    "inter_object_relation_satisfaction-ontop_large_scale": 12,
    "inter_object_relation_satisfaction-ontop_small_scale": 20, "disembedding": 21, "inter_object_relation_satisfaction-under_small_scale": 22,
    "inter_object_relation_satisfaction-touching": 30, "alignment-world_vertical": 31, "alignment-world_horizontal": 32,
}
# default loop alias per subtask (the first-demo alias) and the sample_isg scene restriction where one is needed
DEFAULT_ALIAS = {
    "camera_relocalization-standard": "camera_relocalization-standard", "camera_relocalization-obstructed": "camera_relocalization-obstructed-footstool_1",
    "object_scaling": "object_scaling-banana", "mental_rotation": "mental_rotation-banana",
    "disembedding": "disembedding-bottle_of_milk-bottle_of_milk_1-breakfast_table",
    "configuration_matching-block_design": "configuration_matching-block_design-breakfast_table",
    "configuration_matching-attachment": "configuration_matching-attachment-cork-wine_bottle-breakfast_table",
    "configuration_matching-articulation_open_small_scale": "configuration_matching-articulation_open_small_scale-hinged_jar-breakfast_table",
    "configuration_matching-articulation_close_small_scale": "configuration_matching-articulation_close_small_scale-hinged_jar-breakfast_table",
    "configuration_matching-articulation_open_large_scale": "configuration_matching-articulation_open_large_scale-bottom_cabinet",
    "configuration_matching-articulation_close_large_scale": "configuration_matching-articulation_close_large_scale-bottom_cabinet",
    "alignment-world_vertical": "alignment-world_vertical-painting_s070-walls", "alignment-world_horizontal": "alignment-world_horizontal-license_plate-walls",
    "alignment-axial": "alignment-axial-board_game-breakfast_table",
    "inter_object_relation_satisfaction-ontop_small_scale": "inter_object_relation_satisfaction-ontop_small_scale-pack_of_kielbasa-platter-breakfast_table",
    "inter_object_relation_satisfaction-under_small_scale": "inter_object_relation_satisfaction-under_small_scale-teacup-saucer-breakfast_table",
    "inter_object_relation_satisfaction-inside": "inter_object_relation_satisfaction-inside-brussels_sprouts-bowl-breakfast_table",
    "inter_object_relation_satisfaction-nextto": "inter_object_relation_satisfaction-nextto-marker-colored_pencil-breakfast_table",
    "inter_object_relation_satisfaction-touching": "inter_object_relation_satisfaction-touching-box_of_sanitary_napkins-box_of_tissues-breakfast_table",
    "inter_object_relation_satisfaction-ontop_large_scale": "inter_object_relation_satisfaction-ontop_large_scale-shampoo_dispenser-countertop-countertop",
    "inter_object_relation_satisfaction-under_large_scale": "inter_object_relation_satisfaction-under_large_scale-sandal-bed-bed",
}
DEFAULT_SCENE = {"configuration_matching-articulation_open_large_scale-bottom_cabinet": "office_cubicles_right",
                 "configuration_matching-articulation_close_large_scale-bottom_cabinet": "office_cubicles_right"}
LOCAL_ONLY = {"inter_object_relation_satisfaction-ontop_large_scale"}  # owner 08:19: ontop large-scale on the local GPU
ORACLE_ONLY_OWNERS = {"oracle"}  # fixes by these owners re-enter a subtask as demo-only on its last instance


def now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M PDT")


def log(msg):
    line = f"{now()} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def read_lines(path):
    return [l.strip() for l in open(path) if l.strip()] if os.path.exists(path) else []


def write_lines(path, lines):
    with open(path, "w") as f:
        f.write("".join(l + "\n" for l in lines))


def load_queue():
    if os.path.exists(QUEUE):
        return json.load(open(QUEUE))
    return {"updated": None, "entries": [], "holds": {}}


def save_queue(q):
    q["updated"] = now()
    q["entries"].sort(key=lambda e: (e.get("state") not in ("ready", "running"), e.get("priority", 50), e.get("added_at", "")))
    json.dump(q, open(QUEUE, "w"), indent=1)


def load_state():
    return json.load(open(STATE)) if os.path.exists(STATE) else {"idle_since": {}}


def save_state(s):
    json.dump(s, open(STATE, "w"), indent=1)


# ----------------------------------------------------------------------------------------------------------- pods
def live_pods(api):
    try:
        return {p["id"]: p for p in api.myself()["pods"]}
    except Exception as e:  # noqa: BLE001
        log(f"runpod api failed: {e}")
        return None


def lock_free(pid):
    """True when nobody holds the remote_run lock of this pool pod (flock probe; released at once)."""
    os.makedirs(LOCKS, exist_ok=True)
    fh = open(os.path.join(LOCKS, pid + ".lock"), "a")
    try:
        fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        return False
    fcntl.flock(fh, fcntl.LOCK_UN)
    fh.close()
    return True


# a job is in progress iff its .job file sits in /root/samp/running/ (the runner moves it there at the start and to finished/ at
# the end); current.txt is written at the start and never cleared between jobs, and done.txt keeps the lines of earlier
# attempts of a re-queued job - neither tells idle from busy
BAND_CMD = ("cat /root/samp/current.txt 2>/dev/null; echo ---; for f in /root/samp/running/*.job; do [ -f \"$f\" ] && cut -f1 \"$f\"; done; echo ---; ls /root/samp/queue 2>/dev/null; echo ---; "
            "kill -0 $(cat /root/samp/runner.pid 2>/dev/null) 2>/dev/null && echo RUNNER || echo NORUNNER; echo ---; test -f /root/BOOTSTRAP_DONE && echo BOOT")


def survey_band_pod(api, pid):
    ep = rp.pod_ssh(api, pid)
    if ep is None:
        return pid, None
    try:
        r = rp.ssh_run(*ep, BAND_CMD, capture=True, timeout=60)
    except SystemExit:
        return pid, None
    parts = [p.strip() for p in (r.stdout or "").split("---")]
    if len(parts) < 5:
        return pid, None
    cur, in_progress = parts[0], [l for l in parts[1].splitlines() if l]
    queue = [re.sub(r"^w\d+_\d+_", "", f)[: -len(".job")] if f.endswith(".job") else f for f in parts[2].splitlines() if f]
    running = in_progress[0] if in_progress else ""
    return pid, {"ep": ep, "running": running, "stale_current": bool(cur and not running), "queue": queue, "runner": parts[3] == "RUNNER", "boot": "BOOT" in parts[4]}


def survey_band(api, pods):
    out = {}
    with cf.ThreadPoolExecutor(min(16, max(1, len(pods)))) as ex:
        for pid, st in ex.map(lambda p: survey_band_pod(api, p), pods):
            out[pid] = st
    return out


def inflight_runs(customs):
    """first_demo runs alive on this machine (any agent's): subtask -> [(pid, kind, alias, local)]."""
    out = {}
    try:
        ps = subprocess.run(["pgrep", "-af", "first_demo.py (run|demo-only)"], capture_output=True, text=True).stdout
    except Exception:  # noqa: BLE001
        return out
    for line in ps.splitlines():
        if "first_demo.py" not in line or "pgrep" in line or line.split()[1:2] == ["/bin/bash"]:
            continue
        if line.split(None, 1)[1].startswith("/bin/bash") or "bash -c" in line:
            continue
        pid = int(line.split()[0])
        m = re.search(r"--alias (\S+)", line)
        alias = m.group(1) if m else None
        if alias is None:
            m = re.search(r"--tro (\S+)", line)
            alias = os.path.basename(m.group(1)).split("__")[0] if m else None
        if not alias:
            continue
        st = fd.subtask_of(alias, customs)
        out.setdefault(st, []).append({"pid": pid, "kind": "demo-only" if "demo-only" in line else "run", "alias": alias, "local": "--local" in line})
    return out


# ------------------------------------------------------------------------------------------------- ready queue
def fix_entries():
    return fd.read_jsonl(fd.FIXLOG)


def newest_missing(run_commit, fixes):
    """fixes (affecting a subtask) that the run's commit does not contain."""
    return [f for f in fixes if not fd.commit_contains(run_commit, f["commit"])]


def auto_enter(q, customs, inflight, head):
    """Apply the entry rule to every subtask; returns explanations for the subtasks that do not enter."""
    ledger, fixes = fd.read_jsonl(fd.LEDGER), fix_entries()
    manual = json.load(open(fd.BOARD_STATE)) if os.path.exists(fd.BOARD_STATE) else {}
    explain = {}
    queued = {e["subtask"] for e in q["entries"] if e.get("state") in ("ready", "running")}
    for st in fd.SUBTASKS:
        req = fd.fixes_for(st, fixes)
        hold = q.get("holds", {}).get(st)
        if hold and any(f["time"] > hold.get("since", "") for f in req):  # a fix logged after the hold lifts it (the decision landed)
            log(f"hold on {st} lifted: fix {[f['commit'][:7] for f in req if f['time'] > hold.get('since', '')]} logged after {hold.get('since')}")
            q["holds"].pop(st, None)
            hold = None
        if hold:
            explain[st] = f"hold: {hold['reason']}"
            continue
        not_in_head = [f["commit"][:7] for f in req if not fd.commit_contains(head, f["commit"])]
        if not_in_head:
            explain[st] = f"fix commit(s) not in HEAD: {', '.join(not_in_head)}"
            continue
        recs = sorted([r for r in ledger if r["subtask"] == st], key=lambda r: r["time"])
        valid = [r for r in recs if (r.get("demo") or {}).get("success") and not newest_missing(r.get("code_commit"), req)]
        if valid:
            explain[st] = "valid demo on the board"
            continue
        if st in inflight:
            explain[st] = f"run in flight (pid {inflight[st][0]['pid']}, {inflight[st][0]['kind']} {inflight[st][0]['alias']})"
            continue
        if st in queued:
            explain[st] = "already in the ready queue"
            continue
        last = recs[-1] if recs else None
        missing = newest_missing(last.get("code_commit"), req) if last else req
        if last and not missing:
            block = (manual.get(st) or {}).get("blocking", "")
            owner = (manual.get(st) or {}).get("owner", "")
            explain[st] = f"last run ({last['time'][-9:]}) already carried every fix and did not pass - waiting for a fix: {block[:160] or '-'} (owner: {owner or '-'})"
            continue
        # enters: pick the alias / kind
        alias = (last or {}).get("alias") or DEFAULT_ALIAS.get(st)
        if not alias:
            explain[st] = "no alias known"
            continue
        kind = "run"
        tro = None
        if last and last.get("tro") and os.path.exists(os.path.join(REPO, last["tro"])) and all((f.get("owner") or "") in ORACLE_ONLY_OWNERS for f in missing):
            kind, tro = "demo-only", last["tro"]
        e = {"id": f"{st}:{time.strftime('%m%d-%H%M%S')}", "priority": PRIORITY.get(st, 50 + fd.SUBTASKS.index(st)), "subtask": st, "alias": alias, "kind": kind, "tro": tro,
             "scene": DEFAULT_SCENE.get(alias), "local": st in LOCAL_ONLY, "state": "ready", "added_by": "rule",
             "added_at": now(), "note": f"auto: new fix(es) {', '.join(f['commit'][:7] for f in missing)} since the last run" if last else "auto: no run yet",
             "missing_fixes": [f["commit"][:7] for f in missing]}
        q["entries"].append(e)
        queued.add(st)
        log(f"queue += {st} ({kind} {alias}{' local' if e['local'] else ''}) prio {e['priority']}: {e['note']}")
    return explain


def launch(e, dry):
    os.makedirs(FD_LOGS, exist_ok=True)
    tag = time.strftime("%H%M%S")
    logf = os.path.join(FD_LOGS, f"{e['alias'][:90]}.sched-{tag}.log")
    if e["kind"] == "demo-only":
        cmd = [sys.executable, os.path.join(HERE, "first_demo.py"), "demo-only", "--tro", os.path.join(REPO, e["tro"]), "--timeout", "3600", "--note", f"scheduler: {e.get('note', '')}"]
    else:
        cmd = [sys.executable, os.path.join(HERE, "first_demo.py"), "run", "--alias", e["alias"], "--max-attempts", str(e.get("max_attempts", 8)), "--timeout", str(e.get("timeout", 7200)),
               "--note", f"scheduler: {e.get('note', '')}"]
        if e.get("scene"):
            cmd += ["--scene", e["scene"]]
        if e.get("local"):
            cmd += ["--local"]
    if dry:
        log(f"DRY launch: {' '.join(cmd)}")
        return None
    with open(logf, "w") as lf:
        p = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, cwd=REPO, start_new_session=True, env=dict(os.environ, PYTHONWARNINGS="ignore"))
    e.update(state="running", launched_at=now(), pid=p.pid, log=os.path.relpath(logf, REPO))
    log(f"launched {e['subtask']} {e['kind']} {e['alias']} pid {p.pid} ({'local' if e.get('local') else 'pool'}) -> {e['log']}")
    return p.pid


def pid_alive(pid):
    try:
        os.kill(int(pid), 0)
        return True
    except Exception:  # noqa: BLE001
        return False


def settle_running(q):
    ledger = fd.read_jsonl(fd.LEDGER)
    for e in q["entries"]:
        if e.get("state") == "running" and e.get("pid") and not pid_alive(e["pid"]):
            e["state"] = "done"
            e["finished_at"] = now()
            recs = [r for r in ledger if r["subtask"] == e["subtask"] and r["time"] >= e.get("launched_at", "")]
            if recs:
                r = recs[-1]
                d, s = r.get("demo") or {}, r.get("sampling") or {}
                e["verdict"] = {"n_written": s.get("n_written") if s else None, "rejections": (s or {}).get("rejections"), "demo": d.get("status"), "reason": d.get("reason"),
                                "failed": d.get("failed"), "meta": d.get("meta"), "commit": r.get("code_commit", "")[:7]}
            log(f"finished {e['subtask']} pid {e['pid']}: {json.dumps(e.get('verdict'))[:300]}")


def local_gpu_free():
    if os.path.exists(fd.LOCAL_LOCK):
        fh = open(fd.LOCAL_LOCK, "a")
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fh, fcntl.LOCK_UN)
        except OSError:
            fh.close()
            return False
        fh.close()
    return fd.gpu_free_mb() >= 12000


# ------------------------------------------------------------------------------------------------- band fleet
def band_manifests():
    ms = read_lines(BAND_MANIFESTS_FILE) or [os.path.join(RB, m) for m in BAND_MANIFESTS_DEFAULT]
    return [m if os.path.isabs(m) else os.path.join(REPO, m) for m in ms if os.path.exists(m if os.path.isabs(m) else os.path.join(REPO, m))]


def band_remaining(band_pods):
    """Jobs of the active band manifests that are neither done (band file) nor running (fleet_remaining.py) - the source of truth the owner named."""
    ms = band_manifests()
    if not ms or not band_pods:
        return [], ms
    out = os.path.join(RB, "jobs_remaining_sched.json")
    r = subprocess.run([sys.executable, FLEET_REMAINING, "--manifest"] + ms + ["--pods-file", BAND_FILE, "--pulled", os.path.join(REPO, "runlogs", "runpod", "rb"), "--out", out],
                       capture_output=True, text=True, cwd=REPO, timeout=900)
    if r.returncode != 0:
        log(f"fleet_remaining failed: {r.stderr.strip()[-300:]}")
        return None, ms
    return json.load(open(out)), ms


def dispatch_band(jobs, pods, dry):
    man = os.path.join(RB, f"jobs_sched_{time.strftime('%m%d-%H%M%S')}.json")
    json.dump(jobs, open(man, "w"), indent=1)
    cmd = [sys.executable, DISPATCH, "--manifest", man, "--pods-file", BAND_FILE, "--only"] + pods + ["--wave", "40"]
    if dry:
        log(f"DRY band dispatch: {' '.join(cmd)}")
        return
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=REPO, timeout=1800)
    log(f"band dispatch of {len(jobs)} job(s) to {pods}: rc {r.returncode} {r.stdout.strip()[-400:]} {r.stderr.strip()[-200:]}")


# ------------------------------------------------------------------------------------------------- role moves
def todo_activity(line):
    with open(TODO, "a") as f:
        f.write(f"- {now()} pod_scheduler: {line}\n")


def paused_note(line):
    with open(PAUSED, "a") as f:
        f.write(f"\n- {now()} pod_scheduler: {line}\n")


def move_pods(pids, src_role, dst_role, names, jobs_moved, dry):
    if not pids:
        return
    if dry:
        log(f"DRY move {pids} {src_role}->{dst_role}")
        return
    pool, band, band_role = read_lines(POOL_FILE), read_lines(BAND_FILE), read_lines(BAND_ROLE_FILE)
    for pid in pids:
        if dst_role == "pool":
            band = [p for p in band if p != pid]
            band_role = [p for p in band_role if p != pid]
            if pid not in pool:
                pool.append(pid)
        else:
            pool = [p for p in pool if p != pid]
            if pid not in band:
                band.append(pid)
            if pid not in band_role:
                band_role.append(pid)
    write_lines(POOL_FILE, pool)
    write_lines(BAND_FILE, band)
    write_lines(BAND_ROLE_FILE, band_role)
    ids = ", ".join(f"{p} ({names.get(p, '?')})" for p in pids)
    msg = (f"role change {src_role} -> {dst_role}: {ids}; queued jobs moved: {jobs_moved or 'none (queues were empty)'}; files updated: smoke/pods.txt, rb/pods.txt, pods_bands.txt"
           " (owner rule 11:05: one role per pod, nothing finished recomputed)")
    log(msg)
    todo_activity(msg)
    paused_note(msg + "; coverage chunks stay paused")


# ------------------------------------------------------------------------------------------------- one pass
def tick(dry=False):
    api = rp.Api(rp.load_env()["RUNPOD_API_KEY"])
    live = live_pods(api)
    fleet = rp.load_fleet()
    names = {pid: (fleet["pods"].get(pid) or {}).get("name", "?") for pid in fleet["pods"]}
    customs = json.load(open(fd.CUSTOMS))
    head = fd.git("rev-parse", "HEAD")
    q, state = load_queue(), load_state()
    settle_running(q)
    inflight = inflight_runs(customs)
    explain = auto_enter(q, customs, inflight, head)

    pool = [p for p in read_lines(POOL_FILE) if live is None or p in live]
    band = [p for p in read_lines(BAND_FILE) if live is None or p in live]
    pool_free = [p for p in pool if lock_free(p)]
    n_remote_inflight = sum(1 for rs in inflight.values() for r in rs if not r["local"])
    pending = max(0, n_remote_inflight - (len(pool) - len(pool_free)))  # launched runs still waiting for a lock
    slots = max(0, len(pool_free) - pending)

    # (c) pool launches, priority order, one per free pod, one per subtask
    launched = []
    for e in sorted([e for e in q["entries"] if e.get("state") == "ready"], key=lambda e: (e.get("priority", 50), e.get("added_at", ""))):
        if e["subtask"] in inflight or e["subtask"] in {x["subtask"] for x in launched}:
            continue
        if e.get("local"):
            if local_gpu_free() and not any(r["local"] for rs in inflight.values() for r in rs):
                launch(e, dry)
                launched.append(e)
                inflight.setdefault(e["subtask"], []).append({"pid": e.get("pid"), "kind": e["kind"], "alias": e["alias"], "local": True})
            continue
        if slots <= 0:
            continue
        launch(e, dry)
        launched.append(e)
        slots -= 1
    save_queue(q)

    # band fleet: survey, refill, role moves
    bstate = survey_band(api, band) if band else {}
    band_idle = [p for p, s in bstate.items() if s and s["boot"] and not s["running"] and not s["queue"]]
    band_busy = [p for p, s in bstate.items() if s and (s["running"] or s["queue"])]
    queued_ids = {jid for s in bstate.values() if s for jid in s["queue"]}
    remaining, manifests = band_remaining(band) if band else ([], band_manifests())
    unassigned = [j for j in (remaining or []) if not any(j["id"].startswith(qid) or qid.startswith(j["id"][:100]) for qid in queued_ids)] if remaining is not None else None
    for p in band_idle:  # stale current.txt markers make idle pods read as busy to other checks (owner 11:05)
        if bstate[p]["stale_current"] and not dry:
            try:
                rp.ssh_run(*bstate[p]["ep"], "rm -f /root/samp/current.txt", capture=True, timeout=30)
            except SystemExit:
                pass
    if unassigned:
        take = band_idle[: len(unassigned)]
        if take:
            dispatch_band(unassigned[: len(take)], take, dry)
            band_idle = [p for p in band_idle if p not in take]
            unassigned = unassigned[len(take):]
    # role moves (owner 11:05): band pods with nothing left -> pool; free pool pods -> bands only while band jobs wait for a pod
    ready_remote = [e for e in q["entries"] if e.get("state") == "ready" and not e.get("local")]
    if unassigned is not None and not unassigned and band_idle:
        move_pods(band_idle, "bands", "pool", names, None, dry)
        pool_free += band_idle
        band_idle = []
    elif unassigned and not ready_remote and pending == 0 and pool_free:
        n = min(len(pool_free), len(unassigned))
        movers = [p for p in pool_free if lock_free(p)][:n]
        move_pods(movers, "pool", "bands", names, None, dry)
        if movers:
            dispatch_band(unassigned[: len(movers)], movers, dry)
            unassigned = unassigned[len(movers):]
        pool_free = [p for p in pool_free if p not in movers]

    # (d) idle report
    if launched and not dry:
        time.sleep(30)  # let the launched remote_runs take their locks before the pods are read as free
    pool = read_lines(POOL_FILE) if not dry else sorted(set(pool) | set(pool_free))
    band = read_lines(BAND_FILE) if not dry else [p for p in band if p not in pool_free]
    pool_free_now = [p for p in pool_free if lock_free(p)] if not dry else pool_free
    idle_pods = {p: "pool" for p in pool_free_now}
    idle_pods.update({p: "bands" for p in band_idle})
    for p in list(state["idle_since"]):
        if p not in idle_pods:
            state["idle_since"].pop(p)
    for p in idle_pods:
        state["idle_since"].setdefault(p, now())
    save_state(state)
    ready_n = len([e for e in q["entries"] if e.get("state") == "ready"])
    holds = q.get("holds", {})
    waiting = {st: ex for st, ex in explain.items() if ex.startswith("last run")}
    lines = [f"# Idle pods ({now()})", "",
             f"pool: {len(pool)} pods, {len(pool) - len(pool_free_now)} busy (locks held), {len(pool_free_now)} free; first-demo runs in flight: "
             f"{sum(len(v) for v in inflight.values())} (pending a lock: {pending}); ready queue: {ready_n} entr{'y' if ready_n == 1 else 'ies'}"
             + (" (all local-only)" if ready_n and not ready_remote else ""),
             f"bands: {len(band)} pods, {len(band_busy)} busy, {len(band_idle)} idle; manifests {[os.path.basename(m) for m in manifests]}: "
             f"{len(remaining) if remaining is not None else '?'} remaining (not done / not running), {len(unassigned) if unassigned is not None else '?'} unassigned",
             f"local GPU: {'free' if local_gpu_free() else 'busy'} (reserved for ontop large-scale; runs in flight: {sum(1 for rs in inflight.values() for r in rs if r['local'])})", "",
             "| pod | name | role | idle since | why nothing was schedulable |", "|---|---|---|---|---|"]
    why_pool = ("ready queue empty for the pool: " + (f"{len(holds)} subtask(s) on hold for an owner decision ({', '.join(holds)}); " if holds else "")
                + (f"{len(waiting)} waiting for a fix ({', '.join(s.replace('inter_object_relation_satisfaction-', 'IORS ').replace('configuration_matching-', 'CM ') for s in waiting)}); " if waiting else "")
                + f"{sum(1 for ex in explain.values() if ex == 'valid demo on the board')} valid; {sum(1 for ex in explain.values() if ex.startswith('run in flight'))} in flight; coverage paused")
    why_band = "band manifests exhausted: nothing unassigned; pool did not need the pod this pass (moved to the pool when this line is gone)" if not unassigned else "dispatch pending"
    for p, role in idle_pods.items():
        lines.append(f"| {p} | {names.get(p, '?')} | {role} | {state['idle_since'].get(p, '?')} | {why_pool if role == 'pool' else why_band} |")
    if not idle_pods:
        lines.append("| - | - | - | - | no idle pod |")
    lines += ["", "## Subtasks not in the queue", "", "| subtask | why |", "|---|---|"]
    for st, ex in explain.items():
        lines.append(f"| {st} | {ex.replace('|', '/')} |")
    lines += ["", "## Ready queue", "", "| prio | state | subtask | kind | alias | note |", "|---|---|---|---|---|---|"]
    for e in q["entries"]:
        if e.get("state") in ("ready", "running"):
            lines.append(f"| {e.get('priority')} | {e.get('state')} | {e['subtask']} | {e['kind']}{' (local)' if e.get('local') else ''} | {e['alias']} | {(e.get('note') or '')[:80]} |")
    with open(IDLE_MD, "w") as f:
        f.write("\n".join(lines) + "\n")
    log(f"tick: pool {len(pool)} ({len(pool_free_now)} free), bands {len(band)} ({len(band_idle)} idle), inflight {sum(len(v) for v in inflight.values())}, "
        f"launched {len(launched)}, ready {ready_n}, band remaining {len(remaining) if remaining is not None else '?'} / unassigned {len(unassigned) if unassigned is not None else '?'}")
    return q


def cmd_loop(args):
    log(f"scheduler loop start (interval {args.interval}s{' DRY' if args.dry_run else ''})")
    while True:
        try:
            tick(args.dry_run)
        except Exception as e:  # noqa: BLE001
            import traceback

            log(f"tick failed: {e}\n{traceback.format_exc()[-1200:]}")
        time.sleep(args.interval)


def cmd_add(args):
    q = load_queue()
    customs = json.load(open(fd.CUSTOMS))
    st = fd.subtask_of(args.alias, customs)
    e = {"id": f"{st}:{time.strftime('%m%d-%H%M%S')}", "priority": args.priority, "subtask": st, "alias": args.alias, "kind": args.kind, "tro": args.tro,
         "scene": args.scene or DEFAULT_SCENE.get(args.alias), "local": args.local or st in LOCAL_ONLY, "state": "ready", "added_by": args.by, "added_at": now(), "note": args.note,
         "max_attempts": args.max_attempts, "timeout": args.timeout}
    q["entries"].append(e)
    save_queue(q)
    log(f"queue += {st} ({args.kind} {args.alias}) prio {args.priority} by {args.by}: {args.note}")


def cmd_hold(args):
    q = load_queue()
    q.setdefault("holds", {})[args.subtask] = {"reason": args.reason, "since": now()}
    for e in q["entries"]:
        if e["subtask"] == args.subtask and e.get("state") == "ready":
            e["state"] = "held"
    save_queue(q)
    log(f"hold {args.subtask}: {args.reason}")


def cmd_release(args):
    q = load_queue()
    q.get("holds", {}).pop(args.subtask, None)
    save_queue(q)
    log(f"release {args.subtask}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("loop")
    s.add_argument("--interval", type=int, default=300)
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=cmd_loop)
    s = sub.add_parser("tick")
    s.add_argument("--dry-run", action="store_true")
    s.set_defaults(fn=lambda a: tick(a.dry_run))
    s = sub.add_parser("status")
    s.set_defaults(fn=lambda a: print(open(IDLE_MD).read() if os.path.exists(IDLE_MD) else "no report yet"))
    s = sub.add_parser("add")
    s.add_argument("--alias", required=True)
    s.add_argument("--kind", choices=["run", "demo-only"], default="run")
    s.add_argument("--tro", default=None)
    s.add_argument("--scene", default=None)
    s.add_argument("--priority", type=int, default=40)
    s.add_argument("--local", action="store_true")
    s.add_argument("--max-attempts", type=int, default=8)
    s.add_argument("--timeout", type=int, default=7200)
    s.add_argument("--note", default="")
    s.add_argument("--by", default="agent")
    s.set_defaults(fn=cmd_add)
    s = sub.add_parser("hold")
    s.add_argument("--subtask", required=True)
    s.add_argument("--reason", required=True)
    s.set_defaults(fn=cmd_hold)
    s = sub.add_parser("release")
    s.add_argument("--subtask", required=True)
    s.set_defaults(fn=cmd_release)
    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
