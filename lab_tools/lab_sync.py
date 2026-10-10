#!/usr/bin/env python3
"""Lab-side reporting for the ISG goal-conditioning study (coordination repo ``kmy17518/isg-goal-coordination``, prefix ``lab/``).

    lab_sync.py note "text"            append one timestamped entry to lab/notes.md (append-only)
    lab_sync.py status                 rewrite lab/status.json from lab_tools/state.json + the local Tier B results
    lab_sync.py pull                   download m1/ and docs/ of the coordination repo; print decisions added since the last pull
    lab_sync.py upload                 one commit: hf upload <lab dir> lab (--commit-message "lab sync")
    lab_sync.py sync                   pull + status + upload (what the 15-minute loop runs)
    lab_sync.py set KEY VALUE          set a state key (JSON value), e.g. set blocked null, set current '{"run": ...}'
    lab_sync.py add-decision "text"    append a needs_decision entry (deduplicated)

Only ``lab/`` is written in the repo. Tokens come from the worktree's ``.env`` (HF_TOKEN) and are never printed.
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAB = os.path.join(WT, "lab")
STATE = os.path.join(WT, "lab_tools", "state.json")
COORD = os.path.join(WT, "datasets", "isg-goal-coordination")
REPO = "kmy17518/isg-goal-coordination"
HF = os.path.join(WT, "act-isg", ".venv", "bin", "hf")
TIERB_LOCAL = os.path.join(WT, "lab", "tierB")


def utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def load_env() -> None:
    os.environ.pop("HF_HUB_OFFLINE", None)  # the simulator jobs export it; the sync must reach the Hub
    p = os.path.join(WT, ".env")
    if os.path.exists(p):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip())


def read_state() -> dict:
    if os.path.exists(STATE):
        with open(STATE) as f:
            return json.load(f)
    return {"current": None, "needs_decision": [], "blocked": None, "phase": "setup"}


def write_state(st: dict) -> None:
    tmp = STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATE)


def note(text: str) -> None:
    os.makedirs(LAB, exist_ok=True)
    p = os.path.join(LAB, "notes.md")
    new = not os.path.exists(p)
    with open(p, "a") as f:
        if new:
            f.write("# Lab notes (append-only)\n\nProblems, deviations and questions of the lab (Tier B), oldest first. UTC timestamps.\n\n")
        f.write(f"## {utc()}\n{text.rstrip()}\n\n")


def gpu() -> dict:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip().split("\n")[0]
        used, total, util = [int(float(x)) for x in out.split(",")]
        return {"memory_used_mib": used, "memory_total_mib": total, "utilization_pct": util}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)}


def episode_files(run: str, step: int):
    return sorted(glob.glob(os.path.join(TIERB_LOCAL, run, f"step_{step:08d}", "*", "episode_*.json")))


def progress(st: dict) -> dict:
    cur = st.get("current") or {}
    out = {"run": cur.get("run"), "step": cur.get("step"), "split": cur.get("split"), "replay_split": cur.get("replay_split"), "horizon": cur.get("horizon"),
           "episodes_done": 0, "episodes_total": cur.get("episodes_total") or 0, "episodes_per_hour": None, "started_at": cur.get("started_at"),
           "lab_dir": cur.get("lab_dir")}
    files = []
    if cur.get("lab_dir"):
        files = sorted(glob.glob(os.path.join(WT, cur["lab_dir"], "episode_*.json")))
        if cur.get("horizon") is not None:
            kept = []
            for p in files:
                try:
                    with open(p) as f:
                        if json.load(f).get("horizon") == cur["horizon"]:
                            kept.append(p)
                except Exception:  # noqa: BLE001
                    pass
            files = kept
    elif cur.get("run") and cur.get("step") is not None:
        files = episode_files(cur["run"], int(cur["step"]))
    if files:
        out["episodes_done"] = len(files)
        times = []
        for p in files:
            try:
                with open(p) as f:
                    d = json.load(f)
                if d.get("finished_at") and (not cur.get("started_at") or d["finished_at"] >= cur["started_at"]):
                    times.append(d["finished_at"])  # the rate of this entry's own episodes (older records of the entry do not count)
            except Exception:  # noqa: BLE001
                pass
        if len(times) >= 2:
            t = sorted(times)
            span_h = max(1e-6, (time.mktime(time.strptime(t[-1], "%Y-%m-%dT%H:%M:%SZ")) - time.mktime(time.strptime(t[0], "%Y-%m-%dT%H:%M:%SZ"))) / 3600.0)
            out["episodes_per_hour"] = round((len(t) - 1) / span_h, 2)
    return out


def status() -> dict:
    st = read_state()
    os.makedirs(LAB, exist_ok=True)
    doc = {
        "format": "isg-lab-status/v1",
        "updated_at": utc(),
        "host": os.uname().nodename,
        "phase": st.get("phase"),
        "current": progress(st),
        "episodes_per_hour": progress(st)["episodes_per_hour"] if st.get("current") else st.get("episodes_per_hour_measured"),
        "gpu": gpu(),
        "instances_verified": st.get("instances_verified"),
        "instances_train_verified": st.get("instances_train_verified"),
        "queue_seen_updated_at": st.get("queue_seen_updated_at"),
        "queue_paused": os.path.exists(os.path.join(os.path.dirname(os.path.abspath(__file__)), "PAUSE")),  # the loop's flag file is the truth
        "queue_horizon_steps": st.get("queue_horizon_steps"),
        "completed": st.get("completed", []),
        "needs_decision": list(st.get("needs_decision") or []),
        "blocked": st.get("blocked"),
        "extra": st.get("extra", {}),
    }
    with open(os.path.join(LAB, "status.json"), "w") as f:
        json.dump(doc, f, indent=1)
    return doc


def pull() -> None:
    load_env()
    before = ""
    dec = os.path.join(COORD, "m1", "decisions.md")
    if os.path.exists(dec):
        with open(dec) as f:
            before = f.read()
    subprocess.run([HF, "download", REPO, "--repo-type", "dataset", "--local-dir", COORD, "--include", "m1/*", "docs/*"],
                   check=True, capture_output=True, text=True, timeout=600)
    with open(dec) as f:
        after = f.read()
    if after != before:
        new = after[len(before):] if after.startswith(before) else after
        print("DECISIONS CHANGED:\n" + new)
    else:
        print("decisions.md unchanged")


def upload() -> None:
    load_env()
    r = subprocess.run([HF, "upload", REPO, LAB, "lab", "--repo-type", "dataset", "--commit-message", "lab sync"],
                       capture_output=True, text=True, timeout=1800)
    if r.returncode != 0:
        print("UPLOAD FAILED:\n" + r.stderr[-2000:])
        sys.exit(r.returncode)
    print("uploaded: " + (r.stdout.strip().splitlines() or ["ok"])[-1][:200])


def main(argv) -> int:
    cmd = argv[0] if argv else "status"
    if cmd == "note":
        note(argv[1])
    elif cmd == "status":
        print(json.dumps(status(), indent=1))
    elif cmd == "pull":
        pull()
    elif cmd == "upload":
        status()
        upload()
    elif cmd == "sync":
        try:
            pull()
        except Exception as e:  # noqa: BLE001
            print(f"pull failed: {e!r}")
        status()
        upload()
    elif cmd == "set":
        st = read_state()
        st[argv[1]] = json.loads(argv[2])
        write_state(st)
    elif cmd == "add-decision":
        st = read_state()
        nd = st.setdefault("needs_decision", [])
        if argv[1] not in nd:
            nd.append(argv[1])
        write_state(st)
    else:
        print(__doc__)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
