#!/usr/bin/env python3
"""Tier B loop of the lab, queue format ``isg-tierB-queue/v2`` (relocalization phase, from 2026-10-07).

Every ``--poll-s`` (10 min) the loop pulls ``m1/`` of the coordination repo and reads ``m1/tierB-queue.json``. While
``paused`` is true nothing runs. Otherwise the entries are worked in order; an entry is skipped once a complete
``summary.json`` exists for its (run, step, split) at the queue's ``horizon_steps``. Entry kinds (``split``):

* ``replay`` - positive control: the recorded actions of the listed episodes of ``replay_split`` (``eval50`` = test set
  ``datasets/isg-init-eval`` + ``lab/instances.json``; ``train`` = ``datasets/isg-init`` + ``lab/instances-train.json``) are
  replayed open-loop (``lab_tools/replay_episodes.py``) -> ``lab/diagnostics/replay/<replay_split>/episode_XXXXXX.json``;
* ``train`` - the checkpoint (downloaded, sha256 checked; the ACT server restarted on it) on training episodes ->
  ``lab/tierB-train/<run>/step_XXXXXXXX/camera_relocalization-standard/episode_XXXXXX.json``;
* ``eval50`` - the checkpoint on the test set -> ``lab/tierB/<run>/step_XXXXXXXX/camera_relocalization-standard/episode_XXXXXX.json``.

``episodes`` null = every relocalization episode of the split with a verified instance. Every episode records the
queue's horizon; existing records at another horizon are never overwritten (the entry is reported in ``needs_decision``
and skipped). One ``summary.json`` per (run, step, split) is written next to the episode files when every listed episode
has an error-free record (errored episodes are re-run once). Idempotent: finished episodes are skipped, crashed
evaluator processes are restarted. The 15-minute status loop uploads ``lab/`` as episodes finish.

    python lab_tools/tierb_loop.py [--parallel 2] [--episodes-per-process 10] [--once]
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

WT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, WT)
from lab_tools import lab_sync as LS  # noqa: E402

CONDA_PY = os.path.expanduser("~/miniconda3/envs/ispatialgym-v2.0-vectorized/bin/python")
ACT = os.path.join(WT, "act-isg")
ACT_PY = os.path.join(ACT, ".venv", "bin", "python")
HF = os.path.join(ACT, ".venv", "bin", "hf")
QUEUE = os.path.join(WT, "datasets", "isg-goal-coordination", "m1", "tierB-queue.json")
#: the lab's own entries (same format; worked before machine 1's queue): the execution-settings test
LOCAL_QUEUE = os.path.join(WT, "lab_tools", "local_queue.json")
#: while this file exists the loop launches nothing (lab maintenance that needs the GPU, e.g. first-frame renders); remove it to resume
PAUSE_FLAG = os.path.join(WT, "lab_tools", "PAUSE")
SERVER_DIR = os.path.join(WT, "runlogs", "tierB", "server")
TASK = "camera_relocalization-standard"
LOG = os.path.join(WT, "lab_tools", "tierb_loop.log")
#: GPU memory an Isaac Sim evaluator needs on top of what is in use (MiB); the loop waits for it before every launch
EVALUATOR_GPU_MIB = 10000
SPLITS = {
    "eval50": {"dataset": os.path.join(WT, "datasets", "isg-init-eval"), "instances": os.path.join(WT, "lab", "instances.json")},
    "train": {"dataset": os.path.join(WT, "datasets", "isg-init"), "instances": os.path.join(WT, "lab", "instances-train.json")},
}


def log(msg: str) -> None:
    line = f"{LS.utc()} {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def gpu_free_mib() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=20).stdout
        used, total = [int(float(x)) for x in out.strip().split("\n")[0].split(",")]
        return total - used
    except Exception:  # noqa: BLE001
        return 0


def wait_for_gpu(need_mib: int, what: str, max_wait_s: int = 3 * 3600) -> bool:
    t0 = time.time()
    warned = False
    while gpu_free_mib() < need_mib:
        if not warned:
            log(f"waiting for {need_mib} MiB of free GPU memory before {what} (free {gpu_free_mib()} MiB; another process holds the card)")
            warned = True
        if time.time() - t0 > max_wait_s:
            log(f"gave up waiting for GPU memory for {what}")
            return False
        time.sleep(60)
    return True


# ---------------------------------------------------------------------------------------------------------------------
# checkpoint + server
# ---------------------------------------------------------------------------------------------------------------------

def ensure_checkpoint(entry: dict) -> str:
    LS.load_env()
    local = os.path.join(ACT, "ckpts", entry["run"], entry["file"])
    if not (os.path.exists(local) and sha256_of(local) == entry["sha256"]):
        log(f"downloading {entry['repo']} {entry['file']}")
        subprocess.run([HF, "download", entry["repo"], entry["file"], "--local-dir", os.path.join(ACT, "ckpts", entry["run"])],
                       check=True, capture_output=True, text=True, timeout=3600)
        actual = sha256_of(local)
        if actual != entry["sha256"]:
            raise RuntimeError(f"sha256 mismatch for {entry['run']}: {actual} != {entry['sha256']}")
    return local


def git_pull_act() -> None:
    """``git pull`` of the ACT checkout (branch isg-wave2-dev) before a new checkpoint is served (read-only use of the remote)."""
    try:
        r = subprocess.run(["git", "-C", ACT, "pull", "--ff-only"], capture_output=True, text=True, timeout=300)
        head = subprocess.run(["git", "-C", ACT, "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=30).stdout.strip()
        log(f"act-isg git pull: rc={r.returncode} HEAD {head} {(r.stdout or r.stderr).strip().splitlines()[-1][:120] if (r.stdout or r.stderr).strip() else ''}")
    except Exception as e:  # noqa: BLE001
        log(f"act-isg git pull failed: {e!r}")


def server_alive(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=5) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def current_server() -> dict | None:
    p = os.path.join(SERVER_DIR, "current.json")
    if os.path.exists(p):
        try:
            return json.load(open(p))
        except Exception:  # noqa: BLE001
            return None
    return None


def stop_own_server() -> None:
    """Stop the ACT server this loop started (recorded pid; checked against /proc to never signal a reused pid)."""
    cur = current_server()
    pid = (cur or {}).get("pid")
    legacy = os.path.join(SERVER_DIR, "current.pid")
    if pid is None and os.path.exists(legacy):
        try:
            pid = int(open(legacy).read().split()[0])
        except Exception:  # noqa: BLE001
            pid = None
    if pid is None:
        return
    try:
        cmd = open(f"/proc/{pid}/cmdline").read().replace("\0", " ")
    except Exception:  # noqa: BLE001
        return
    if "scripts/b1k/serve_b1k.py" in cmd and (cur is None or str(cur.get("port", "")) in cmd):
        log(f"stopping the lab's ACT server pid {pid}")
        try:
            os.kill(int(pid), signal.SIGTERM)
        except ProcessLookupError:
            pass
        for _ in range(30):
            if not os.path.exists(f"/proc/{pid}"):
                break
            time.sleep(1)


def start_server(ckpt: str, run: str, step: int, port: int, extra_args: list | None = None) -> dict:
    """(Re)start the ACT server on ``ckpt`` with the entry's extra serve_b1k.py arguments (none = the default action horizon);
    returns its args record. A server already serving this checkpoint with the same arguments on the port is kept."""
    os.makedirs(SERVER_DIR, exist_ok=True)
    extra = list(extra_args or [])
    cur = current_server()
    if cur and cur.get("ckpt") == ckpt and cur.get("port") == port and (cur.get("extra_args") or []) == extra and server_alive(port) and os.path.exists(f"/proc/{cur.get('pid', -1)}"):
        log(f"server already up for {run} step {step} {extra or ''} (pid {cur['pid']})")
        return cur
    stop_own_server()
    for _ in range(30):
        if not server_alive(port):
            break
        time.sleep(1)
    if server_alive(port):
        raise RuntimeError(f"port {port} is served by a process this loop did not start; refusing to use it")
    args = ["scripts/b1k/serve_b1k.py", "--model-path", os.path.relpath(ckpt, ACT), "--port", str(port), "--device", "cuda", *extra]
    logf = open(os.path.join(SERVER_DIR, f"{run}.log"), "a")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="0")
    proc = subprocess.Popen([ACT_PY, *args], cwd=ACT, stdout=logf, stderr=subprocess.STDOUT, env=env)
    rec = {"pid": proc.pid, "port": port, "cmd": "CUDA_VISIBLE_DEVICES=0 .venv/bin/python " + " ".join(args), "ckpt": ckpt, "run": run, "step": step,
           "extra_args": extra, "started_at": LS.utc()}
    with open(os.path.join(SERVER_DIR, "current.json"), "w") as f:
        json.dump(rec, f, indent=1)
    with open(os.path.join(SERVER_DIR, "current.pid"), "w") as f:
        f.write(f"{proc.pid} {run}\n")
    for _ in range(120):
        if server_alive(port) and proc.poll() is None:
            break
        if proc.poll() is not None:
            raise RuntimeError(f"server exited with {proc.returncode}; see {logf.name}")
        time.sleep(2)
    else:
        raise RuntimeError("server did not come up in 240 s")
    time.sleep(2)
    if proc.poll() is not None:
        raise RuntimeError(f"server died right after starting (port {port} taken?); see {logf.name}")
    log(f"server up for {run} step {step} (pid {proc.pid}, port {port})")
    return rec


# ---------------------------------------------------------------------------------------------------------------------
# entries: paths, episode lists, completeness
# ---------------------------------------------------------------------------------------------------------------------

def entry_split(entry: dict) -> str:
    return entry.get("split") or "eval50"


def data_split(entry: dict) -> str:
    """The dataset split the episodes come from (replay entries name it in ``replay_split``)."""
    if entry_split(entry) == "replay":
        return entry.get("replay_split") or "eval50"
    return entry_split(entry)


def entry_tag(entry: dict) -> str | None:
    """Execution-settings tag (``tag`` + ``server_args`` in the entry): results go under lab/diagnostics/exec/<tag>/..."""
    t = entry.get("tag")
    return str(t) if t else None


def server_extra_args(entry: dict) -> list:
    return [str(a) for a in (entry.get("server_args") or [])]


def expected_action_horizon(entry: dict, default: int) -> int:
    """What the server will report as action_horizon for the entry's server args (b1k_server.Session)."""
    args = server_extra_args(entry)
    if "--action-horizon" in args:
        return int(args[args.index("--action-horizon") + 1])
    if "--temporal-agg" in args:
        return 1
    return default


def lab_dir(entry: dict) -> str:
    sp = entry_split(entry)
    if sp == "replay":
        return os.path.join(WT, "lab", "diagnostics", "replay", data_split(entry))
    if entry_tag(entry):
        return os.path.join(WT, "lab", "diagnostics", "exec", entry_tag(entry), entry["run"], f"step_{int(entry['step']):08d}", TASK)
    root = "tierB" if sp == "eval50" else "tierB-train"
    return os.path.join(WT, "lab", root, entry["run"], f"step_{int(entry['step']):08d}", TASK)


def raw_dir(entry: dict) -> str:
    sp = entry_split(entry)
    if sp == "replay":
        return os.path.join(WT, "runlogs", "tierB", "replay", data_split(entry))
    if entry_tag(entry):
        return os.path.join(WT, "runlogs", "tierB-exec", entry_tag(entry), entry["run"], f"step_{int(entry['step']):08d}")
    root = "tierB" if sp == "eval50" else "tierB-train"
    return os.path.join(WT, "runlogs", root, entry["run"], f"step_{int(entry['step']):08d}")


def entry_key(entry: dict) -> str:
    sp = entry_split(entry)
    if sp == "replay":
        return f"replay/{data_split(entry)}"
    tag = f"exec:{entry_tag(entry)}/" if entry_tag(entry) else ""
    return f"{tag}{sp}/{entry['run']}/step_{int(entry['step']):08d}"


def load_instances(split: str) -> dict | None:
    p = SPLITS[split]["instances"]
    if not os.path.exists(p):
        return None
    doc = json.load(open(p))
    return doc.get("episodes", doc)


def wanted_episodes(entry: dict, instances: dict) -> list:
    """The entry's episode indices (null = every verified relocalization episode of the split)."""
    verified = sorted(int(k) for k, v in instances.items() if v.get("task") == TASK and v.get("instance") and v.get("verified"))
    if entry.get("episodes") is None:
        return verified
    return sorted(int(e) for e in entry["episodes"])


def read_record(path: str) -> dict:
    try:
        return json.load(open(path))
    except Exception:  # noqa: BLE001
        return {"error": "unreadable"}


def episode_records(entry: dict) -> dict:
    out = {}
    for p in glob.glob(os.path.join(lab_dir(entry), "episode_*.json")):
        try:
            out[int(os.path.basename(p)[len("episode_"):-len(".json")])] = read_record(p)
        except ValueError:
            pass
    return out


def entry_state(entry: dict, horizon: int, instances: dict | None) -> dict:
    """What is left for the entry: missing / errored / wrong-horizon episodes, unmapped episodes, summary status."""
    st = {"key": entry_key(entry), "missing": [], "errored": [], "other_horizon": [], "unmapped": [], "done": [], "summary_ok": False}
    if instances is None:
        st["no_instances"] = SPLITS[data_split(entry)]["instances"]
        return st
    want = wanted_episodes(entry, instances)
    recs = episode_records(entry)
    for e in want:
        v = instances.get(str(e))
        if v is None or v.get("task") != TASK or not v.get("instance") or not v.get("verified"):
            st["unmapped"].append(e)
            continue
        r = recs.get(e)
        if r is None:
            st["missing"].append(e)
        elif r.get("horizon") != horizon:
            st["other_horizon"].append(e)
        elif r.get("error"):
            st["errored"].append(e)
        else:
            st["done"].append(e)
    st["want"] = want
    sp = os.path.join(lab_dir(entry), "summary.json")
    if os.path.exists(sp):
        s = read_record(sp)
        st["summary_ok"] = bool(s.get("complete")) and s.get("horizon") == horizon and sorted(s.get("episodes") or []) == want
    return st


def complete(entry: dict, horizon: int, instances: dict | None) -> bool:
    st = entry_state(entry, horizon, instances)
    return st["summary_ok"] and not st["missing"] and not st["errored"] and not st["other_horizon"]


# ---------------------------------------------------------------------------------------------------------------------
# running
# ---------------------------------------------------------------------------------------------------------------------

def spawn_eval(entry: dict, horizon: int, server: dict, args, episodes: list, overwrite: bool = False):
    sp = data_split(entry)
    cmd = [CONDA_PY, os.path.join(WT, "lab_tools", "eval_episodes.py"), "--run", entry["run"], "--step", str(entry["step"]),
           "--ckpt-repo", entry["repo"], "--ckpt-file", entry["file"], "--ckpt-sha256", entry["sha256"], "--ckpt-path", server["ckpt"],
           "--server", f"ws://127.0.0.1:{server['port']}", "--server-args", server["cmd"], "--action-horizon", str(expected_action_horizon(entry, args.action_horizon)),
           "--task", TASK, "--split", sp, "--dataset", SPLITS[sp]["dataset"], "--instances", SPLITS[sp]["instances"],
           "--lab-out", os.path.dirname(lab_dir(entry)), "--raw-out", raw_dir(entry), "--horizon", str(horizon), "--video-size", str(args.video_size),
           "--episodes", *map(str, episodes)]
    if entry_tag(entry):
        cmd += ["--tag", entry_tag(entry)]
    if overwrite:
        cmd.append("--overwrite")
    if getattr(args, "no_planner", False):
        cmd.append("--no-planner")
    if getattr(args, "cleanup", False):
        cmd.append("--cleanup")
    os.makedirs(os.path.join(raw_dir(entry), "logs"), exist_ok=True)
    logf = open(os.path.join(raw_dir(entry), "logs", f"{TASK}.{time.strftime('%Y%m%d-%H%M%S')}.{os.getpid()}.log"), "w")
    env = dict(os.environ, OMNIGIBSON_HEADLESS="1", OMNI_KIT_ACCEPT_EULA="YES", HF_HUB_OFFLINE="1")
    return subprocess.Popen(cmd, cwd=WT, stdout=logf, stderr=subprocess.STDOUT, env=env)


def spawn_replay(entry: dict, horizon: int, args, episodes: list, overwrite: bool = False):
    sp = data_split(entry)
    cmd = [CONDA_PY, os.path.join(WT, "lab_tools", "replay_episodes.py"), "--split", sp, "--dataset", SPLITS[sp]["dataset"], "--instances", SPLITS[sp]["instances"],
           "--episodes", *map(str, episodes), "--horizon", str(horizon), "--lab-out", lab_dir(entry), "--raw-out", raw_dir(entry), "--video-size", str(args.video_size)]
    if overwrite:
        cmd.append("--overwrite")
    if getattr(args, "no_planner", False):
        cmd.append("--no-planner")
    os.makedirs(os.path.join(raw_dir(entry), "logs"), exist_ok=True)
    logf = open(os.path.join(raw_dir(entry), "logs", f"replay.{time.strftime('%Y%m%d-%H%M%S')}.{os.getpid()}.log"), "w")
    env = dict(os.environ, OMNIGIBSON_HEADLESS="1", OMNI_KIT_ACCEPT_EULA="YES", HF_HUB_OFFLINE="1")
    return subprocess.Popen(cmd, cwd=WT, stdout=logf, stderr=subprocess.STDOUT, env=env)


# ---------------------------------------------------------------------------------------------------------------------
# persistent evaluator workers (evaluator fix 2.2): one Isaac Sim process per slot, the loaded world kept across entries
# ---------------------------------------------------------------------------------------------------------------------

WORKERS_DIR = os.path.join(WT, "runlogs", "tierB", "workers")


def worker_dir(i: int) -> str:
    return os.path.join(WORKERS_DIR, str(i))


def worker_pid(i: int) -> int | None:
    p = os.path.join(worker_dir(i), "worker.pid")
    if not os.path.exists(p):
        return None
    try:
        pid = int(open(p).read().split()[0])
    except Exception:  # noqa: BLE001
        return None
    try:
        cmd = open(f"/proc/{pid}/cmdline").read().replace("\0", " ")
    except Exception:  # noqa: BLE001
        return None
    return pid if "eval_worker.py" in cmd and f"--worker-id {i}" in cmd else None


def ensure_worker(i: int, args) -> int | None:
    """The worker of slot ``i`` alive (spawned when missing; GPU memory for a fresh world is awaited); returns its pid."""
    pid = worker_pid(i)
    if pid is not None:
        return pid
    d = worker_dir(i)
    os.makedirs(d, exist_ok=True)
    stop = os.path.join(d, "STOP")
    if os.path.exists(stop):
        os.remove(stop)
    need = EVALUATOR_GPU_MIB if getattr(args, "num_envs", 1) <= 1 else 13000 + 1200 * int(args.num_envs)  # a 4-environment world runs at ~15.3 GB
    if not wait_for_gpu(need, f"worker {i} (fresh world)"):
        return None
    logf = open(os.path.join(d, f"stdout.{time.strftime('%Y%m%d-%H%M%S')}.log"), "w")
    env = dict(os.environ, OMNIGIBSON_HEADLESS="1", OMNI_KIT_ACCEPT_EULA="YES", HF_HUB_OFFLINE="1")
    cmd = [CONDA_PY, os.path.join(WT, "lab_tools", "eval_worker.py"), "--worker-id", str(i), "--jobs-dir", d, "--max-gpu-mib", str(args.worker_max_gpu_mib)]
    proc = subprocess.Popen(cmd, cwd=WT, stdout=logf, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    log(f"worker {i}: spawned pid {proc.pid} (a fresh scene load follows)")
    for _ in range(60):
        if worker_pid(i) == proc.pid:
            break
        if proc.poll() is not None:
            log(f"worker {i}: exited right away rc={proc.returncode}; see {logf.name}")
            return None
        time.sleep(1)
    time.sleep(60)  # let the Isaac Sim start-up allocate before the next worker is considered
    return proc.pid


def stop_workers(reason: str) -> None:
    for d in glob.glob(os.path.join(WORKERS_DIR, "*")):
        if os.path.isdir(d):
            open(os.path.join(d, "STOP"), "w").write(reason)


def next_job_index(d: str) -> int:
    n = 0
    for p in glob.glob(os.path.join(d, "job_*.json")):
        b = os.path.basename(p)
        if b.endswith(".done.json"):
            continue
        try:
            n = max(n, int(b[len("job_"):-len(".json")]) + 1)
        except ValueError:
            pass
    return n


def job_for(entry: dict, horizon: int, server: dict | None, args, episodes: list, overwrite: bool) -> dict:
    sp = data_split(entry)
    job = {"kind": "replay" if entry_split(entry) == "replay" else "eval", "run": entry["run"], "step": entry.get("step"), "episodes": list(episodes), "horizon": horizon,
           "task": TASK, "split": sp, "dataset": SPLITS[sp]["dataset"], "instances": SPLITS[sp]["instances"], "video_size": args.video_size, "overwrite": bool(overwrite),
           "tag": entry_tag(entry), "no_planner": bool(args.no_planner), "cleanup": bool(args.cleanup), "created_at": LS.utc(), "entry_key": entry_key(entry)}
    if job["kind"] == "eval":
        job.update({"ckpt_repo": entry["repo"], "ckpt_file": entry["file"], "ckpt_sha256": entry["sha256"], "ckpt_path": server["ckpt"], "server": f"ws://127.0.0.1:{server['port']}",
                    "server_args": server["cmd"], "action_horizon": expected_action_horizon(entry, args.action_horizon), "lab_out": os.path.dirname(lab_dir(entry)), "raw_out": raw_dir(entry)})
    else:
        job.update({"lab_out": lab_dir(entry), "raw_out": raw_dir(entry)})
    if getattr(args, "num_envs", 1) > 1:
        # the vectorized evaluator (lab_tools/vec_eval.py on the worker's cloned world): N episodes per batch; replay entries keep the
        # replay mode (one tool for both), eval entries the policy mode; lab_out is the task's parent for both tools
        job.update({"vectorized": True, "num_envs": int(args.num_envs), "mode": "replay" if job["kind"] == "replay" else "policy", "warm_renders": int(args.warm_renders), "settle_steps": 50})
        if job["kind"] == "replay":
            job.update({"lab_out": lab_dir(entry), "raw_out": raw_dir(entry), "no_task_subdir": True})  # the replay layout has no task subdirectory
    return job


def run_workers(entry: dict, horizon: int, server: dict | None, args, pending_fn, overwrite: bool = False, max_rounds: int = 6) -> None:
    """Dispatch the pending episodes to the persistent workers (round-robin, at most ``--episodes-per-job`` per job) and wait for the
    jobs; a worker that dies (or restarts itself above its memory cap) is respawned and its episodes re-dispatched."""
    rounds = 0
    while True:
        pending = pending_fn()
        if not pending:
            return
        if os.path.exists(PAUSE_FLAG):
            log(f"{entry_key(entry)}: paused ({os.path.relpath(PAUSE_FLAG, WT)} exists); no new job until it is removed")
            return
        if rounds >= max_rounds:
            log(f"{entry_key(entry)}: giving up after {rounds} dispatch rounds; still pending {pending}")
            return
        rounds += 1
        slots = [i for i in range(args.parallel) if ensure_worker(i, args) is not None]
        if not slots:
            log(f"{entry_key(entry)}: no worker could be started")
            return
        chunks = {i: [] for i in slots}
        for k, e in enumerate(pending):
            chunks[slots[k % len(slots)]].append(e)
        waiting = {}
        for i, eps in chunks.items():
            for start in range(0, len(eps), args.episodes_per_job):
                sub = eps[start:start + args.episodes_per_job]
                d = worker_dir(i)
                n = next_job_index(d)
                job = job_for(entry, horizon, server, args, sub, overwrite)
                path = os.path.join(d, f"job_{n:04d}.json")
                with open(path + ".tmp", "w") as f:
                    json.dump(job, f, indent=1)
                os.replace(path + ".tmp", path)
                waiting.setdefault(i, []).append(path)
                log(f"{entry_key(entry)}: job {os.path.basename(path)} -> worker {i} (pid {worker_pid(i)}) episodes {sub}")
        # wait for every job of this round: done file, or a dead worker (its remaining jobs are dropped and re-dispatched next round)
        while waiting:
            time.sleep(15)
            for i in list(waiting):
                pid = worker_pid(i)
                if pid is None:
                    for p in waiting[i]:
                        if not os.path.exists(p[:-5] + ".done.json"):
                            log(f"{entry_key(entry)}: worker {i} is gone; dropping {os.path.basename(p)} (re-dispatched)")
                            try:
                                os.remove(p)
                            except OSError:
                                pass
                    del waiting[i]
                    continue
                waiting[i] = [p for p in waiting[i] if not os.path.exists(p[:-5] + ".done.json")]
                if not waiting[i]:
                    del waiting[i]
        for i in slots:
            d = worker_dir(i)
            dones = sorted(glob.glob(os.path.join(d, "job_*.done.json")))
            if dones:
                try:
                    dn = json.load(open(dones[-1]))
                    m = dn.get("memory") or {}
                    log(f"{entry_key(entry)}: worker {i} after job: gpu {m.get('gpu_mib')} MiB rss {m.get('rss_mib')} MiB, episodes this process {dn.get('episodes_this_process')}, scene loads {dn.get('scene_loads_this_process')}")
                except Exception:  # noqa: BLE001
                    pass


def run_pool(entry: dict, horizon: int, server: dict | None, args, pending_fn, overwrite: bool = False, max_rounds: int = 6) -> None:
    """Run the pending episodes in up to ``--parallel`` processes of at most ``--episodes-per-process`` episodes each (round-robin
    split), restarting processes while episodes are left (crashes, the per-process limit). With ``--use-workers`` the persistent
    workers run them instead (:func:`run_workers`)."""
    if getattr(args, "use_workers", False):
        return run_workers(entry, horizon, server, args, pending_fn, overwrite, max_rounds)
    rounds = 0
    procs: list = []
    while True:
        pending = pending_fn()
        running_eps = {e for _, eps in procs for e in eps}
        todo = [e for e in pending if e not in running_eps]
        while todo and len(procs) < args.parallel:
            if os.path.exists(PAUSE_FLAG):
                log(f"{entry_key(entry)}: paused ({os.path.relpath(PAUSE_FLAG, WT)} exists); no new process until it is removed")
                break
            if not wait_for_gpu(EVALUATOR_GPU_MIB, f"{entry_key(entry)} evaluator"):
                break
            chunk = todo[: args.episodes_per_process]
            todo = todo[len(chunk):]
            p = spawn_replay(entry, horizon, args, chunk, overwrite) if entry_split(entry) == "replay" else spawn_eval(entry, horizon, server, args, chunk, overwrite)
            log(f"{entry_key(entry)}: started process pid {p.pid} for episodes {chunk}")
            procs.append((p, chunk))
            time.sleep(90)  # Isaac Sim start-up before the next launch (memory settles)
        if not procs:
            if todo and rounds >= max_rounds:
                log(f"{entry_key(entry)}: giving up after {rounds} rounds; still pending {todo}")
            return
        time.sleep(30)
        for p, eps in list(procs):
            if p.poll() is not None:
                log(f"{entry_key(entry)}: process pid {p.pid} exited rc={p.returncode} (episodes {eps})")
                procs.remove((p, eps))
                rounds += 1
                if rounds > max_rounds * max(1, args.parallel):
                    log(f"{entry_key(entry)}: too many process restarts; leaving the entry for the next poll")
                    for q, _ in procs:
                        q.wait()
                    return


def write_summary(entry: dict, horizon: int, instances: dict, server: dict | None, args) -> dict:
    import statistics

    st = entry_state(entry, horizon, instances)
    recs = episode_records(entry)
    want = st["want"]
    ok = [recs[e] for e in want if e in recs and not recs[e].get("error") and recs[e].get("horizon") == horizon and isinstance(recs[e].get("success"), bool)]

    def stat(key):
        vals = [r["score"][key] for r in ok if isinstance(r.get("score"), dict) and isinstance(r["score"].get(key), (int, float))]
        return {"mean": (sum(vals) / len(vals)) if vals else None, "median": statistics.median(vals) if vals else None, "n": len(vals)}

    sp = entry_split(entry)
    out = {"format": "isg-lab-tierB-summary/v2", "run": entry["run"], "step": entry.get("step"), "split": sp, "replay_split": entry.get("replay_split"),
           "task": TASK, "checkpoint": {"repo": entry.get("repo"), "file": entry.get("file"), "sha256": entry.get("sha256")} if sp != "replay" else None,
           "dataset_repo": entry.get("repo") if sp == "replay" else None, "created_at": LS.utc(), "horizon": horizon,
           "action_horizon": expected_action_horizon(entry, args.action_horizon) if sp != "replay" else None,
           "tag": entry_tag(entry), "server_args_extra": server_extra_args(entry) or None,
           "server_args": (server or {}).get("cmd") if sp != "replay" else None, "episodes": want, "n": len(ok), "n_listed": len(want),
           "n_success": sum(1 for r in ok if r["success"]), "success_rate": (sum(1 for r in ok if r["success"]) / len(ok)) if ok else None,
           "n_binary_success_ever": sum(1 for r in ok if r.get("binary_success_ever")),
           "binary_success_ever_rate": (sum(1 for r in ok if r.get("binary_success_ever")) / len(ok)) if ok else None,
           "geodesic_distance_m": stat("geodesic_distance_m"), "position_error_m": stat("position_error_m"), "orientation_error_deg": stat("orientation_error_deg"),
           "within_tolerance_rate": (sum(1 for r in ok if (r.get("score") or {}).get("within_tolerance")) / len(ok)) if ok else None,
           "error_count": len(st["errored"]) + len([e for e in want if e in recs and recs[e].get("error")]), "missing": st["missing"], "other_horizon": st["other_horizon"],
           "unmapped": st["unmapped"], "failure_reasons": {}, "mean_steps": (sum(r.get("steps") or 0 for r in ok) / len(ok)) if ok else None,
           "mean_wall_s": (sum(r.get("wall_s") or 0 for r in ok) / len(ok)) if ok else None}
    for r in ok:
        if not r["success"]:
            k = r.get("reason") or "unknown"
            out["failure_reasons"][k] = out["failure_reasons"].get(k, 0) + 1
    if sp == "replay":
        def agg(key):
            vals = [r.get(key) for r in ok if isinstance(r.get(key), (int, float))]
            return {"max": max(vals) if vals else None, "mean": (sum(vals) / len(vals)) if vals else None, "n": len(vals),
                    "per_episode": {str(r["episode_index"]): r.get(key) for r in ok}}

        out["base_trajectory_max_deviation_m"] = agg("base_trajectory_max_deviation_m")
        out["base_trajectory_method_floor_m"] = agg("base_trajectory_method_floor_m")
        out["base_trajectory_max_deviation_raw_m"] = agg("base_trajectory_max_deviation_raw_m")
        out["base_end_vs_goal_base_m"] = agg("base_end_vs_goal_base_m")
        out["base_end_vs_goal_base_yaw_deg"] = agg("base_end_vs_goal_base_yaw_deg")
        out["base_trajectory_note"] = ("base_trajectory_max_deviation_m: max over the streamed steps of |base_xy(replay) - base_xy(demo)|, the demo trajectory dead-reckoned "
                                       "from its own observation.state[0:3] (robot-frame base velocity) at 30 Hz from the restored init pose, with the proprio yaw-rate scale "
                                       "measured in the same replay taken out; base_trajectory_method_floor_m is the same integration of the replay's OWN recorded velocities "
                                       "against its simulated path (the method's resolution: the proprio rz-joint velocity over-reads the yaw rate by 3-17 % and reads "
                                       "+0.01 rad/s while driving straight, so the dead reckoning drifts in yaw); where floor ~ deviation the metric is at its limit. "
                                       "base_end_vs_goal_base_m: the replay's final base pose vs the instance's goal base pose (exact reference).")
        out["first_success_step"] = {str(r["episode_index"]): r.get("first_success_step") for r in ok}
        out["demo_length"] = {str(r["episode_index"]): r.get("demo_length") for r in ok}
        out["actions_streamed"] = {str(r["episode_index"]): r.get("actions_streamed") for r in ok}
    out["complete"] = not st["missing"] and not st["other_horizon"] and not st["unmapped"] and out["error_count"] == 0 and len(ok) == len(want)
    os.makedirs(lab_dir(entry), exist_ok=True)
    with open(os.path.join(lab_dir(entry), "summary.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


def work_entry(entry: dict, horizon: int, instances: dict, args) -> None:
    key = entry_key(entry)
    st = LS.read_state()
    st["current"] = {"run": entry["run"], "step": entry.get("step"), "split": entry_split(entry), "replay_split": entry.get("replay_split"), "horizon": horizon,
                     "lab_dir": os.path.relpath(lab_dir(entry), WT), "episodes_total": len(wanted_episodes(entry, instances)), "started_at": LS.utc()}
    st["phase"] = "tierB-relocalization"
    LS.write_state(st)
    LS.status()
    server = None
    if entry_split(entry) != "replay":
        git_pull_act()
        ckpt = ensure_checkpoint(entry)
        server = start_server(ckpt, entry["run"], int(entry["step"]), args.port, server_extra_args(entry))
    run_pool(entry, horizon, server, args, lambda: entry_state(entry, horizon, instances)["missing"])
    if server_rejections(entry):
        block_on_rejection(entry)
        write_summary(entry, horizon, instances, server, args)
        return
    # errored episodes: one re-run (the re-run's record keeps the retried flag, so an episode is retried once, not every poll)
    bad = entry_state(entry, horizon, instances)["errored"]
    bad = [e for e in bad if not episode_records(entry).get(e, {}).get("retried")]
    if bad:
        log(f"{key}: re-running {len(bad)} errored episodes {bad}")
        run_pool(entry, horizon, server, args, lambda: [e for e in entry_state(entry, horizon, instances)["errored"] if e in bad], overwrite=True, max_rounds=1)
        for e in bad:
            p = os.path.join(lab_dir(entry), f"episode_{e:06d}.json")
            if os.path.exists(p):
                r = read_record(p)
                if r.get("error"):
                    r["retried"] = True
                    json.dump(r, open(p, "w"), indent=1)
        if server_rejections(entry):
            block_on_rejection(entry)
            write_summary(entry, horizon, instances, server, args)
            return
    summary = write_summary(entry, horizon, instances, server, args)
    log(f"{key}: summary complete={summary['complete']} n={summary['n']}/{summary['n_listed']} success={summary['n_success']} "
        f"binary_ever={summary['n_binary_success_ever']} errors={summary['error_count']} missing={summary['missing']}")
    st = LS.read_state()
    st.setdefault("completed", []).append({"run": entry["run"], "step": entry.get("step"), "split": entry_split(entry), "replay_split": entry.get("replay_split"),
                                           "horizon": horizon, "complete": summary["complete"], "n": summary["n"], "n_success": summary["n_success"], "at": LS.utc()})
    st["current"] = None
    LS.write_state(st)
    LS.status()


def server_rejections(entry: dict) -> list:
    """Episodes whose error is a server-side rejection of the connection / observation (websocket close 1008, 'Rejected
    observation', a missing goal view ...): a bridge <-> checkpoint mismatch, never a policy result."""
    out = []
    for e, r in episode_records(entry).items():
        err = str(r.get("error") or "")
        if any(s in err for s in ("ConnectionClosedError", "code=1008", "Goal-conditioned checkpoint needs", "needs goal images for views", "Rejected observation")):
            out.append(e)
    return sorted(out)


def block_on_rejection(entry: dict) -> None:
    """The user's rule (2026-10-07 18:30Z): a rejected connection stops the entry and blocks the loop; a note says why."""
    bad = server_rejections(entry)
    first = episode_records(entry).get(bad[0], {}) if bad else {}
    reason = (f"server rejected the bridge's connection / observation for {entry_key(entry)} (episodes {bad}): "
              f"{str(first.get('error') or '')[:300]}. The loop stopped on this entry (nothing else runs until `blocked` is cleared); "
              "the bridge <-> checkpoint mismatch must be fixed first.")
    log("BLOCKED: " + reason)
    st = LS.read_state()
    st["blocked"] = reason
    LS.write_state(st)
    LS.note("BLOCKED (loop stopped): " + reason)
    LS.status()


def add_decision(text: str) -> None:
    st = LS.read_state()
    nd = st.setdefault("needs_decision", [])
    if text not in nd:
        nd.append(text)
        LS.write_state(st)
        LS.note(text)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parallel", type=int, default=2, help="evaluator processes at a time (each ~9-10 GB of GPU memory)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--action-horizon", type=int, default=16)
    ap.add_argument("--video-size", type=int, default=540)
    ap.add_argument("--episodes-per-process", type=int, default=10, help="an evaluator process exits after this many episodes (GPU memory grows ~2.5 GB/h); the pool starts a fresh one")
    ap.add_argument("--use-workers", action="store_true", help="evaluator fix 2.2: persistent eval_worker.py processes (one scene load per process lifetime) instead of a process per entry")
    ap.add_argument("--episodes-per-job", type=int, default=5, help="--use-workers: episodes per job file (small jobs keep the pause latency short)")
    ap.add_argument("--worker-max-gpu-mib", type=int, default=20000, help="--use-workers: a worker restarts itself (logged) above this GPU memory")
    ap.add_argument("--no-planner", action="store_true", help="evaluator fix 2.1: no cuRobo planner in policy / replay streams (evaluator_patches.NoPlanner)")
    ap.add_argument("--cleanup", action="store_true", help="evaluator fix 2.1: gc + torch.cuda.empty_cache after every episode")
    ap.add_argument("--num-envs", type=int, default=1, help="> 1: the vectorized evaluator (N cloned scenes per worker process, one batched env.step per control step); 1 = the single-env path")
    ap.add_argument("--warm-renders", type=int, default=100, help="vectorized: renders after a batch's reset before the first observation")
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--poll-s", type=int, default=600)
    args = ap.parse_args(argv)
    skipped_reported: set = set()
    while True:
        try:
            try:
                LS.pull()
            except Exception as e:  # noqa: BLE001
                log(f"pull failed: {e!r}")
            queue = json.load(open(QUEUE))
            st = LS.read_state()
            st["queue_seen_updated_at"] = queue.get("updated_at")
            st["queue_paused"] = bool(queue.get("paused"))
            st["queue_horizon_steps"] = queue.get("horizon_steps")
            LS.write_state(st)
            if os.path.exists(PAUSE_FLAG):
                log(f"paused by {os.path.relpath(PAUSE_FLAG, WT)} (lab maintenance: GPU needed for instance work); nothing launched")
            elif st.get("blocked"):
                log(f"blocked: {str(st['blocked'])[:160]} -- nothing run (clear `blocked` in lab_tools/state.json after the fix)")
            elif queue.get("format") not in ("isg-tierB-queue/v2",):
                log(f"queue format {queue.get('format')!r} is not isg-tierB-queue/v2; nothing run")
            elif queue.get("paused"):
                log("queue paused: nothing run")
            else:
                horizon = int(queue.get("horizon_steps") or 1500)
                worked = False
                # the lab's own entries first (lab_tools/local_queue.json: the user's execution-settings test, 2026-10-07 18:30Z), then machine 1's queue in order
                entries = list(queue.get("entries", []))
                if os.path.exists(LOCAL_QUEUE):
                    try:
                        local = json.load(open(LOCAL_QUEUE))
                        entries = list(local.get("entries", [])) + entries
                    except Exception as e:  # noqa: BLE001
                        log(f"local queue unreadable: {e!r}")
                for entry in entries:
                    if TASK not in (entry.get("tasks") or [TASK]):
                        continue
                    sp = data_split(entry)
                    if sp not in SPLITS:
                        add_decision(f"queue entry {entry_key(entry)}: unknown split {sp!r}; skipped")
                        continue
                    instances = load_instances(sp)
                    es = entry_state(entry, horizon, instances)
                    if complete(entry, horizon, instances):
                        continue
                    if es.get("no_instances"):
                        msg = f"queue entry {entry_key(entry)} needs {os.path.relpath(es['no_instances'], WT)} (training episode -> instance map), which does not exist yet; skipped until it does"
                        if msg not in skipped_reported:
                            log(msg)
                            skipped_reported.add(msg)
                        continue
                    if es["other_horizon"]:
                        add_decision(f"queue entry {entry_key(entry)}: {len(es['other_horizon'])} existing episode records were run at another horizon than the queue's "
                                     f"{horizon} (episodes {es['other_horizon'][:10]}...). They are not overwritten; say whether to re-run them at {horizon} into the same path "
                                     "(replacing the old records) or to keep the old results.")
                        if not es["missing"] and not es["errored"]:
                            continue
                    if es["unmapped"]:
                        msg = (f"queue entry {entry_key(entry)}: episodes {es['unmapped']} have no verified instance in {os.path.relpath(SPLITS[sp]['instances'], WT)}; "
                               "they are skipped (the summary lists them under unmapped) until the map covers them")
                        if msg not in skipped_reported:
                            log(msg)
                            LS.note(msg)
                            skipped_reported.add(msg)
                        if not es["missing"] and not es["errored"]:
                            write_summary(entry, horizon, instances, current_server(), args)
                            continue
                    work_entry(entry, horizon, instances, args)
                    worked = True
                    break  # re-read the queue
                if worked:
                    continue
                log("queue: nothing to do")
        except Exception as e:  # noqa: BLE001
            log(f"loop error: {e!r}")
            import traceback

            traceback.print_exc()
        if args.once:
            return 0
        time.sleep(args.poll_s)


if __name__ == "__main__":
    sys.exit(main())
