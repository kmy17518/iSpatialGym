"""i7-vast worker on a Vast instance (boot.sh runs it after the bundles are unpacked): runs the fast-flow jobs the coordinator assigns
to this instance - each one the job_fast.sh invocation submit_fast.py's sbatch would have run on the cluster, at the same paths, with
the SLURM_* variables a slurm job sees, pinned to CPUs of its own - and uploads each finished job dir.

R2 ($ISG_S3, this job's prefix only):
  assign/<instance>/<id>.json   a job spec written by the coordinator (vastctl.py): id, name, job_dir, script, time_s, cpus, log,
                                code / jobdata archives, scenes, git roots
  hb/<instance>.json            this worker's heartbeat (every 60 s): slots, running, done, free resources
  out/<id>/                     jobdir.tar.zst (the whole job dir), slurm.log (the job's stdout + stderr = its slurm log),
                                cache_new.tar.zst (sampling cache files the instance wrote meanwhile: the coordinator adds the
                                ones the cluster lacks, as a cluster job would have written them)
  done/<id>.json                written after every upload succeeded: rc, timing, GPU, instance
  ctl/stop, ctl/stop_<instance> finish the running jobs, take no new ones, exit
Exit 0: stopped or idle for ISG_IDLE_S; 3: the host is unusable (boot.sh destroys the instance either way).
"""
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
import time

S3 = os.environ["ISG_S3"].rstrip("/")
EP = os.environ["S3_ENDPOINT_URL"]
INST = os.environ.get("ISG_INSTANCE") or os.uname().nodename
IDLE_S = int(os.environ.get("ISG_IDLE_S", "1200"))
VRAM_PER_JOB_GB = float(os.environ.get("ISG_VRAM_PER_JOB_GB", "20"))
RAM_PER_JOB_GB = float(os.environ.get("ISG_RAM_PER_JOB_GB", "40"))
MAX_JOBS = int(os.environ.get("ISG_MAX_JOBS", "0"))  # 0: as many as fit
LC = "/vision/u/minyeong/projects/ispatialgym-lab-vec-eval-cache"
CACHES = ["/vision/u/minyeong/projects/isg_home/trainfast/cache/sampling"]  # the asset-geometry cache stays on the instance (derived from decrypted meshes)
SECRET_ENV = ("AWS_", "CONTAINER_API_KEY", "VAST_", "GHCR_", "ISG_", "S3_", "REPLAY_S3", "RUNPOD_")
LOCK = threading.Lock()
STATE = {"running": {}, "done": 0, "failed": 0, "started": time.time(), "last_work": time.time(), "phase": "ready"}


def log(*a):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}]", *a, flush=True)


def s5(*args, check=True, inp=None, timeout=3600):
    r = subprocess.run(["s5cmd", "--endpoint-url", EP, "--retry-count", "10", *args], capture_output=True, text=True, input=inp,
                       timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"s5cmd {args[0]} failed: {r.stderr.strip()[:300]}")
    return r


def s3_ls(prefix):
    r = s5("ls", f"{S3}/{prefix}*", check=False)
    return [l.split()[-1] for l in r.stdout.splitlines() if l.strip() and not l.strip().startswith("DIR")]


def s3_exists(key):
    return s5("ls", f"{S3}/{key}", check=False).returncode == 0


def fetch_extract(url, dest="/"):
    """Archive at url -> extracted under dest (zstd tar or plain tar), retried; mtimes kept, owner not."""
    os.makedirs("/work/dl", exist_ok=True)
    f = "/work/dl/" + url.replace("/", "_").replace(":", "_")
    for attempt in range(4):
        try:
            s5("cp", "-c", "4", "-p", "64", url, f, timeout=1800)
            if url.endswith(".zst"):
                cmd = f"zstd -d -q -c {shlex.quote(f)} | tar -x --no-same-owner -C {shlex.quote(dest)}"
            else:
                cmd = f"tar -x --no-same-owner -C {shlex.quote(dest)} -f {shlex.quote(f)}"
            subprocess.run(["bash", "-o", "pipefail", "-c", cmd], check=True, timeout=1800)
            os.remove(f)
            return
        except Exception as e:  # noqa: BLE001
            log(f"FETCH_RETRY {url} ({attempt + 1}): {str(e)[:200]}")
            if os.path.exists(f):
                os.remove(f)
            time.sleep(10)
    raise RuntimeError(f"fetch failed: {url}")


# ---------------------------------------------------------------------------------------------------------------- resources


def effective_cpus():
    """The CPUs this container may use: its affinity, cut to the CFS quota (hosts advertise 128+ CPUs and allow ~20)."""
    cpus = sorted(os.sched_getaffinity(0))
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()
        if q != "max":
            cpus = cpus[: max(1, int(int(q) // int(p)))]
    except (OSError, ValueError):
        try:
            q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
            p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
            if q > 0:
                cpus = cpus[: max(1, q // p)]
        except (OSError, ValueError):
            pass
    return cpus


def gpu_info():
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    name, drv, mem = [x.strip() for x in out[0].split(",")]
    return {"gpu": name, "driver": drv, "vram_gb": float(mem) / 1024, "n_gpus": len(out)}


def ram_gb():
    try:
        m = open("/sys/fs/cgroup/memory.max").read().strip()
        if m != "max":
            return int(m) / 2**30
    except OSError:
        pass
    return int(open("/proc/meminfo").read().split()[1]) / 2**20


GPU = gpu_info()
CPUS = effective_cpus()
RAM = ram_gb()
FREE_CPUS = list(CPUS[1:])  # CPU 0 for the worker, the transfers and the heartbeat
CPUS_PER_JOB = int(os.environ.get("ISG_CPUS_PER_JOB", "9"))
SLOTS = int(min(GPU["n_gpus"] * (GPU["vram_gb"] // VRAM_PER_JOB_GB), len(FREE_CPUS) // CPUS_PER_JOB, RAM // RAM_PER_JOB_GB))
if MAX_JOBS:
    SLOTS = min(SLOTS, MAX_JOBS)


def heartbeat():
    with LOCK:
        run = {k: {"start": v["start"], "name": v["name"], "cpus": v["cpus_str"]} for k, v in STATE["running"].items()}
        hb = {"instance": INST, "phase": STATE["phase"], "time": int(time.time()), "slots": SLOTS, "running": run, "done": STATE["done"],
              "failed": STATE["failed"], "free_cpus": len(FREE_CPUS), **GPU, "cpus": len(CPUS), "ram_gb": round(RAM, 1),
              "disk_free_gb": round(shutil.disk_usage("/work").free / 2**30, 1), "up_s": int(time.time() - STATE["started"])}
    try:
        hb["gpu_used_mb"] = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                                           capture_output=True, text=True, timeout=30).stdout.strip()
        hb["load"] = open("/proc/loadavg").read().split()[0]
        hb["mem_used_gb"] = round((int(open("/proc/meminfo").read().split("MemAvailable:")[0].split()[1]) -
                                   int(open("/proc/meminfo").read().split("MemAvailable:")[1].split()[0])) / 2**20, 1)
    except Exception:  # noqa: BLE001
        pass
    s5("pipe", f"{S3}/hb/{INST}.json", inp=json.dumps(hb), check=False, timeout=120)


def hb_loop():
    while True:
        try:
            heartbeat()
        except Exception as e:  # noqa: BLE001
            log(f"heartbeat failed: {e!r}")
        time.sleep(60)


# ---------------------------------------------------------------------------------------------------------------- inputs

HAVE = "/work/have"
FETCH_LOCKS = {}


def ensure(key, url, dest="/"):
    """Fetch + extract an archive once per instance (code snapshots, Behavior-1K copies, scene textures)."""
    with LOCK:
        lk = FETCH_LOCKS.setdefault(key, threading.Lock())
    with lk:
        mark = os.path.join(HAVE, key.replace("/", "_"))
        if os.path.exists(mark):
            return
        t = time.time()
        fetch_extract(url, dest)
        os.makedirs(HAVE, exist_ok=True)
        open(mark, "w").write(f"{time.time() - t:.0f}\n")
        log(f"HAVE {key} ({time.time() - t:.0f} s)")


def ensure_scene(scene, replay_s3):
    """A scene's pre-converted Kit textures (the replay's packs) into the local texture store, once."""
    with LOCK:
        lk = FETCH_LOCKS.setdefault("tex:" + scene, threading.Lock())
    with lk:
        done = f"/work/texstore/.scenes/{scene}.complete"
        if os.path.exists(done):
            return
        r = subprocess.run(["s5cmd", "--endpoint-url", EP, "cat", f"{replay_s3}/bundle/textures/{scene}/parts.txt"], capture_output=True,
                           text=True)
        if r.returncode != 0:
            log(f"TEXTURES none for {scene} (Kit converts them)")
            open(done, "w").write("none\n")
            return
        t = time.time()
        parts = [l.split()[0] for l in r.stdout.splitlines() if l.strip()]
        ths = [threading.Thread(target=fetch_extract, args=(f"{replay_s3}/bundle/textures/{scene}/{p}", "/work/texstore")) for p in parts]
        for th in ths:
            th.start()
        for th in ths:
            th.join()
        subprocess.run(["bash", "-c", "find /work/texstore -maxdepth 1 -type f -exec chmod 444 {} +"], check=False)
        open(done, "w").write(f"{len(parts)} parts {time.time() - t:.0f} s\n")
        log(f"TEXTURES {scene}: {len(parts)} parts ({time.time() - t:.0f} s)")


def add_git_roots(roots):
    """/etc/isg_git_heads rows for the git stand-in (dir, full sha, porcelain file)."""
    with LOCK:
        have = open("/etc/isg_git_heads").read() if os.path.exists("/etc/isg_git_heads") else ""
        with open("/etc/isg_git_heads", "a") as f:
            for r in roots:
                line = f"{r['dir']}\t{r['sha']}\t{r.get('porcelain') or '-'}"
                if line not in have.splitlines():
                    f.write(line + "\n")


# ---------------------------------------------------------------------------------------------------------------- jobs


def job_env(spec, cpus):
    """The environment of a slurm batch job for this spec: the instance's own variables minus every credential / worker variable,
    the SLURM_* the job scripts read, USER minyeong (the per-job /tmp/$USER folders), the job's CPUs as the thread caps see them."""
    env = {k: v for k, v in os.environ.items() if not k.startswith(SECRET_ENV)}
    start = int(time.time())
    env.update({"USER": "minyeong", "LOGNAME": "minyeong", "HOME": "/root", "SLURM_JOB_ID": str(spec["id"]), "SLURM_JOBID": str(spec["id"]),
                "SLURM_JOB_NAME": spec["name"], "SLURM_CPUS_PER_TASK": str(spec["cpus"]), "SLURM_JOB_END_TIME": str(start + int(spec["time_s"])),
                "SLURM_JOB_START_TIME": str(start), "SLURM_SUBMIT_DIR": spec.get("cwd") or "/", "SLURM_JOB_ACCOUNT": spec.get("acct", "vision"),
                "SLURM_JOB_PARTITION": spec.get("part", "svl"), "SLURM_NTASKS": "1", "SLURM_NNODES": "1", "SLURM_CPUS_ON_NODE": str(len(cpus)),
                "CUDA_VISIBLE_DEVICES": "0", "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8"})
    env.pop("TMPDIR", None)
    return env, start


def live_logs(jid, jd, logp, p):
    """While the job runs: its job.txt and slurm log -> out/<id>/live/ every 5 min (progress for the coordinator / agents)."""
    while p.poll() is None:
        time.sleep(300)
        for f, name in ((os.path.join(jd, "job.txt"), "job.txt"), (logp, "slurm.log")):
            if os.path.exists(f):
                s5("cp", f, f"{S3}/out/{jid}/live/{name}", check=False, timeout=300)


def run_job(spec, replay_s3):
    jid = str(spec["id"])
    jd = spec["job_dir"]
    cpus = spec["_cpus"]
    t_fetch = time.time()
    rc, err = None, None
    try:
        for c in spec["code"]:
            ensure(c["key"], f"{S3}/{c['key']}", "/")
        for sc in spec.get("scenes") or []:
            ensure_scene(sc, replay_s3)
        if os.path.exists(jd):  # a retried job: the attempt before on this instance is replaced by the cluster's state
            shutil.rmtree(jd)
        fetch_extract(f"{S3}/{spec['jobdata']}", "/")
        add_git_roots(spec.get("git_roots") or [])
    except Exception as e:  # noqa: BLE001
        err = f"inputs: {e!r}"
    t_run = time.time()
    logp = spec["log"]
    os.makedirs(os.path.dirname(logp), exist_ok=True)
    marker = f"/work/marks/{jid}"
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    open(marker, "w").close()
    if err is None:
        env, start = job_env(spec, cpus)
        cpus_str = ",".join(map(str, cpus))
        with LOCK:
            STATE["running"][jid]["phase"] = "run"
        with open(logp, "ab") as out:
            out.write(f"VAST_BACKEND id={jid} instance={INST} gpu=\"{GPU['gpu']}\" driver={GPU['driver']} cpus={cpus_str} "
                      f"start={time.strftime('%Y-%m-%dT%H:%M:%S%z')}\n".encode())
            out.flush()
            # slurm's time limit: SIGTERM at the limit, SIGKILL 30 s later (KillWait); the job's EXIT trap runs on TERM
            p = subprocess.Popen(["taskset", "-c", cpus_str, "bash", spec["script"], jd], stdout=out, stderr=subprocess.STDOUT, env=env,
                                 cwd=spec.get("cwd") if os.path.isdir(spec.get("cwd") or "") else "/", start_new_session=True)
            threading.Thread(target=live_logs, args=(jid, jd, logp, p), daemon=True).start()
            try:
                rc = p.wait(timeout=int(spec["time_s"]))
            except subprocess.TimeoutExpired:
                out.write(b"slurmstepd: error: *** JOB CANCELLED DUE TO TIME LIMIT (vast worker) ***\n")
                out.flush()
                os.killpg(p.pid, 15)
                try:
                    rc = p.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(p.pid, 9)
                    rc = p.wait()
                rc = rc if rc else -15
        # leftovers of the job's process group (collectors whose parent died)
        try:
            os.killpg(p.pid, 9)
        except OSError:
            pass
    t_end = time.time()
    stats = {}
    try:
        stats = json.load(open(os.path.join(jd, "job_stats.json")))
    except Exception:  # noqa: BLE001
        pass
    backend = {"backend": "vast", "instance": INST, "gpu": GPU["gpu"], "driver": GPU["driver"], "vram_gb": round(GPU["vram_gb"], 1),
               "cpus": ",".join(map(str, cpus)), "instance_cpus": len(CPUS), "instance_ram_gb": round(RAM, 1), "slots": SLOTS,
               "rc": rc, "error": err, "fetch_s": round(t_run - t_fetch, 1), "elapsed_s": round(t_end - t_run, 1),
               "start": t_run, "end": t_end, "label": spec.get("label"), "machine_id": spec.get("machine_id"),
               "account": spec.get("account"), "dph": spec.get("dph")}
    up_err = None
    try:
        if os.path.isdir(jd):
            json.dump(backend, open(os.path.join(jd, "backend.json"), "w"), indent=1)
            pk = f"/work/pack/{jid}"
            os.makedirs(pk, exist_ok=True)
            # the whole job dir (never a decrypted asset: job dirs hold json / hdf5 / logs; usd files are refused)
            subprocess.run(["bash", "-o", "pipefail", "-c",
                            f"tar -C / --format=posix --exclude='*.usd' --exclude='*.usda' --exclude='*.usdc' --exclude='*.encrypted*' "
                            f"-cf - {shlex.quote(jd.lstrip('/'))} | zstd -q -T2 -3 -o {pk}/jobdir.tar.zst -f"], check=True)
            s5("cp", f"{pk}/jobdir.tar.zst", f"{S3}/out/{jid}/jobdir.tar.zst")
            # cache files written on this instance since the job started (the coordinator keeps the ones the cluster lacks)
            lst = f"{pk}/cache_new.txt"
            with open(lst, "w") as fl:
                for c in CACHES:
                    if os.path.isdir(c):
                        r = subprocess.run(["find", c, "-maxdepth", "1", "-type", "f", "-newer", marker, "-not", "-name", ".*"],
                                           capture_output=True, text=True)
                        for l in r.stdout.splitlines():
                            fl.write(l.lstrip("/") + "\n")
            if os.path.getsize(lst):
                subprocess.run(["bash", "-o", "pipefail", "-c", f"tar -C / --format=posix -cf - -T {lst} | zstd -q -T2 -3 -o {pk}/cache_new.tar.zst -f"],
                               check=True)
                s5("cp", f"{pk}/cache_new.tar.zst", f"{S3}/out/{jid}/cache_new.tar.zst")
            shutil.rmtree(pk, ignore_errors=True)
        if os.path.exists(logp):
            s5("cp", logp, f"{S3}/out/{jid}/slurm.log")
    except Exception as e:  # noqa: BLE001
        up_err = repr(e)
    done = {**backend, "id": jid, "name": spec["name"], "job_dir": jd, "upload_error": up_err,
            "attempts": stats.get("attempts"), "written": stats.get("written"), "demos": stats.get("demos"), "successes": stats.get("successes")}
    s5("pipe", f"{S3}/done/{jid}.json", inp=json.dumps(done), check=False)
    with LOCK:
        STATE["running"].pop(jid, None)
        FREE_CPUS.extend(cpus)
        FREE_CPUS.sort()
        STATE["done" if rc == 0 and not err and not up_err else "failed"] += 1
        STATE["last_work"] = time.time()
    log(f"JOB_END {jid} {spec['name']} rc={rc} err={err} upload_err={up_err} run_s={t_end - t_run:.0f} stats={ {k: done[k] for k in ('attempts', 'written', 'demos', 'successes')} }")
    try:
        heartbeat()
    except Exception:  # noqa: BLE001
        pass


def main():
    replay_s3 = os.environ.get("REPLAY_S3", "s3://isg-demos/isg-1020-b1k")
    log(f"WORKER {INST} gpu={GPU} cpus={len(CPUS)} ram={RAM:.0f} GB slots={SLOTS} (vram/{VRAM_PER_JOB_GB} GB, cpus/{CPUS_PER_JOB}, ram/{RAM_PER_JOB_GB} GB)")
    if SLOTS < 1:
        STATE["phase"] = "unusable"
        heartbeat()
        log("UNUSABLE: no job fits")
        return 3
    threading.Thread(target=hb_loop, daemon=True).start()
    started = set()
    while True:
        stop = s3_exists("ctl/stop") or s3_exists(f"ctl/stop_{INST}")
        with LOCK:
            n_run = len(STATE["running"])
        if stop:
            STATE["phase"] = "draining"
            if n_run == 0:
                log("STOP")
                STATE["phase"] = "stopped"
                heartbeat()
                return 0
        else:
            for name in sorted(s3_ls(f"assign/{INST}/")):
                jid = name[:-5] if name.endswith(".json") else name
                if jid in started or s3_exists(f"done/{jid}.json"):
                    started.add(jid)
                    continue
                with LOCK:
                    if len(STATE["running"]) >= SLOTS:
                        break
                    spec = json.loads(s5("cat", f"{S3}/assign/{INST}/{name}").stdout)
                    need = int(spec.get("cpus") or CPUS_PER_JOB)
                    if len(FREE_CPUS) < need:
                        break
                    cpus = FREE_CPUS[:need]
                    del FREE_CPUS[:need]
                    spec["_cpus"] = cpus
                    STATE["running"][jid] = {"start": int(time.time()), "name": spec["name"], "cpus_str": ",".join(map(str, cpus)), "phase": "fetch"}
                    STATE["phase"] = "run"
                    STATE["last_work"] = time.time()
                started.add(jid)
                log(f"JOB_START {jid} {spec['name']} cpus={cpus} {spec['job_dir']}")
                threading.Thread(target=run_job, args=(spec, replay_s3), daemon=True).start()
        with LOCK:
            idle = not STATE["running"] and time.time() - STATE["last_work"] > IDLE_S
            if not STATE["running"] and STATE["phase"] == "run":
                STATE["phase"] = "ready"
        if idle:
            log(f"IDLE for {IDLE_S} s")
            STATE["phase"] = "idle_exit"
            heartbeat()
            return 0
        time.sleep(15)


if __name__ == "__main__":
    sys.exit(main())
