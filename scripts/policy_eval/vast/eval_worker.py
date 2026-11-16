"""Policy-eval worker on one rented GPU instance (Vast.ai, og-isgeval-; started by boot.sh once the bundle is unpacked).

Run spec ``runs/<run>/spec.json`` (fetched by boot.sh to /work/units.json): the code snapshot, the eval dataset bundle, the
checkpoints (``ckpts``: tag -> kind pi | act | dp | gt, HF repo + step, server / client / run_isg arguments) and the units (one per
checkpoint x eval instance: ``<tag>__<dataset index>``). The instance works on one checkpoint at a time: it fetches the
checkpoint, starts its policy servers (``servers_per_gpu`` per GPU, each on its own port), and ``procs_per_gpu`` simulator slots per
GPU claim that checkpoint's units (conditional puts of ``claims/<unit>.json``, longest budget first), each one
``python -m ispatialgym.run_isg --mode policy --policy ispatialgym.eval.ws_run_policy:WSPolicy`` process (one instance per
process, Kit app data seeded from this instance's template). A finished unit uploads its run folder to ``out/<tag>/<unit>.tar.zst``
and writes ``done/<unit>.json`` (the RUN_ISG_RESULT record + the meta.json fields the summary needs: success / evaluation / qa /
timings / policy client). When the checkpoint has no open unit left, its servers stop and the next checkpoint with open units is
taken. Exits 0 when nothing is left (boot.sh destroys the instance), 3 when the instance looks unusable.
"""

import json
import os
import re
import shutil
import socket
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import boto3
import botocore

ROOT_URL = os.environ["ISG_S3"]  # s3://<bucket>/jobs/og-isgeval-20261116: bundle/, kit_tpl/, runs/
RUN = os.environ["ISG_RUN"]
S3_URL = f"{ROOT_URL}/runs/{RUN}"
BUCKET, PREFIX = S3_URL[len("s3://"):].split("/", 1)
TEX_URL = os.environ.get("ISG_TEX_S3", f"s3://{BUCKET}/isg-1020-b1k/bundle/textures")
#: read-only Kit templates of earlier fleets (same Isaac Sim, same Behavior-1K commit): the render fleet's, isg-1020-b1k's
EXTRA_TPL_URLS = [u for u in os.environ.get("ISG_EXTRA_TPL_S3", f"s3://{BUCKET}/jobs/og-isgrend-20261115/kit_tpl,s3://{BUCKET}/isg-1020-b1k/kit_tpl").split(",") if u]
INSTANCE = os.environ["ISG_INSTANCE"]
TAG = os.environ.get("ISG_TAG", "")
ENVP = os.environ["ENVP"]
WORK = Path("/work")
LOG = Path("/var/log/isg")
B1K_COMMIT = "6a6194600"
B1K_DIR = os.environ.get("ISG_B1K_DIR", "/vision/u/minyeong/projects/isg_home/oracle_fix/code/b1k_6a6194600_p59fc36c6")
S5 = ["s5cmd", "--endpoint-url", os.environ["S3_ENDPOINT_URL"], "--numworkers", "32"]
STAGGER_S = float(os.environ.get("ISG_STAGGER_S", "15"))
MAX_UNIT_FAILS = 3
PIN = os.environ.get("ISG_PIN", "1") == "1"
TEMPLATE_EXCLUDE = ("local/logs", "tmp", "local/data/documents/Kit/shared/screenshots", "global/cache/texturecache")
EVALDATA = WORK / "evaldata"

s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"], region_name="auto",
                  config=botocore.config.Config(retries={"max_attempts": 10, "mode": "adaptive"}))
t_start = time.time()
lock = threading.Lock()
stop_claiming = threading.Event()
state = {"phase": "start", "running": {}, "done": [], "failed": [], "successes": 0, "startup_fails": 0, "episodes_ok": 0, "ckpt": None,
         "servers": {}}


def log(msg):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} +{time.time() - t_start:.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------- R2
def key(k):
    return f"{PREFIX}/{k}"


def put_json(k, obj, if_none_match=False):
    kw = {"IfNoneMatch": "*"} if if_none_match else {}
    try:
        s3.put_object(Bucket=BUCKET, Key=key(k), Body=json.dumps(obj, default=str).encode(), **kw)
        return True
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("PreconditionFailed", "412") or e.response["ResponseMetadata"]["HTTPStatusCode"] == 412:
            return False
        raise


def get_json(k):
    try:
        return json.loads(s3.get_object(Bucket=BUCKET, Key=key(k))["Body"].read())
    except botocore.exceptions.ClientError as e:
        if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
            return None
        raise


def exists(k):
    try:
        s3.head_object(Bucket=BUCKET, Key=key(k))
        return True
    except botocore.exceptions.ClientError:
        return False


def list_names(prefix):
    names, token = [], None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": key(prefix)}
        if token:
            kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        names += [o["Key"][len(key(prefix)):] for o in r.get("Contents", [])]
        if not r.get("IsTruncated"):
            return names
        token = r["NextContinuationToken"]


def delete(k):
    try:
        s3.delete_object(Bucket=BUCKET, Key=key(k))
    except botocore.exceptions.ClientError:
        pass


def s5(*args, check=True):
    r = subprocess.run(S5 + list(args), capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"s5cmd {' '.join(args[:3])} failed: {r.stderr.strip()[-500:]}")
    return r


def fetch_tar(url, dest):
    """A .tar.zst (s3 url, or a key under the job root) unpacked at dest."""
    url = url if url.startswith("s3://") else f"{ROOT_URL}/{url}"
    Path(dest).mkdir(parents=True, exist_ok=True)
    for attempt in range(4):
        r = subprocess.run(f"{' '.join(S5)} cat '{url}' | zstd -d -q | tar -x --no-same-owner -C '{dest}'", shell=True, capture_output=True, text=True)
        if r.returncode == 0:
            return
        log(f"FETCH_RETRY {url}: {r.stderr[-200:]}")
        time.sleep(10)
    raise RuntimeError(f"fetch failed: {url}")


# ---------------------------------------------------------------- host
def gpus():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    rows = [[c.strip() for c in line.split(",")] for line in out.strip().splitlines()]
    return [{"index": int(r[0]), "name": r[1], "driver": r[2], "mem_mb": int(float(r[3]))} for r in rows]


ALLOWED = sorted(os.sched_getaffinity(0))


def cpu_quota():
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()
        return None if q == "max" else int(q) / int(p)
    except (OSError, ValueError):
        return None


QUOTA = cpu_quota()
Q = min(len(ALLOWED), QUOTA) if QUOTA else float(len(ALLOWED))


def mem_gb():
    total = int(re.search(r"MemTotal:\s+(\d+)", open("/proc/meminfo").read()).group(1)) / 2**20
    try:
        m = open("/sys/fs/cgroup/memory.max").read().strip()
        if m != "max":
            total = min(total, int(m) / 2**30)
    except OSError:
        pass
    return total


def slot_cpus(k, n_slots, reserve):
    """Slot k's CPUs: a disjoint block of the quota minus ``reserve`` CPUs (the policy servers, the worker)."""
    c = max(2, int((Q - reserve) // n_slots))
    return [ALLOWED[(k * c + i) % len(ALLOWED)] for i in range(c)]


def utilization():
    out = {}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
        out["gpu_util"] = [int(x.split(",")[0]) for x in r.stdout.strip().splitlines()]
        out["gpu_mem_mb"] = [int(x.split(",")[1]) for x in r.stdout.strip().splitlines()]
    except Exception:  # noqa: BLE001
        pass
    f = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    idle, total = f[3] + f[4], sum(f)
    prev = getattr(utilization, "prev", None)
    utilization.prev = (idle, total)
    if prev and total > prev[1]:
        out["cpu_busy"] = round(1 - (idle - prev[0]) / (total - prev[1]), 3)
    mi = open("/proc/meminfo").read()
    tot = int(re.search(r"MemTotal:\s+(\d+)", mi).group(1))
    avail = int(re.search(r"MemAvailable:\s+(\d+)", mi).group(1))
    out["mem_used_gb"] = round((tot - avail) / 2**20, 1)
    return out


def kit_key(g):
    return re.sub(r"[^A-Za-z0-9.\n]", "_", f"{g['name']}, {g['driver']}") + "_" + B1K_COMMIT


# ---------------------------------------------------------------- Kit app data (the render worker's rules)
TPL = WORK / "kit_template"
TEX = WORK / "texstore"
tpl_lock = threading.Lock()


def template_ready():
    return (TPL / ".complete").exists()


def fetch_template(kk):
    for base in [f"{ROOT_URL}/kit_tpl"] + EXTRA_TPL_URLS:
        for kind in ("unit", "warm"):
            url = f"{base}/{kk}/{kind}.tar.zst"
            if s5("ls", url, check=False).returncode == 0:
                tmp = WORK / "kit_template.dl"
                shutil.rmtree(tmp, ignore_errors=True)
                tmp.mkdir(parents=True)
                r = subprocess.run(f"{' '.join(S5)} cat '{url}' | zstd -d -q | tar -x --no-same-owner -C {tmp}", shell=True)
                if r.returncode == 0:
                    (tmp / ".complete").touch()
                    tmp.rename(TPL)
                    log(f"KIT_TEMPLATE fetched {url}")
                    return url
                shutil.rmtree(tmp, ignore_errors=True)
    return None


def save_template(appdata, kk, kind):
    with tpl_lock:
        if template_ready():
            return
        tmp = WORK / f"kit_template.tmp.{os.getpid()}.{threading.get_ident()}"
        shutil.rmtree(tmp, ignore_errors=True)
        subprocess.run(["chmod", "-R", "u+rwX", str(appdata)], capture_output=True)
        texcl = " ".join(f"--exclude=./{e}" for e in TEMPLATE_EXCLUDE)
        tmp.mkdir(parents=True)
        r = subprocess.run(f"tar -C {appdata} {texcl} -cf - . | tar -C {tmp} -xf -", shell=True, capture_output=True, text=True)
        if r.returncode:
            shutil.rmtree(tmp, ignore_errors=True)
            return
        (tmp / ".complete").touch()
        tmp.rename(TPL)
    k = f"{ROOT_URL}/kit_tpl/{kk}/{kind}.tar.zst"
    if s5("ls", k, check=False).returncode != 0:
        subprocess.run(f"tar -C {TPL} --exclude=./.complete -cf - . | zstd -q -T4 -3 | {' '.join(S5)} pipe '{k}'", shell=True, capture_output=True)
        log(f"KIT_TEMPLATE uploaded {k}")


def make_appdata(path):
    shutil.rmtree(path, ignore_errors=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    if template_ready():
        with tpl_lock:
            subprocess.run(["cp", "-a", f"{TPL}/.", str(path)], check=True)
        (path / ".complete").unlink(missing_ok=True)
        warm = True
    else:
        path.mkdir(parents=True)
        warm = False
    tex = path / "global/cache/texturecache"
    tex.mkdir(parents=True, exist_ok=True)
    if TEX.exists():
        names = os.listdir(TEX)
        complete = {x[:-len(".ovtex.desc")] for x in names if x.endswith(".ovtex.desc")}
        for name in names:
            stem, sep, _ = name.partition(".ovtex.")
            if sep and stem in complete and not os.path.lexists(tex / name):
                os.symlink(TEX / name, tex / name)
    (path / "tmp").mkdir(exist_ok=True)
    return warm


scene_locks = {}


def ensure_scene_textures(scene):
    marker = TEX / ".scenes" / f"{scene}.complete"
    with lock:
        sl = scene_locks.setdefault(scene, threading.Lock())
    with sl:
        if marker.exists():
            return True
        r = s5("cp", f"{TEX_URL}/{scene}/parts.txt", str(WORK / f"tex_{scene}.parts"), check=False)
        if r.returncode:
            return False
        for part in (WORK / f"tex_{scene}.parts").read_text().split():
            try:
                fetch_tar(f"{TEX_URL}/{scene}/{part}", TEX)
            except RuntimeError:
                return False
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.touch()
        return True


def save_textures(appdata):
    """Textures this process converted (regular files in its cache) into the instance store, for the next processes."""
    tex = appdata / "global/cache/texturecache"
    n = 0
    if tex.exists():
        for f in tex.iterdir():
            if f.is_file() and not f.is_symlink() and not (TEX / f.name).exists():
                try:
                    shutil.copy2(f, TEX / f.name)
                    n += 1
                except OSError:
                    pass
    return n


snap_locks = {}


def ensure_snapshot(name):
    with lock:
        sl = snap_locks.setdefault(name, threading.Lock())
    with sl:
        root = WORK / "snap" / name
        if (root / ".ready").exists():
            return root
        tmp = WORK / "snap" / f"{name}.tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        fetch_tar(f"bundle/code/{name}.tar.zst", tmp)
        b = tmp / "Behavior-1K"
        shutil.rmtree(b, ignore_errors=True)
        (b / "datasets").mkdir(parents=True)
        for d in ("OmniGibson", "bddl3", "docs"):
            os.symlink(Path(B1K_DIR) / d, b / d)
        for d in ("behavior-1k-assets", "omnigibson-robot-assets", "2026-challenge-task-instances", "omnigibson.key"):
            os.symlink(WORK / "data" / d, b / "datasets" / d)
        os.symlink(root / "ispatialgym/assets/datasets/ispatialgym-assets", b / "datasets" / "ispatialgym-assets")
        (tmp / ".ready").touch()
        tmp.rename(root)
        log(f"snapshot {name} ready")
        return root


# ---------------------------------------------------------------- spec, dataset, policies, checkpoints
SPEC = json.load(open(WORK / "units.json"))
UNITS = {u["unit"]: u for u in SPEC["units"]}
CKPTS = SPEC["ckpts"]
data_lock = threading.Lock()


def ensure_dataset():
    with data_lock:
        if (EVALDATA / ".ready").exists():
            return EVALDATA
        fetch_tar(SPEC["dataset"], EVALDATA)
        (EVALDATA / ".ready").touch()
        log(f"dataset ready: {SPEC['dataset']}")
        return EVALDATA


def ensure_policy_env(kind):
    """The serving checkout + venv + its uv Python of ``kind`` (spec ``policy_envs[kind]``: a tar of absolute paths, unpacked at /)."""
    if kind == "gt":
        return
    marker = WORK / f".policy_env_{kind}"
    if marker.exists():
        return
    fetch_tar(SPEC["policy_envs"][kind], "/")
    marker.touch()
    log(f"policy env {kind} ready")


def ensure_ckpt(tag):
    """The checkpoint's eval file(s) from Hugging Face: pi -> the step tar extracted (its <step>/ dir); ACT / DP -> the .pt."""
    c = CKPTS[tag]
    if c["kind"] == "gt":
        return None
    d = WORK / "ckpt" / tag
    ready = d / ".ready"
    if ready.exists():
        return Path(ready.read_text().strip())
    d.mkdir(parents=True, exist_ok=True)
    s8 = f"{int(c['step']):08d}"
    url = f"https://huggingface.co/{c['repo']}/resolve/main/eval/step-{s8}." + ("tar" if c["kind"] == "pi" else "pt")
    hdr = f"-H 'Authorization: Bearer {os.environ['HF_TOKEN']}'" if os.environ.get("HF_TOKEN") else ""
    t0 = time.time()
    for attempt in range(4):
        if c["kind"] == "pi":
            cmd = f"curl -sSfL --retry 5 {hdr} '{url}' | tar -x -C '{d}'"
        else:
            cmd = f"curl -sSfL --retry 5 {hdr} -o '{d}/model.pt' '{url}'"
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if r.returncode == 0:
            break
        log(f"CKPT_RETRY {tag}: {r.stderr[-200:]}")
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        time.sleep(20)
    else:
        raise RuntimeError(f"checkpoint {tag} download failed")
    path = next(p for p in d.iterdir() if p.is_dir()) if c["kind"] == "pi" else d / "model.pt"
    ready.write_text(str(path))
    log(f"checkpoint {tag} ready in {time.time() - t0:.0f}s: {path}")
    return path


class Server:
    """One policy server process on one GPU."""

    def __init__(self, tag, gpu, port, path):
        self.tag, self.gpu, self.port, self.path = tag, gpu, port, path
        self.proc = None
        self.logf = LOG / f"server_{tag}_{gpu}_{port}.log"

    def start(self):
        c = CKPTS[self.tag]
        pe = SPEC["policy_paths"][c["kind"]]
        env = {k: v for k, v in os.environ.items() if k in ("PATH", "LANG", "LC_ALL", "HOME", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES")}
        env.update({"CUDA_VISIBLE_DEVICES": str(self.gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "HF_HUB_OFFLINE": "1", "PYTHONUNBUFFERED": "1",
                    "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "OPENPI_DATA_HOME": str(WORK / "openpi_data"), "TORCH_HOME": str(WORK / "torch_home"),
                    "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"})
        if c["kind"] == "pi":
            cmd = [pe["python"], "scripts/b1k/serve_b1k.py", "--robot", "b1k/R1Pro", "--task", f"b1k/{c['task_alias']}", "--policy.config", "pi05_b1k",
                   "--policy.dir", str(self.path), "--port", str(self.port)] + list(c.get("server_args") or [])
        else:
            cmd = [pe["python"], "scripts/b1k/serve_b1k.py", "--model-path", str(self.path), "--port", str(self.port)] + list(c.get("server_args") or [])
        fh = open(self.logf, "ab")
        self.proc = subprocess.Popen(cmd, cwd=pe["cwd"], env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
        t0 = time.time()
        while time.time() - t0 < 900:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server {self.tag} gpu {self.gpu} exited rc={self.proc.returncode}: {self.logf.read_text(errors='replace')[-800:]}")
            text = self.logf.read_text(errors="replace")
            if "listening on" in text:
                log(f"SERVER {self.tag} gpu {self.gpu} port {self.port} ready in {time.time() - t0:.0f}s")
                return
            time.sleep(3)
        raise RuntimeError(f"server {self.tag} gpu {self.gpu} not ready after 900s")

    def alive(self):
        return self.proc is not None and self.proc.poll() is None

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, 15)
                self.proc.wait(timeout=30)
            except Exception:  # noqa: BLE001
                try:
                    os.killpg(self.proc.pid, 9)
                except ProcessLookupError:
                    pass


# ---------------------------------------------------------------- units
local_fails = {}


def taken_units():
    return set(n[:-5] for n in list_names("done/")) | set(n[:-5] for n in list_names("claims/")) | set(n[:-5] for n in list_names("dead/"))


def open_units(tag=None):
    taken = taken_units()
    return [u for u in UNITS.values() if u["unit"] not in taken and local_fails.get(u["unit"], 0) < 2 and (tag is None or u["ckpt"] == tag)]


def claim_next(tag):
    if stop_claiming.is_set():
        return None
    with lock:
        if exists("ctl/stop") or exists(f"ctl/stop_{INSTANCE}"):
            log("CTL stop: no new claims")
            stop_claiming.set()
            return None
        cands = sorted(open_units(tag), key=lambda u: -u["budget"])
        for u in cands:
            if put_json(f"claims/{u['unit']}.json", {"instance": INSTANCE, "tag": TAG, "time": time.time()}, if_none_match=True):
                return u
        return None


KEEP_ENV = ("PATH", "LANG", "LC_ALL", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "VK_DRIVER_FILES", "TERM")


def proc_env(snap, appdata, home, gpu, ncpu):
    ncpu = max(1, min(int(ncpu), 8))
    env = {k: os.environ[k] for k in KEEP_ENV if k in os.environ}
    env.update({
        "OMNI_CRASHREPORTER_ENABLED": "0",
        "PATH": f"{ENVP}/bin:" + env.get("PATH", "/usr/bin:/bin"), "CONDA_PREFIX": ENVP,
        "PYTHONPATH": f"{snap}:{snap}/Behavior-1K/OmniGibson:{snap}/Behavior-1K/bddl3",
        "OMNIGIBSON_DATA_PATH": f"{snap}/Behavior-1K/datasets", "OMNIGIBSON_APPDATA_PATH": str(appdata), "TMPDIR": str(appdata / "tmp"),
        "OMNIGIBSON_HEADLESS": "1", "OMNI_KIT_ACCEPT_EULA": "YES", "HDF5_USE_FILE_LOCKING": "FALSE", "HF_HUB_OFFLINE": "1",
        "OMNIGIBSON_TRACK_STATE_UPDATES": "1", "OMP_NUM_THREADS": "1", "OMNIGIBSON_CHANGE_DETECTION_NUMPY": "1",
        "ISPATIALGYM_FAST_GEOMETRY": "1", "ISPATIALGYM_FAST_SUPPORT": "1", "ISPATIALGYM_FAST_ASSIGN": "1",
        "ISPATIALGYM_SHARED_GOAL_EVAL": "1", "ISPATIALGYM_GC_FREEZE": "1", "ISPATIALGYM_BATCHED_WORLD_SYNC": "1",
        "ISPATIALGYM_SHARD_FORMAT": "legacy", "ISPATIALGYM_SEEDED_SPHERE_FIT": "off", "NUMPY_MADVISE_HUGEPAGE": "0",
        "ISPATIALGYM_CUROBO_WARMUP_ITERS": "0", "PYTHONHASHSEED": "0", "PYTHONDONTWRITEBYTECODE": "1",
        "HOME": str(home), "XDG_CACHE_HOME": f"{home}/.cache", "TORCHINDUCTOR_CACHE_DIR": str(WORK / "inductor_cache"),
        "TRITON_CACHE_DIR": str(WORK / "triton_cache"), "NUMBA_CACHE_DIR": f"{home}/numba",
        "ISPATIALGYM_REACH_BANDS": f"{home}/reach_bands", "ISPATIALGYM_CACHE_DIR": f"{home}/isg_cache",
        "ISPATIALGYM_SAMPLING_CACHE": f"{home}/isg_cache/sampling", "ISPATIALGYM_HELD_HOLD_CACHE": f"{home}/isg_cache/held_hold.json",
        "ISPATIALGYM_REACH_VOLUME": f"{home}/reach_volume", "MKL_NUM_THREADS": str(ncpu), "OPENBLAS_NUM_THREADS": str(ncpu),
        "NUMBA_NUM_THREADS": str(ncpu), "CUDA_VISIBLE_DEVICES": str(gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_CACHE_PATH": str(WORK / "cuda_cache"), "CUDA_CACHE_MAXSIZE": str(4 * 2**30),
        # the eval dataset and its fixes (scripts/policy_eval/job_env.sh)
        "ISPATIALGYM_INSTANCES_PATH": str(EVALDATA / "ispatialgym-eval-instances"),
        "ISPATIALGYM_EVAL_BDDL_OVERLAY": str(EVALDATA / "bddl_overlay"),
        "ISPATIALGYM_EVAL_IGNORE_COLLECTION_LIGHTING": "1",
        "ISG_TRAIN_TASKS": str(EVALDATA / "train_tasks.jsonl"),
    })
    for d in ("numba", "reach_bands", "isg_cache/sampling", "reach_volume"):
        Path(home, d).mkdir(parents=True, exist_ok=True)
    for d in ("inductor_cache", "triton_cache", "cuda_cache"):
        (WORK / d).mkdir(exist_ok=True)
    return env


def run_proc(cmd, env, cwd, logf, status, startup_s, stall_s, total_s, cpus=None):
    """Run one run_isg process in its own session, pinned to ``cpus``; kill it on a start / log stall or timeout. Returns
    (rc, the RUN_ISG_RESULT record or None, reason)."""
    if PIN and cpus:
        cmd = ["taskset", "-c", ",".join(map(str, cpus))] + list(cmd)
    with open(logf, "ab") as fh:
        p = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    status["pid"] = p.pid
    t0 = last_growth = time.time()
    size, pending, result, result_at, started = 0, "", None, None, False
    reason = ""
    while True:
        rc = p.poll()
        try:
            text_size = os.path.getsize(logf)
        except OSError:
            text_size = size
        if text_size != size:
            with open(logf, "rb") as fh:
                fh.seek(size)
                chunk = fh.read().decode(errors="replace")
            size = text_size
            last_growth = time.time()
            lines = (pending + chunk).split("\n")
            pending = lines.pop()
            for line in lines:
                if line.startswith("RUN_ISG_RESULT "):
                    try:
                        result = json.loads(line[len("RUN_ISG_RESULT "):])
                    except ValueError:
                        result = {"status": "unparsable"}
                    result_at = time.time()
                elif "Welcome to OmniGibson" in line:
                    started = True
                    status["phase"] = "load"
        if rc is not None:
            return rc, result, reason
        now = time.time()
        if result_at is not None and now - result_at > 180:
            reason = "exit hang after RUN_ISG_RESULT"
        elif not started and now - t0 > startup_s:
            reason = f"no start after {startup_s}s"
        elif now - last_growth > stall_s:
            reason = f"log stalled {stall_s}s"
        elif now - t0 > total_s:
            reason = f"timeout {total_s}s"
        if reason:
            log(f"KILL {status.get('unit')}: {reason}")
            try:
                os.killpg(p.pid, 9)
            except ProcessLookupError:
                pass
            p.wait()
            return -9, result, reason
        time.sleep(3)


def meta_summary(meta):
    """The fields of a run's meta.json the eval summary reads (scripts/policy_eval/summarize.py)."""
    keep = ("success", "status", "reason", "n_steps", "evaluation_step", "evaluation_kind", "evaluation_settled", "success_first_step",
            "success_final", "closest_approach", "closest_goal_approach", "settle_eval", "settle_steps", "budget", "timings", "qa",
            "policy_client", "task_family", "task", "subtask", "variant", "instance_id", "error_metric", "errors", "gt_policy")
    return {k: meta.get(k) for k in keep if k in meta}


def run_unit(unit, gpu, slot, port, kk, cpus):
    uid, c = unit["unit"], CKPTS[unit["ckpt"]]
    udir = WORK / "units" / uid
    shutil.rmtree(udir, ignore_errors=True)
    udir.mkdir(parents=True)
    status = {"unit": uid, "gpu": gpu, "slot": slot, "port": port, "budget": unit["budget"], "started": time.time(), "phase": "setup"}
    with lock:
        state["running"][slot] = status
    t0 = time.time()
    appdata = WORK / "appdata" / f"{uid}.{slot}"
    try:
        snap = ensure_snapshot(SPEC["code"])
        data = ensure_dataset()
        ensure_scene_textures("common")
        ensure_scene_textures(unit["scene"])
        warm = make_appdata(appdata)
        env = proc_env(snap, appdata, udir / "home", gpu, len(cpus))
        tro = data / "ispatialgym-eval-instances" / unit["tro"]
        out = udir / "out"
        if c["kind"] == "gt":
            policy, pk = "ispatialgym.eval.gt_policy:GTPolicy", dict(c.get("policy_kwargs") or {})
        else:
            policy, pk = "ispatialgym.eval.ws_run_policy:WSPolicy", {"url": f"ws://127.0.0.1:{port}", **(c.get("policy_kwargs") or {})}
        cmd = [f"{ENVP}/bin/python", "-u", "-m", "ispatialgym.run_isg", "--instance", str(tro), "--out", str(out), "--mode", "policy",
               "--policy", policy, "--policy-kwargs", json.dumps(pk)] + list(SPEC.get("run_isg_args") or []) + list(c.get("run_isg_args") or [])
        logf = udir / "run.log"
        status["phase"] = "start"
        log(f"START {uid} slot={slot} gpu={gpu} port={port} budget={unit['budget']} warm={warm}")
        rc, result, reason = run_proc(cmd, env, str(snap), logf, status, startup_s=1500, stall_s=1800,
                                      total_s=2400 + 3.0 * unit["budget"], cpus=cpus)
        elapsed = time.time() - t0
        runs = [d for d in out.iterdir() if d.is_dir()] if out.exists() else []
        meta = json.loads((runs[0] / "meta.json").read_text()) if runs and (runs[0] / "meta.json").exists() else {}
        status_ = (result or {}).get("status") or meta.get("status")
        if result is None or status_ in (None, "error"):
            raise RuntimeError(f"rc={rc} status={status_} {reason} {str((result or {}).get('error'))[:300]}")
        status["phase"] = "upload"
        shutil.copy2(logf, runs[0] / "run.log")
        r = subprocess.run(f"tar -C {out} -cf - . | zstd -q -T2 -3 | {' '.join(S5)} pipe '{S3_URL}/out/{unit['ckpt']}/{uid}.tar.zst'",
                           shell=True, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"upload failed: {r.stderr[-300:]}")
        put_json(f"done/{uid}.json", {"unit": uid, "ckpt": unit["ckpt"], "index": unit["index"], "instance": INSTANCE, "tag": TAG,
                                      "gpu_name": GPUS[gpu]["name"], "driver": GPUS[gpu]["driver"], "slot": slot, "procs_per_gpu": PER_GPU,
                                      "servers_per_gpu": SERVERS_PER_GPU, "cpus": len(cpus), "seconds": elapsed, "warm": warm, "rc": rc,
                                      "result": result, "meta": meta_summary(meta), "time": time.time()})
        with lock:
            state["done"].append(uid)
            state["successes"] += 1
            state["episodes_ok"] += bool(result.get("success"))
            first = state["successes"] == 1
        log(f"DONE {uid} success={result.get('success')} steps={result.get('n_steps')} in {elapsed:.0f}s "
            f"(load {(meta.get('timings') or {}).get('load_s')}, episode {(meta.get('timings') or {}).get('episode_s')})")
        save_textures(appdata)
        if first:
            save_template(appdata, kk, "unit")
        return True
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        log(f"FAIL {uid} slot={slot}: {msg}")
        with lock:
            state["failed"].append(uid)
            local_fails[uid] = local_fails.get(uid, 0) + 1
            if state["successes"] == 0:
                state["startup_fails"] += 1
        try:
            n = len(list_names(f"fail/{uid}/"))
            (udir / "fail.txt").write_text(f"instance {INSTANCE} slot {slot} gpu {GPUS[gpu]['name']}\n{msg}\n{traceback.format_exc()}\n")
            s5("cp", str(udir / "fail.txt"), f"{S3_URL}/fail/{uid}/{INSTANCE}-{n}.txt", check=False)
            if (udir / "run.log").exists():
                s5("cp", str(udir / "run.log"), f"{S3_URL}/logs/units/{uid}/{INSTANCE}-{n}-run.log", check=False)
            if n + 1 >= MAX_UNIT_FAILS and state["successes"] > 0:
                put_json(f"dead/{uid}.json", {"unit": uid, "fails": n + 1, "last": msg}, if_none_match=True)
            delete(f"claims/{uid}.json")
        except Exception as e2:  # noqa: BLE001
            log(f"fail bookkeeping for {uid} failed: {e2}")
        return False
    finally:
        with lock:
            state["running"].pop(slot, None)
        shutil.rmtree(appdata, ignore_errors=True)
        shutil.rmtree(udir, ignore_errors=True)


def heartbeat_loop(kk):
    n = 0
    while True:
        util = utilization()
        with lock:
            running = [{k: v for k, v in r.items() if k != "pid"} for r in state["running"].values()]
            hb = {"instance": INSTANCE, "tag": TAG, "phase": state["phase"], "time": time.time(), "uptime": time.time() - t_start,
                  "gpus": [g["name"] for g in GPUS], "driver": GPUS[0]["driver"], "cpus": Q, "mem_gb": round(mem_gb(), 1), "run": RUN,
                  "kit_key": kk, "template": template_ready(), "ckpt": state["ckpt"], "procs_per_gpu": PER_GPU,
                  "servers_per_gpu": SERVERS_PER_GPU, "servers": state["servers"], "running": running, "done": len(state["done"]),
                  "failed": len(state["failed"]), "episodes_ok": state["episodes_ok"], "load1": os.getloadavg()[0],
                  "disk_free_gb": shutil.disk_usage("/work").free / 2**30, **util}
        try:
            put_json(f"hb/{INSTANCE}.json", hb)
            if n % 5 == 0:
                s5("cp", f"{LOG}/*", f"{S3_URL}/logs/{INSTANCE}/", check=False)
        except Exception as e:  # noqa: BLE001
            log(f"heartbeat failed: {e}")
        n += 1
        if state["phase"] in ("done", "failed"):
            return
        time.sleep(60)


def pick_ckpt():
    """The checkpoint with the most open units (ISG_CKPT pins one; ISG_PREFER_CKPT orders)."""
    pinned = os.environ.get("ISG_CKPT")
    opens = {}
    for u in open_units():
        opens[u["ckpt"]] = opens.get(u["ckpt"], 0) + 1
    if pinned:
        return pinned if opens.get(pinned) else None
    prefer = [t for t in os.environ.get("ISG_PREFER_CKPT", "").split(",") if t in opens]
    if prefer:
        return prefer[0]
    return max(opens, key=opens.get) if opens else None


def layout(tag):
    """Simulator processes and policy servers per GPU for a checkpoint: spec ``procs_per_gpu`` / ``servers_per_gpu`` (or the
    checkpoint's), else by VRAM: a 224 px eval process ~8 GB, a pi server ~10 GB, ACT / DP ~1.5 GB each."""
    c = CKPTS[tag]
    servers = int(os.environ.get("ISG_SERVERS_PER_GPU") or c.get("servers_per_gpu") or 1) if c["kind"] != "gt" else 0
    if os.environ.get("ISG_PROCS_PER_GPU") or c.get("procs_per_gpu"):
        procs = int(os.environ.get("ISG_PROCS_PER_GPU") or c["procs_per_gpu"])
    else:
        vram = min(g["mem_mb"] for g in GPUS) / 1024
        srv = {"pi": 10.0, "act": 1.5, "dp": 1.5, "gt": 0.0}[c["kind"]] * servers
        procs = max(1, int((vram - srv - 1.0) // 8.0))
        procs = min(procs, max(1, int((Q - 1 - 2 * servers * len(GPUS)) // (3 * len(GPUS)))), max(1, int(mem_gb() / len(GPUS) // 14)))
    return procs, servers


def slot_loop(slot, gpu, port, tag, kk, cpus):
    time.sleep(STAGGER_S * (slot // max(1, len(GPUS))))
    while not stop_claiming.is_set():
        srv = [s for s in state["server_objs"] if s.port == port]
        if srv and not srv[0].alive():
            log(f"server on port {port} died: restarting it")
            try:
                srv[0].start()
            except Exception as e:  # noqa: BLE001
                log(f"server restart failed: {e}")
                return
        unit = claim_next(tag)
        if unit is None:
            break
        run_unit(unit, gpu, slot, port, kk, cpus)
        with lock:
            bad = state["successes"] == 0 and state["startup_fails"] >= max(2, len(GPUS) * PER_GPU)
        if bad:
            log("instance unusable: every unit failed and none succeeded")
            stop_claiming.set()
    log(f"slot {slot} exits")


def main():
    global GPUS, PER_GPU, SERVERS_PER_GPU
    GPUS = gpus()
    PER_GPU, SERVERS_PER_GPU = 0, 0
    kk = kit_key(GPUS[0])
    log(f"WORKER instance={INSTANCE} gpus={[g['name'] for g in GPUS]} driver={GPUS[0]['driver']} cpus={Q} mem={mem_gb():.0f}GB "
        f"units={len(UNITS)} ckpts={list(CKPTS)} kit_key={kk}")
    hb = threading.Thread(target=heartbeat_loop, args=(kk,), daemon=True)
    state["phase"] = "setup"
    state["server_objs"] = []
    hb.start()
    if not fetch_template(kk):
        log("no Kit template for this GPU + driver: processes start cold")
    ensure_snapshot(SPEC["code"])
    ensure_dataset()
    port0 = 8100
    while not stop_claiming.is_set():
        tag = pick_ckpt()
        if tag is None:
            break
        c = CKPTS[tag]
        state["phase"], state["ckpt"] = "ckpt_setup", tag
        try:
            ensure_policy_env(c["kind"])
            path = ensure_ckpt(tag)
        except Exception as e:  # noqa: BLE001
            log(f"checkpoint {tag} setup failed: {e}")
            stop_claiming.set()
            break
        PER_GPU, SERVERS_PER_GPU = layout(tag)
        servers = []
        try:
            for g in GPUS:
                for k in range(SERVERS_PER_GPU):
                    s = Server(tag, g["index"], port0 + 10 * g["index"] + k, path)
                    s.start()
                    servers.append(s)
        except Exception as e:  # noqa: BLE001
            log(f"servers of {tag} failed: {e}")
            for s in servers:
                s.stop()
            stop_claiming.set()
            break
        state["server_objs"] = servers
        state["servers"] = {s.port: s.gpu for s in servers}
        n_slots = PER_GPU * len(GPUS)
        reserve = 1 + 2 * len(servers)
        slots = []
        for j in range(PER_GPU):
            for i, g in enumerate(GPUS):
                slot = j * len(GPUS) + i
                gp = [s.port for s in servers if s.gpu == g["index"]]
                port = gp[j % len(gp)] if gp else 0
                slots.append((slot, g["index"], port, slot_cpus(slot, n_slots, reserve)))
        log(f"CKPT {tag} ({c['kind']}): {PER_GPU} processes + {SERVERS_PER_GPU} servers per GPU, {len(slots)} slots, "
            f"{len(slots[0][3])} CPUs each, open units {len(open_units(tag))}")
        state["phase"] = "run"
        threads = [threading.Thread(target=slot_loop, args=(s, g, p, tag, kk, cp)) for s, g, p, cp in slots]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        for s in servers:
            s.stop()
        state["server_objs"], state["servers"] = [], {}
        if os.environ.get("ISG_CKPT"):
            break
    state["phase"] = "done" if state["successes"] or not state["failed"] else "failed"
    log(f"WORKER END done={len(state['done'])} failed={len(state['failed'])} ok_episodes={state['episodes_ok']}")
    time.sleep(70)
    return 0 if state["phase"] == "done" else 3


if __name__ == "__main__":
    sys.exit(main())
