"""Batched replay (render) worker on one Vast.ai instance, 2026-11-15 (og-isgrend-; adapted from train85/work/vast_replay/worker).

Claims work units of run $ISG_RUN (runs/<run>/units.json: the demos of one template and one purpose, train or eval) through
conditional puts of runs/<run>/claims/<unit>.json, unpacks the unit's inputs (demo dirs, TROs, template files at their absolute
cluster paths) at /, replays it with the run's code snapshot (``python -m ispatialgym.multienv.replay`` --slots 1
--batched-usd-edits --async-flush --proprio loaded --warmup-s 15, ISPATIALGYM_SHARD_FORMAT=legacy; --scene full, --image-size 224,
--settle-pad, --goal-images on; train: --scene-pert remove --lighting-pert replay, eval: both off), checks the LeRobot dataset,
uploads it with its goal images and records to runs/<run>/out/<unit>/ and writes runs/<run>/done/<unit>.json. Exits 0 when no
unit is left to claim (boot.sh then destroys the instance), non-zero when the instance looks unusable.

One Kit app-data folder per replay process, seeded from this instance's template (built by a short empty-scene start when
R2 has no template for this GPU model + driver + Behavior-1K commit, refreshed from the first successful unit, shared
through kit_tpl/<key>/ on R2) with the texture cache linked to an instance-local store of textures earlier processes
converted (the rules of cluster_tools/kit_appdata.sh, minus NFS).
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import boto3
import botocore

ROOT_URL = os.environ["ISG_S3"]  # s3://<bucket>/jobs/og-isgrend-20261115: bundle/, kit_tpl/, runs/
RUN = os.environ["ISG_RUN"]
S3_URL = f"{ROOT_URL}/runs/{RUN}"
BUCKET, PREFIX = S3_URL[len("s3://"):].split("/", 1)
ROOT_PREFIX = ROOT_URL[len("s3://"):].split("/", 1)[1]
#: read-only texture packs and Kit templates of the isg-1020-b1k replay (same Isaac Sim, same Behavior-1K commit)
TEX_URL = os.environ.get("ISG_TEX_S3", f"s3://{BUCKET}/isg-1020-b1k/bundle/textures")
EXTRA_TPL_URL = os.environ.get("ISG_EXTRA_TPL_S3", f"s3://{BUCKET}/isg-1020-b1k/kit_tpl")
INSTANCE = os.environ["ISG_INSTANCE"]
TAG = os.environ.get("ISG_TAG", "")
ENVP = os.environ["ENVP"]
WORK = Path("/work")
LOG = Path("/var/log/isg")
B1K_COMMIT = "6a6194600"  # p59fc36c6 patches: same Isaac Sim / Kit, same template key
#: the patched Behavior-1K the campaign snapshots run (boot list: unpacked at / from the eval bundle, read only)
B1K_DIR = os.environ.get("ISG_B1K_DIR", "/vision/u/minyeong/projects/isg_home/oracle_fix/code/b1k_6a6194600_p59fc36c6")
S5 = ["s5cmd", "--endpoint-url", os.environ["S3_ENDPOINT_URL"], "--numworkers", "32"]
#: ispatialgym.multienv.replay --warmup-s: renders of the first state before the first frame (texture streaming, exposure)
WARMUP_S = float(os.environ.get("ISG_WARMUP_S", "15"))
STAGGER_S = float(os.environ.get("ISG_STAGGER_S", "20"))
MAX_UNIT_FAILS = 3
GPU_MODE = os.environ.get("ISG_GPU_ID_MODE", "cuda_visible")
#: pin every Kit / replay process to its own share of the container's CPU quota (taskset; children inherit the mask)
PIN = os.environ.get("ISG_PIN", "1") == "1"

TEMPLATE_EXCLUDE = ("local/logs", "tmp", "local/data/documents/Kit/shared/screenshots", "global/cache/texturecache")

s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"], region_name="auto",
                  config=botocore.config.Config(retries={"max_attempts": 10, "mode": "adaptive"}))
t_start = time.time()
lock = threading.Lock()
SLOT_CPUS = {}
stop_claiming = threading.Event()
state = {"phase": "start", "running": {}, "done": [], "failed": [], "frames_done": 0, "startup_fails": 0, "successes": 0}


def log(msg):
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} +{time.time() - t_start:.0f}s] {msg}", flush=True)


# ---------------------------------------------------------------- R2
def key(k):
    return f"{PREFIX}/{k}"


def put_json(k, obj, if_none_match=False):
    kw = {"IfNoneMatch": "*"} if if_none_match else {}
    try:
        s3.put_object(Bucket=BUCKET, Key=key(k), Body=json.dumps(obj).encode(), **kw)
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
    """Object names under <run prefix>/<prefix> (without it)."""
    out, token = [], None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": key(prefix)}
        if token:
            kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        out += [o["Key"][len(key(prefix)):] for o in r.get("Contents", [])]
        if not r.get("IsTruncated"):
            return out
        token = r["NextContinuationToken"]


def delete(k):
    try:
        s3.delete_object(Bucket=BUCKET, Key=key(k))
    except Exception as e:  # noqa: BLE001
        log(f"delete {k} failed: {e}")


def s5(*args, check=True):
    r = subprocess.run(S5 + list(args), capture_output=True, text=True)
    if check and r.returncode:
        raise RuntimeError(f"s5cmd {' '.join(args[:3])} failed: {r.stderr.strip()[-500:]}")
    return r


# ---------------------------------------------------------------- instance
def gpus():
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    rows = [[c.strip() for c in line.split(",")] for line in out.strip().splitlines()]
    return [{"index": int(r[0]), "name": r[1], "driver": r[2], "mem_mb": int(float(r[3]))} for r in rows]


ALLOWED = sorted(os.sched_getaffinity(0))


def cpu_quota():
    """The cgroup's CPU quota in CPUs (cpu.max quota / period), None without one."""
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()
        return None if q == "max" else int(q) / int(p)
    except (OSError, ValueError):
        return None


QUOTA = cpu_quota()
#: usable CPUs: the visible ones, capped by the quota (Vast containers often see 128-256 CPUs with a 10-60 CPU quota)
Q = min(len(ALLOWED), QUOTA) if QUOTA else float(len(ALLOWED))


def cpu_count():
    return max(1, int(Q))


def slot_cpus(k, n_slots):
    """Slot k's CPUs: a disjoint block of max(2, floor((Q - 1) / n_slots)) allowed CPUs (1 CPU of the quota is left to the
    worker, s5cmd and zstd)."""
    c = max(2, int((Q - 1) // n_slots))
    return [ALLOWED[(k * c + i) % len(ALLOWED)] for i in range(c)]


def solo_cpus():
    """CPUs of a process that runs alone (warm-up, diagnostics): floor(Q - 1) of them."""
    return ALLOWED[:max(2, min(len(ALLOWED), int(Q - 1)))]


def cpu_stat():
    """The cgroup's CPU accounting (cpu.stat): usage and quota throttling counters."""
    out = {}
    try:
        for line in open("/sys/fs/cgroup/cpu.stat"):
            k, v = line.split()
            if k in ("usage_usec", "nr_periods", "nr_throttled", "throttled_usec"):
                out[k] = int(v)
    except (OSError, ValueError):
        pass
    return out


def mem_gb():
    total = int(re.search(r"MemTotal:\s+(\d+)", open("/proc/meminfo").read()).group(1)) / 2**20
    try:
        m = open("/sys/fs/cgroup/memory.max").read().strip()
        if m != "max":
            total = min(total, int(m) / 2**30)
    except OSError:
        pass
    return total


def kit_key(g):
    return re.sub(r"[^A-Za-z0-9.\n]", "_", f"{g['name']}, {g['driver']}") + "_" + B1K_COMMIT


# ---------------------------------------------------------------- Kit app data: template + texture store
TPL = WORK / "kit_template"
TEX = WORK / "texstore"
tpl_lock = threading.Lock()


def template_ready():
    return (TPL / ".complete").exists()


def fetch_template(kk):
    for url in (f"{ROOT_URL}/kit_tpl/{kk}/unit.tar.zst", f"{ROOT_URL}/kit_tpl/{kk}/warm.tar.zst", f"{EXTRA_TPL_URL}/{kk}/unit.tar.zst",
                f"{EXTRA_TPL_URL}/{kk}/warm.tar.zst"):
        if s5("ls", url, check=False).returncode == 0:
            k = url
            tmp = WORK / "kit_template.dl"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            r = subprocess.run(f"{' '.join(S5)} cat '{url}' | zstd -d -q | tar -x --no-same-owner -C {tmp}", shell=True)
            if r.returncode == 0:
                (tmp / ".complete").touch()
                tmp.rename(TPL)
                log(f"KIT_TEMPLATE fetched {k}")
                return k
            shutil.rmtree(tmp, ignore_errors=True)
    return None


def save_template(appdata, kk, kind):
    """Instance template from a finished process's app data; uploaded to kit_tpl/<key>/<kind>.tar.zst unless one exists."""
    with tpl_lock:
        tmp = WORK / f"kit_template.tmp.{os.getpid()}.{threading.get_ident()}"
        shutil.rmtree(tmp, ignore_errors=True)
        subprocess.run(["chmod", "-R", "u+rwX", str(appdata)], capture_output=True)
        excl = sum([["--exclude", e] for e in TEMPLATE_EXCLUDE], [])
        r = subprocess.run(["rsync", "-a", *excl, f"{appdata}/", f"{tmp}/"], capture_output=True, text=True) \
            if shutil.which("rsync") else None
        if r is None:  # no rsync in the image: tar pipe with the same excludes
            texcl = " ".join(f"--exclude=./{e}" for e in TEMPLATE_EXCLUDE)
            tmp.mkdir(parents=True)
            r = subprocess.run(f"tar -C {appdata} {texcl} -cf - . | tar -C {tmp} -xf -", shell=True, capture_output=True, text=True)
        if r.returncode:
            log(f"template save failed: {r.stderr[-300:]}")
            shutil.rmtree(tmp, ignore_errors=True)
            return
        (tmp / ".complete").touch()
        old = WORK / "kit_template.old"
        shutil.rmtree(old, ignore_errors=True)
        if TPL.exists():
            TPL.rename(old)
        tmp.rename(TPL)
        shutil.rmtree(old, ignore_errors=True)
        size = sum(f.stat().st_size for f in TPL.rglob("*") if f.is_file()) / 2**20
        log(f"KIT_TEMPLATE saved from {appdata.name} ({kind}, {size:.0f} MB)")
    k = f"{ROOT_URL}/kit_tpl/{kk}/{kind}.tar.zst"
    if s5("ls", k, check=False).returncode != 0:
        r = subprocess.run(f"tar -C {TPL} --exclude=./.complete -cf - . | zstd -q -T4 -3 | {' '.join(S5)} pipe '{k}'",
                           shell=True, capture_output=True, text=True)
        log(f"KIT_TEMPLATE upload {k}: rc={r.returncode}")


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
    n = 0
    if TEX.exists():
        names = os.listdir(TEX)
        complete = {x[:-len(".ovtex.desc")] for x in names if x.endswith(".ovtex.desc")}
        for name in names:
            stem, sep, _ = name.partition(".ovtex.")
            if sep and stem in complete and not os.path.lexists(tex / name):
                os.symlink(TEX / name, tex / name)
                n += 1
    (path / "tmp").mkdir(exist_ok=True)
    return warm, n


scene_locks = {}


def ensure_scene_textures(scene):
    """The scene's pack of pre-converted textures (bundle/textures/<scene>/, built from the cluster's texture store) in
    /work/texstore, read-only, once per instance; make_appdata() then links them into every process's texture cache."""
    marker = TEX / ".scenes" / f"{scene}.complete"
    with lock:
        sl = scene_locks.setdefault(scene, threading.Lock())
    with sl:
        if marker.exists():
            return True
        r = s5("cp", f"{TEX_URL}/{scene}/parts.txt", str(WORK / f"tex_{scene}.parts"), check=False)
        if r.returncode:
            log(f"TEXTURES {scene}: no pack")
            return False
        t0 = time.time()
        parts = [line.split()[0] for line in (WORK / f"tex_{scene}.parts").read_text().splitlines() if line.strip()]
        (TEX / ".scenes").mkdir(parents=True, exist_ok=True)
        dl = WORK / "dl"
        dl.mkdir(exist_ok=True)
        for part in parts:
            f = dl / f"tex_{scene}_{part}"
            for attempt in range(4):
                r = subprocess.run(["timeout", "300"] + S5 + ["--retry-count", "20", "cp", "-c", "6", "-p", "64",
                                                             f"{TEX_URL}/{scene}/{part}", str(f)], capture_output=True)
                if r.returncode == 0 and subprocess.run(["tar", "-x", "--no-same-owner", "-C", str(TEX), "-f", str(f)]).returncode == 0:
                    break
                log(f"TEXTURES {scene}/{part}: retry {attempt + 1}")
            else:
                f.unlink(missing_ok=True)
                log(f"TEXTURES {scene}: part {part} failed")
                return False
            f.unlink(missing_ok=True)
        subprocess.run(f"find {TEX} -maxdepth 1 -type f -perm /222 -exec chmod 444 {{}} +", shell=True)
        marker.touch()
        log(f"TEXTURES {scene}: {len(parts)} parts in {time.time() - t0:.0f}s")
        return True


def new_textures(appdata):
    """Textures a finished process converted itself (regular files in its texture cache, the rest are store links)."""
    tex = appdata / "global/cache/texturecache"
    if not tex.exists():
        return 0
    return sum(1 for n in os.listdir(tex) if n.endswith(".ovtex.desc") and not (tex / n).is_symlink())


def save_textures(appdata):
    """Converted textures of a finished process into the store: data files first, .desc last, never replacing a file."""
    tex = appdata / "global/cache/texturecache"
    if not tex.exists():
        return 0
    TEX.mkdir(exist_ok=True)
    parts = {}
    for name in os.listdir(tex):
        stem, sep, _ = name.partition(".ovtex.")
        if sep and not (tex / name).is_symlink():
            parts.setdefault(stem, []).append(name)
    n = 0
    for stem, names in parts.items():
        desc = stem + ".ovtex.desc"
        if desc not in names or (TEX / desc).exists():
            continue
        for name in sorted(x for x in names if x != desc) + [desc]:
            if not (TEX / name).exists():
                os.chmod(tex / name, 0o444)
                try:
                    os.link(tex / name, TEX / name)
                except FileExistsError:
                    pass
        n += 1
    return n


# ---------------------------------------------------------------- code
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
        tmp.mkdir(parents=True)
        r = subprocess.run(f"{' '.join(S5)} cat '{ROOT_URL}/bundle/code/{name}.tar.zst' | zstd -d -q | tar -x --no-same-owner -C {tmp}",
                           shell=True, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"snapshot {name} fetch failed: {r.stderr[-300:]}")
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


# ---------------------------------------------------------------- units
RUNSPEC = json.load(open(WORK / "units.json"))
UNITS = {u["unit"]: u for u in RUNSPEC["units"]}
CODE = RUNSPEC["code"]
for _u in UNITS.values():
    _u.setdefault("snapshot", CODE)
PREFER = [u for u in os.environ.get("ISG_PREFER", "").split(",") if u]
local_fails = {}
scenes_here = set()


def claim_next():
    """Claim the next unit: preferred ones first, then the longest (LPT, a short tail), favouring scenes this instance has
    already loaded among the 6 longest open units."""
    if stop_claiming.is_set():
        return None
    with lock:
        if exists("ctl/stop") or exists(f"ctl/stop_{INSTANCE}"):
            log("CTL stop: no new claims")
            stop_claiming.set()
            return None
        taken = set(n[:-5] for n in list_names("done/")) | set(n[:-5] for n in list_names("claims/")) \
            | set(n[:-5] for n in list_names("dead/"))
        open_units = [u for u in UNITS.values() if u["unit"] not in taken and local_fails.get(u["unit"], 0) < 2]
        if not open_units:
            return None
        open_units.sort(key=lambda u: -u["frames"])
        head = open_units[:6]
        local = [u for u in head if u["scene"] in scenes_here]
        order = [UNITS[p] for p in PREFER if p in UNITS and UNITS[p] in open_units] + local + open_units
        seen = set()
        for u in order:
            if u["unit"] in seen:
                continue
            seen.add(u["unit"])
            if put_json(f"claims/{u['unit']}.json", {"instance": INSTANCE, "tag": TAG, "time": time.time()}, if_none_match=True):
                scenes_here.add(u["scene"])
                return u
        return None


def fail_count(uid):
    return len(list_names(f"fail/{uid}/"))


def validate(ds, kept, image_size):
    """The unit's dataset against the replayer's records: the kept episodes and their (settle-padded) lengths, the 3 cameras'
    rgb + depth streams at image_size, 23-D action, files present."""
    import pyarrow.parquet as pq
    info = json.load(open(ds / "meta/info.json"))
    lengths = [int(r["length"]) for r in kept]
    if info["total_episodes"] != len(lengths) or info["total_frames"] != sum(lengths):
        raise ValueError(f"info: {info['total_episodes']} episodes / {info['total_frames']} frames, expected {len(lengths)} / {sum(lengths)}")
    feats = info["features"]
    cams = ("zed_link", "left_realsense_link", "right_realsense_link")
    need = [f"observation.rgb.{c}_camera_0" for c in cams] + [f"observation.depth_linear.{c}_camera_0" for c in cams]
    missing = [k for k in need if k not in feats]
    if missing or feats["action"]["shape"] != [23]:
        raise ValueError(f"features: missing {missing}, action {feats['action']['shape']}")
    bad = [k for k in need if image_size and list(feats[k]["shape"][:2]) != [image_size, image_size]]
    if bad:
        raise ValueError(f"image size: {[(k, feats[k]['shape']) for k in bad]} != {image_size}")
    eps = sorted((r for p in sorted((ds / "meta/episodes").glob("*/*.parquet")) for r in pq.read_table(p).to_pylist()),
                 key=lambda r: r["episode_index"])
    got = [int(r["length"]) for r in eps]
    if got != lengths:
        raise ValueError(f"episode lengths {got[:5]}... != replayed {lengths[:5]}...")
    files = set()
    for r in eps:
        files.add(info["data_path"].format(chunk_index=r["data/chunk_index"], file_index=r["data/file_index"]))
        for k in need:
            files.add(info["video_path"].format(video_key=k, chunk_index=r[f"videos/{k}/chunk_index"], file_index=r[f"videos/{k}/file_index"]))
    absent = [f for f in files if not (ds / f).is_file() or (ds / f).stat().st_size == 0]
    if absent:
        raise ValueError(f"missing files {absent[:3]}")
    return {"episodes": len(lengths), "frames": sum(lengths),
            "bytes": sum(f.stat().st_size for f in ds.rglob("*") if f.is_file())}


#: the only variables a replay process inherits (Vast injects credentials and tokens the replay has no use for)
KEEP_ENV = ("PATH", "LANG", "LC_ALL", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "VK_DRIVER_FILES", "TERM")


def proc_env(snap, appdata, home, gpu, ncpu):
    ncpu = max(1, min(int(ncpu), int(os.environ.get("ISG_MAX_THREADS", "8"))))  # BLAS pools: 127 threads on a solo slot is a loss
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
        "TRITON_CACHE_DIR": str(WORK / "triton_cache"), "HF_LEROBOT_HOME": f"{home}/lerobot_home", "NUMBA_CACHE_DIR": f"{home}/numba",
        "ISPATIALGYM_REACH_BANDS": f"{home}/reach_bands", "ISPATIALGYM_CACHE_DIR": f"{home}/isg_cache",
        "ISPATIALGYM_SAMPLING_CACHE": f"{home}/isg_cache/sampling", "ISPATIALGYM_HELD_HOLD_CACHE": f"{home}/isg_cache/held_hold.json",
        "ISPATIALGYM_REACH_VOLUME": f"{home}/reach_volume", "MKL_NUM_THREADS": str(ncpu), "OPENBLAS_NUM_THREADS": str(ncpu),
        "NUMBA_NUM_THREADS": str(ncpu),
        "CUDA_VISIBLE_DEVICES": str(gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_CACHE_PATH": str(WORK / "cuda_cache"),
        "CUDA_CACHE_MAXSIZE": str(4 * 2**30),
    })
    if GPU_MODE == "omnigibson":  # all GPUs visible, Kit told which one
        env.pop("CUDA_VISIBLE_DEVICES")
        env["OMNIGIBSON_GPU_ID"] = str(gpu)
    elif GPU_MODE == "none":
        env.pop("CUDA_VISIBLE_DEVICES")
    for d in ("lerobot_home", "numba", "reach_bands", "isg_cache/sampling", "reach_volume"):
        Path(home, d).mkdir(parents=True, exist_ok=True)
    for d in ("inductor_cache", "triton_cache"):
        (WORK / d).mkdir(exist_ok=True)
    bands = Path(snap, "runlogs/reach_bands_20260915/bands")
    if bands.is_dir():
        for f in bands.iterdir():
            shutil.copy2(f, Path(home, "reach_bands", f.name))
    return env


def run_proc(cmd, env, cwd, logf, status, n_expected, startup_s, stall_s, total_s, cpus=None):
    """Run a replay / Kit process in its own session, pinned to `cpus` (taskset; ISG_PIN=1); kill it on a stall or timeout.
    Records each REPLAY_DEMO's time, episode, frames and wall_s in status["demos"] and the REPLAY_SUMMARY record in
    status["summary"]. Returns (rc, done_count, reason)."""
    if PIN and cpus:
        cmd = ["taskset", "-c", ",".join(map(str, cpus))] + list(cmd)
    with open(logf, "ab") as fh:
        p = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    status["pid"] = p.pid
    status["t0"] = t0 = last_growth = time.time()
    status["demos"] = []
    size, pending, done = 0, "", 0
    seen_first = False
    done_at = summary_at = None
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
            last_growth = now = time.time()
            lines = (pending + chunk).split("\n")
            pending = lines.pop()
            for line in lines:
                if line.startswith("REPLAY_DEMO "):
                    # ispatialgym.multienv.replay: {"demo", "slot", "frames", "episode_index", "wall_s", "dataset"}
                    try:
                        rec = json.loads(line[len("REPLAY_DEMO "):])
                    except ValueError:
                        continue
                    done += 1
                    name = str(rec.get("demo", "")).rstrip("/").rsplit("/", 1)[-1]
                    status["demos"].append([round(now, 1), name, rec.get("frames"), rec.get("wall_s")])
                    status.setdefault("records", []).append(rec)
                elif line.startswith("REPLAY_SUMMARY "):
                    try:
                        status["summary"] = json.loads(line[len("REPLAY_SUMMARY "):])
                    except ValueError:
                        status["summary"] = {}
                    summary_at = now
                elif "KIT_TEST_OK" in line:
                    done += 1
                elif "Welcome to OmniGibson" in line:
                    seen_first = True
            seen_first = seen_first or done > 0
            status["done"] = done
            if done >= n_expected and done_at is None:
                done_at = time.time()
        if rc is not None:
            return rc, done, reason
        now = time.time()
        if summary_at is not None and now - summary_at > 120:
            reason = "exit hang after REPLAY_SUMMARY"
        elif done_at is not None and now - done_at > 900:
            reason = "exit hang after last demo"
        elif not seen_first and now - t0 > startup_s:
            reason = f"no start after {startup_s}s"
        elif seen_first and now - last_growth > stall_s:
            reason = f"log stalled {stall_s}s"
        elif now - t0 > total_s:
            reason = f"timeout {total_s}s"
        if reason:
            log(f"KILL {cmd[2] if len(cmd) > 2 else cmd}: {reason}")
            try:
                os.killpg(p.pid, 9)
            except ProcessLookupError:
                pass
            p.wait()
            return -9, done, reason
        time.sleep(5)


def demo_timing(status):
    """Per-demo frames/s from the replayer's REPLAY_DEMO records (frames, wall_s from the demo's scene restore to its last
    frame). The first demo of a process is reported apart (Kit start, scene load and the --warmup-s renders land there);
    the steady state is demo 2 on."""
    demos = status.get("demos") or []
    out = {}
    if demos:
        out["first_demo_s"] = round(demos[0][0] - status.get("t0", demos[0][0]), 1)
    rest = [d for d in demos[1:] if d[2] and d[3]]
    if rest:
        frames = sum(d[2] for d in rest)
        out.update({"demo_s": [d[3] for d in rest], "demo_fps": [round(d[2] / d[3], 2) for d in rest], "steady_frames": frames,
                    "steady_fps": round(frames / sum(d[3] for d in rest), 3)})
    return out


PURPOSE_FLAGS = {
    # owner 2026-11-15: training instances get replay-time scene perturbation (never camera relocalization: the replayer keeps its
    # collection-time removals) and replay-time lighting (the recorded collection lighting is ignored); eval instances neither
    "train": ["--scene-pert", "remove", "--scene-pert-plan", "auto", "--lighting-pert", "replay", "--lighting-skybox", "episode",
              "--pert-unit", "demo", "--scene-pert-inview-pick", "random",
              "--goal-occlusion", "auto"],
    "eval": ["--scene-pert", "off", "--lighting-pert", "off", "--goal-occlusion", "off"],
}


def run_unit(unit, gpu, slot, kk, cpus):
    uid = unit["unit"]
    udir = WORK / "units" / uid
    shutil.rmtree(udir, ignore_errors=True)
    udir.mkdir(parents=True)
    n_demos = len(unit["demos"])
    status = {"unit": uid, "gpu": gpu, "slot": slot, "n": n_demos, "done": 0, "frames": unit["frames"],
              "started": time.time(), "phase": "setup"}
    with lock:
        state["running"][slot] = status
    t0 = time.time()
    appdata = WORK / "appdata" / f"{uid}.{slot}"
    try:
        snap = ensure_snapshot(unit["snapshot"])
        # the unit's inputs at their absolute cluster paths (demo dirs, TROs and their sidecars, template.json / template_meta.json)
        r = subprocess.run(f"{' '.join(S5)} cat '{S3_URL}/{unit['inputs']}' | zstd -d -q | tar -x --no-same-owner -C /",
                           shell=True, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"inputs fetch failed: {r.stderr[-300:]}")
        s5("cp", f"{S3_URL}/{unit['meta']}", str(udir / "instance_meta.jsonl"))
        dirs = [d["dir"] for d in unit["demos"]]
        for d in dirs:
            if not os.path.isfile(os.path.join(d, "demo.hdf5")):
                raise RuntimeError(f"input demo missing: {d}")
        (udir / "demos.txt").write_text("\n".join(dirs) + "\n")
        status["phase"] = "textures"
        ensure_scene_textures("common")
        ensure_scene_textures(unit["scene"])
        warm, ntex = make_appdata(appdata)
        env = proc_env(snap, appdata, udir / "home", gpu, len(cpus))
        out = udir / "out"
        flags = RUNSPEC.get("replay_flags") or {}
        image_size = int(flags.get("image_size", 224))
        cmd = [f"{ENVP}/bin/python", "-u", "-m", "ispatialgym.multienv.replay", "--demos", f"@{udir / 'demos.txt'}", "--root", str(out),
               "--repo-prefix", uid, "--task", unit["task"], "--slots", "1", "--batched-usd-edits", "--async-flush",
               "--proprio", "loaded", "--warmup-s", str(WARMUP_S), "--scene", "full", "--image-size", str(image_size),
               "--settle-pad", str(flags.get("settle_pad", 0.2)), "--settle-pad-min", str(flags.get("settle_pad_min", 50)),
               "--goal-images", "on", "--instance-meta", str(udir / "instance_meta.jsonl"), "--results", str(out / "results.jsonl"),
               "--scene-pert-seed", str(RUNSPEC.get("seed", 20261115))] + PURPOSE_FLAGS[unit["purpose"]] + list(unit.get("extra_args") or [])
        logf = udir / "replay.log"
        status["phase"] = "replay"
        status["warm"] = warm
        log(f"START {uid} slot={slot} gpu={gpu} purpose={unit['purpose']} task={unit['task']} demos={n_demos} frames={unit['frames']} "
            f"warm={warm} tex_links={ntex}")
        max_len = max(int(d["frames"]) for d in unit["demos"])
        rc, done, reason = run_proc(cmd, env, str(snap), logf, status, n_demos, startup_s=2700 if not warm else 1500,
                                    stall_s=max(1800, max_len * 1.0 + 900), total_s=3600 + unit["frames"] * 2.0 + max_len * 0.5,
                                    cpus=cpus)
        elapsed = time.time() - t0
        if rc == -11 and done == 0 and elapsed < 300:
            with lock:
                state["early_segv"] = state.get("early_segv", 0) + 1
        ds = out / uid
        slot0 = out / f"{uid}_slot0"
        # og.shutdown() after REPLAY_SUMMARY can segfault in Kit's temp-dir cleanup; the writer has finalized the dataset by then
        summary = "summary" in status
        if done == n_demos and summary and slot0.is_dir():
            kept = [r for r in status.get("records", []) if r.get("kept")]
            eps = [json.loads(x) for x in (out / "episodes_isg.jsonl").read_text().splitlines() if x.strip()]
            kept_eps = sorted((e for e in eps if e.get("kept")), key=lambda e: e["episode_index"])
            slot0.rename(ds)
            (out / f"{uid}_slot0.format.json").rename(out / f"{uid}.format.json") if (out / f"{uid}_slot0.format.json").exists() else None
            try:
                stats = validate(ds, kept_eps, image_size)
                if len(kept_eps) != len(kept):
                    raise ValueError(f"{len(kept)} kept REPLAY_DEMO records, {len(kept_eps)} kept episode records")
                missing_goals = [e["episode_index"] for e in kept_eps if not (e.get("goal_image") and (out / e["goal_image"]["rgb_path"]).is_file())]
                if missing_goals:
                    raise ValueError(f"goal images missing for episodes {missing_goals[:5]}")
            except Exception as e:  # noqa: BLE001
                s5("cp", f"{out}/", f"{S3_URL}/suspect/{uid}/{INSTANCE}/", check=False)
                raise RuntimeError(f"validate: {e}") from e
            status["phase"] = "upload"
            shutil.copy2(logf, out / "replay.log")
            s5("cp", f"{out}/", f"{S3_URL}/out/{uid}/")
            timing = parse_timing(status.get("summary"))
            put_json(f"done/{uid}.json", {"unit": uid, "instance": INSTANCE, "tag": TAG, "gpu_name": GPUS[gpu]["name"],
                                          "driver": GPUS[gpu]["driver"], "slot": slot, "slots": len(SLOTS), "procs_per_gpu": PER_GPU,
                                          "seconds": elapsed, "warm": warm, "purpose": unit["purpose"], "task": unit["task"],
                                          "tex_links": ntex, "tex_new": new_textures(appdata), "scene": unit["scene"],
                                          "pin": PIN, "cpus": cpus, "quota": QUOTA, "visible_cpus": len(ALLOWED),
                                          "demos": status.get("demos"), **demo_timing(status), "summary": status.get("summary"),
                                          "kept": len(kept_eps), "dropped": n_demos - len(kept_eps),
                                          "replayer": "ispatialgym.multienv.replay", "code": unit["snapshot"], "rc": rc, "note": reason,
                                          **stats, **timing, "time": time.time()})
            with lock:
                state["done"].append(uid)
                state["frames_done"] += stats["frames"]
                state["successes"] += 1
                first = state["successes"] == 1
            dt = demo_timing(status)
            log(f"DONE {uid} {stats['frames']} frames in {elapsed:.0f}s ({stats['frames'] / elapsed:.2f} fps incl. startup), "
                f"{stats['bytes'] / 2**30:.2f} GB, first demo after {dt.get('first_demo_s')}s, steady {dt.get('steady_fps')} fps")
            ntex = save_textures(appdata)
            if first:
                save_template(appdata, kk, "unit")
            log(f"textures saved {ntex}")
            return True
        raise RuntimeError(f"rc={rc} demos {done}/{n_demos} summary={summary} {reason}")
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        log(f"FAIL {uid} slot={slot}: {msg}")
        logf = udir / "replay.log"
        with lock:
            state["failed"].append(uid)
            local_fails[uid] = local_fails.get(uid, 0) + 1
            if status.get("done", 0) == 0:
                state["startup_fails"] += 1
        try:
            n = fail_count(uid)
            body = f"instance {INSTANCE} slot {slot} gpu {GPUS[gpu]['name']} {GPUS[gpu]['driver']}\n{msg}\n{traceback.format_exc()}\n"
            (udir / "fail.txt").write_text(body)
            s5("cp", str(udir / "fail.txt"), f"{S3_URL}/fail/{uid}/{INSTANCE}-{n}.txt", check=False)
            with lock:
                state.setdefault("fail_keys", []).append((uid, f"fail/{uid}/{INSTANCE}-{n}.txt"))
                proven = state["successes"] > 0
            if logf.exists():
                s5("cp", str(logf), f"{S3_URL}/logs/units/{uid}/{INSTANCE}-{n}-replay.log", check=False)
            if (appdata / "local/logs").exists():
                s5("cp", f"{appdata}/local/logs/", f"{S3_URL}/logs/units/{uid}/{INSTANCE}-{n}-kitlogs/", check=False)
            if n + 1 >= MAX_UNIT_FAILS and proven:
                put_json(f"dead/{uid}.json", {"unit": uid, "fails": n + 1, "last": msg}, if_none_match=True)
            delete(f"claims/{uid}.json")
        except Exception as e2:  # noqa: BLE001
            log(f"fail bookkeeping for {uid} failed: {e2}")
        return False
    finally:
        with lock:
            state["running"].pop(slot, None)
        shutil.rmtree(appdata, ignore_errors=True)
        shutil.rmtree(udir / "out", ignore_errors=True)
        shutil.rmtree(udir / "home", ignore_errors=True)


def parse_timing(summary):
    """Load, warm-up and replay times of a finished replayer run (its REPLAY_SUMMARY)."""
    summary = summary or {}
    return {"startup_s": summary.get("load_s"), "warmup_s": (summary.get("warmup") or {}).get("s"),
            "replay_s": summary.get("replay_wall_s"), "env_steps_per_s": summary.get("env_steps_per_s")}


GPU_MODES = ("cuda_visible", "omnigibson", "none")


def warmup(kk, gpu, ncpu):
    """Empty-scene Kit start (5 steps) that builds the instance's template and checks Kit runs on this host. A start that
    crashes within 5 minutes is retried with the other GPU-selection modes; the first that works is used for every replay
    process. Each attempt's Kit logs go to $LOG/kitlogs_<mode>/."""
    global GPU_MODE
    snap = ensure_snapshot(sorted({u["snapshot"] for u in UNITS.values()})[0])
    code = ("import omnigibson as og\nenv = og.Environment(configs={'scene': {'type': 'Scene'}})\n"
            "for _ in range(5):\n    og.sim.step()\nprint('KIT_TEST_OK', flush=True)\nog.shutdown()\n")
    # a script file, never ``python -c``: Kit's crash reporter reads /proc/self/cmdline up to the first newline and then
    # segfaults counting its NUL-separated arguments (argz_count) when an argument holds a newline
    script = WORK / "warmup.py"
    script.write_text(code)
    modes = [GPU_MODE] + [m for m in GPU_MODES if m != GPU_MODE]
    for mode in modes:
        GPU_MODE = mode
        appdata = WORK / "appdata" / f"warmup_{mode}"
        make_appdata(appdata)
        env = proc_env(snap, appdata, WORK / f"warmup_home_{mode}", gpu, len(solo_cpus()))
        logf = LOG / f"warmup_{mode}.log"
        t0 = time.time()
        rc, done, reason = run_proc([f"{ENVP}/bin/python", "-u", str(script)], env, str(snap), logf, {}, 1,
                                    startup_s=2400, stall_s=1500, total_s=2700, cpus=solo_cpus())
        dt = time.time() - t0
        log(f"WARMUP mode={mode} rc={rc} ok={done} {reason} in {dt:.0f}s")
        if (appdata / "local/logs").exists():
            shutil.copytree(appdata / "local/logs", LOG / f"kitlogs_{mode}", dirs_exist_ok=True)
        if done >= 1:
            save_template(appdata, kk, "warm")
            shutil.rmtree(appdata, ignore_errors=True)
            return True
        s5("cp", f"{LOG}/*", f"{S3_URL}/logs/{INSTANCE}/", check=False)
        if dt > 300:
            break
    kit_diagnostics(snap, gpu, ncpu, code)
    return False


def kit_diagnostics(snap, gpu, ncpu, code):
    """After a failed warm-up: a bare Isaac Sim start, the OmniGibson start under LD_DEBUG=libs, and a gdb backtrace."""
    appdata = WORK / "appdata" / "diag"
    py = f"{ENVP}/bin/python"
    simapp = WORK / "diag_simapp.py"
    simapp.write_text("from isaacsim import SimulationApp\na = SimulationApp({'headless': True})\nprint('SIMAPP_OK', flush=True)\na.close()\n")
    script = WORK / "warmup.py"
    runs = [("simapp", [py, "-u", str(simapp)], {}), ("lddebug", [py, "-u", str(script)], {"LD_DEBUG": "libs"})]
    if shutil.which("gdb"):
        runs.append(("gdb", ["gdb", "-batch", "-ex", "run", "-ex", "bt 40", "-ex", "info threads", "-ex", "thread apply all bt 8",
                             "--args", py, "-u", str(script)], {}))
    for name, cmd, extra in runs:
        make_appdata(appdata)
        env = proc_env(snap, appdata, WORK / f"diag_home_{name}", gpu, len(solo_cpus()))
        env.update(extra)
        if PIN:
            cmd = ["taskset", "-c", ",".join(map(str, solo_cpus()))] + cmd
        out = LOG / f"diag_{name}.log"
        try:
            r = subprocess.run(cmd, env=env, cwd=str(snap), capture_output=True, text=True, timeout=600)
            text = r.stdout + r.stderr
            rc = r.returncode
        except subprocess.TimeoutExpired as e:
            text = f"TIMEOUT\n{e.stdout or ''}{e.stderr or ''}"
            rc = "timeout"
        lines = text.splitlines()
        out.write_text(f"rc={rc}\n" + "\n".join(lines[:200] + ["....."] + lines[-600:]) + "\n")
        log(f"DIAG {name} rc={rc}")
        if (appdata / "local/logs").exists():
            shutil.copytree(appdata / "local/logs", LOG / f"kitlogs_diag_{name}", dirs_exist_ok=True)
    s5("cp", f"{LOG}/*", f"{S3_URL}/logs/{INSTANCE}/", check=False)


#: (name, Kit settings, env changes) tried by diagnostics-only instances (ISG_DIAG_ONLY=1) on drivers whose first RTX
#: frame crashes Kit; an env value of None unsets the variable
DIAG_VARIANTS = [
    ("host_icd", [], {"VK_DRIVER_FILES": None, "VK_ICD_FILENAMES": None}),
    ("no_implicit_layers", [], {"VK_LOADER_LAYERS_DISABLE": "~implicit~"}),
    ("host_icd_no_layers", [], {"VK_DRIVER_FILES": None, "VK_ICD_FILENAMES": None, "VK_LOADER_LAYERS_DISABLE": "~implicit~"}),
    ("no_multigpu", ["--/renderer/multiGpu/enabled=false"], {}),
    ("aa_none", ["--/rtx/post/aa/op=0"], {}),
]


def diag_only(gpu):
    """Diagnostics instance: no units. The host's Vulkan ICD / layer files, a bare Isaac Sim start, a gdb backtrace of the
    OmniGibson start (crashing thread only, untruncated), then the OmniGibson start with each DIAG_VARIANTS change (Kit
    reads settings from argv; OmniGibson extends sys.argv); every log goes to R2."""
    snap = ensure_snapshot(sorted({u["snapshot"] for u in UNITS.values()})[0])
    py = f"{ENVP}/bin/python"
    info = []
    for d in ("/usr/share/vulkan/icd.d", "/etc/vulkan/icd.d", "/usr/share/vulkan/implicit_layer.d", "/etc/vulkan/implicit_layer.d",
              "/usr/share/glvnd/egl_vendor.d", "/usr/share/egl/egl_external_platform.d"):
        for f in sorted(Path(d).glob("*.json")) if Path(d).is_dir() else []:
            info.append(f"== {f}\n{f.read_text()[:1500]}")
    info.append("== env " + " ".join(f"{k}={v}" for k, v in sorted(os.environ.items()) if k.startswith(("VK_", "__GL", "__EGL", "NVIDIA_"))))
    (LOG / "diag_vulkan_files.txt").write_text("\n".join(info) + "\n")
    base = ("import omnigibson as og\nenv = og.Environment(configs={'scene': {'type': 'Scene'}})\n"
            "for _ in range(5):\n    og.sim.step()\nprint('KIT_TEST_OK', flush=True)\nog.shutdown()\n")
    (WORK / "warmup.py").write_text(base)
    (WORK / "diag_simapp.py").write_text("from isaacsim import SimulationApp\na = SimulationApp({'headless': True})\n"
                                         "for _ in range(10):\n    a.update()\nprint('SIMAPP_OK', flush=True)\na.close()\n")
    runs = [("simapp", [py, "-u", str(WORK / "diag_simapp.py")], {})]
    if shutil.which("gdb"):
        runs.append(("gdb", ["gdb", "-batch", "-ex", "set print thread-events off", "-ex", "set pagination off", "-ex", "run",
                             "-ex", "bt 60", "-ex", "x/8i $pc", "-ex", "info registers rip", "--args", py, "-u",
                             str(WORK / "warmup.py")], {}))
    for name, extra, envd in DIAG_VARIANTS:
        f = WORK / f"diag_{name}.py"
        f.write_text(f"import sys\nsys.argv += {extra!r}\n" + base)
        runs.append((name, [py, "-u", str(f)], envd))
    for name, cmd, envd in runs:
        appdata = WORK / "appdata" / f"diag_{name}"
        make_appdata(appdata)
        env = proc_env(snap, appdata, WORK / f"diag_home_{name}", gpu, len(solo_cpus()))
        for k, v in envd.items():
            if v is None:
                env.pop(k, None)
            else:
                env[k] = v
        if PIN:
            cmd = ["taskset", "-c", ",".join(map(str, solo_cpus()))] + cmd
        t0 = time.time()
        try:
            r = subprocess.run(cmd, env=env, cwd=str(snap), capture_output=True, text=True, timeout=900)
            text, rc = r.stdout + r.stderr, r.returncode
        except subprocess.TimeoutExpired as e:
            text, rc = f"TIMEOUT\n{e.stdout or ''}{e.stderr or ''}", "timeout"
        ok = "KIT_TEST_OK" in text or "SIMAPP_OK" in text
        lines = [ln for ln in text.splitlines() if "New Thread" not in ln and "exited with code" not in ln]
        keep = lines if name == "gdb" else lines[:100] + ["....."] + lines[-700:]
        (LOG / f"diag_{name}.log").write_text(f"rc={rc}\n" + "\n".join(keep) + "\n")
        if (appdata / "local/logs").exists():
            shutil.copytree(appdata / "local/logs", LOG / f"kitlogs_diag_{name}", dirs_exist_ok=True)
        shutil.rmtree(appdata, ignore_errors=True)
        log(f"DIAG {name} rc={rc} ok={ok} in {time.time() - t0:.0f}s")
        s5("cp", f"{LOG}/*", f"{S3_URL}/logs/{INSTANCE}/", check=False)


def utilization():
    """GPU util / memory per GPU, host CPU busy fraction since the last call, host memory used."""
    out = {}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
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


def proc_rss(pid):
    """Resident memory (GB) of a replay process and its children, and its anonymous part (RssAnon: the private memory; the
    rest is file-backed, e.g. mapped texture-store files shared by every process)."""
    try:
        import psutil
        p = psutil.Process(pid)
        procs = [p] + p.children(recursive=True)
        rss = sum(x.memory_info().rss for x in procs)
        anon = 0
        for x in procs:
            m = re.search(r"RssAnon:\s+(\d+) kB", open(f"/proc/{x.pid}/status").read())
            anon += int(m.group(1)) * 1024 if m else 0
        return round(rss / 2**30, 2), round(anon / 2**30, 2)
    except Exception:  # noqa: BLE001
        return None, None


def cgroup_mem():
    """The container's memory (GB): memory.current and memory.stat's anon / file; memory.max."""
    out = {}
    try:
        out["cg_mem_gb"] = round(int(open("/sys/fs/cgroup/memory.current").read()) / 2**30, 1)
        for line in open("/sys/fs/cgroup/memory.stat"):
            k, v = line.split()
            if k in ("anon", "file", "shmem"):
                out[f"cg_{k}_gb"] = round(int(v) / 2**30, 1)
        mx = open("/sys/fs/cgroup/memory.max").read().strip()
        out["cg_max_gb"] = None if mx == "max" else round(int(mx) / 2**30, 1)
    except (OSError, ValueError):
        pass
    return out


def running_view(r):
    out = {k: v for k, v in r.items() if k not in ("demos", "t0", "pid")}
    out.update(demo_timing(r))
    for k in ("demo_s", "demo_fps"):
        if k in out:
            out[k] = out[k][-12:]
    if r.get("pid"):
        out["rss_gb"], out["anon_gb"] = proc_rss(r["pid"])
    return out


def heartbeat_loop(kk):
    n = 0
    while True:
        util = utilization()
        with lock:
            running = [running_view(r) for r in state["running"].values()]
            hb = {"instance": INSTANCE, "tag": TAG, "phase": state["phase"], "time": time.time(), "uptime": time.time() - t_start,
                  "gpus": [g["name"] for g in GPUS], "driver": GPUS[0]["driver"], "cpus": NCPU, "mem_gb": round(MEM, 1),
                  "slots": SLOTS, "procs_per_gpu": PER_GPU, "run": RUN, "kit_key": kk, "template": template_ready(), "running": running,
                  "pin": PIN, "visible_cpus": len(ALLOWED), "quota": QUOTA, "Q": round(Q, 2),
                  "cpus_per_slot": len(next(iter(SLOT_CPUS.values()))) if SLOT_CPUS else None, "cpu_stat": cpu_stat(),
                  **cgroup_mem(),
                  "done": len(state["done"]), "failed": len(state["failed"]), "frames_done": state["frames_done"],
                  "load1": os.getloadavg()[0], "disk_free_gb": shutil.disk_usage("/work").free / 2**30, **util}
        try:
            put_json(f"hb/{INSTANCE}.json", hb)
            if n % 5 == 0:
                s5("cp", f"{LOG}/*", f"{S3_URL}/logs/{INSTANCE}/", check=False)
            if n % 3 == 0:  # the tail of every running unit's replay log (Kit's inotify noise left out)
                for r in hb["running"]:
                    lf = WORK / "units" / r["unit"] / "replay.log"
                    if lf.exists():
                        lines = [x for x in lf.read_text(errors="replace").splitlines() if "change watch" not in x]
                        tail = LOG / "running" / f"{r['unit']}.log"
                        tail.parent.mkdir(exist_ok=True)
                        tail.write_text("\n".join(lines[-80:]) + "\n")
                s5("cp", f"{LOG}/running/*", f"{S3_URL}/logs/{INSTANCE}/running/", check=False)
        except Exception as e:  # noqa: BLE001
            log(f"heartbeat failed: {e}")
        n += 1
        if state["phase"] in ("done", "failed"):
            return
        time.sleep(60)


def withdraw_fails():
    """An instance that never finished a unit failed because of its host: its fail records must not count against the
    units (MAX_UNIT_FAILS make a unit dead); the logs stay under logs/units/."""
    with lock:
        keys, state["fail_keys"] = state.get("fail_keys", []), []
    for _, k in keys:
        delete(k)
    if keys:
        log(f"withdrew {len(keys)} fail records (host problem): {sorted({u for u, _ in keys})}")


def slot_loop(slot, gpu, kk, cpus, max_units=None):
    time.sleep(STAGGER_S * slot)
    n = 0
    while not stop_claiming.is_set() and (max_units is None or n < max_units):
        unit = claim_next()
        if unit is None:
            break
        run_unit(unit, gpu, slot, kk, cpus)
        n += 1
        with lock:
            bad = state["successes"] == 0 and state["startup_fails"] >= max(2, len(SLOTS) // 2)
            segv = state["successes"] == 0 and state.get("early_segv", 0) >= 2
            if segv:
                state["diagnose"] = True
        if bad or segv:
            log("instance unusable: replays fail at startup and none succeeded" + (" (segfaults: diagnostics follow)" if segv else ""))
            stop_claiming.set()
            withdraw_fails()
    log(f"slot {slot} exits")


def main():
    global GPUS, NCPU, MEM, SLOTS, SLOT_CPUS, PER_GPU
    GPUS = gpus()
    NCPU, MEM = cpu_count(), mem_gb()
    # a replay process holds ~10 GB of VRAM, ~3 busy cores and ~11 GB of RAM (measured on RTX 3090s)
    # 2026-11-15 sweep (RTX 5090, 224 px): 4 processes per 32 GB GPU gave the most frames/s (VRAM peak 30.5 GB); 24 GB GPUs run 2
    vram = min(g["mem_mb"] for g in GPUS)
    # evalprod2 (2026-11-15): 4 per 32 GB GPU hit CUDA OOM at Kit start on 2 of ~12 RTX 5090 hosts; 3 gives 39.3 vs 39.8 fps/GPU
    by_vram = 3 if vram >= 30000 else 2 if vram >= 20000 else 1
    # a replay process keeps ~3.5 cores busy (REPLAY_SUMMARY cpu_cores_busy): at most quota / 4 processes per GPU
    per_gpu = int(os.environ.get("ISG_PROCS_PER_GPU", "0")) or max(1, min(by_vram, NCPU // len(GPUS) // 4, int(MEM / len(GPUS) // 12)))
    PER_GPU = per_gpu
    SLOTS = [(i * per_gpu + j, g["index"]) for j in range(per_gpu) for i, g in enumerate(GPUS)]
    SLOTS.sort()
    ncpu = max(1, NCPU // len(SLOTS))
    SLOT_CPUS = {s: slot_cpus(k, len(SLOTS)) for k, (s, g) in enumerate(SLOTS)}
    log(f"CPUS visible {len(ALLOWED)} quota {QUOTA} usable Q {Q:.1f} pin {PIN}: {len(SLOT_CPUS[SLOTS[0][0]])} CPUs per slot, "
        f"{len(solo_cpus())} for a solo process; replayer warm-up {WARMUP_S}s")
    for s_, g_ in SLOTS:
        log(f"SLOT {s_} gpu {g_} cpus {','.join(map(str, SLOT_CPUS[s_]))}")
    kk = kit_key(GPUS[0])
    log(f"WORKER instance={INSTANCE} gpus={[g['name'] for g in GPUS]} driver={GPUS[0]['driver']} cpus={NCPU} mem={MEM:.0f}GB "
        f"procs/gpu={per_gpu} slots={len(SLOTS)} units={len(UNITS)} kit_key={kk}")
    hb = threading.Thread(target=heartbeat_loop, args=(kk,), daemon=True)
    state["phase"] = "template"
    hb.start()
    (WORK / "cuda_cache").mkdir(exist_ok=True)
    if os.environ.get("ISG_DIAG_ONLY") == "1":
        state["phase"] = "diag"
        diag_only(GPUS[0]["index"])
        state["phase"] = "failed"
        time.sleep(70)
        return 3
    if not fetch_template(kk):
        log("no Kit template for this GPU + driver on R2: every process starts cold, in parallel with its scene load")
    log(f"GPU_MODE {GPU_MODE}")
    state["phase"] = "run"
    sweep = [int(x) for x in os.environ.get("ISG_SWEEP", "").split(",") if x.strip()]
    if sweep:
        return run_sweep(sweep, kk)
    threads = [threading.Thread(target=slot_loop, args=(s, g, kk, SLOT_CPUS[s])) for s, g in SLOTS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if state.get("diagnose"):
        log("units segfault at startup: running the Kit warm-up and diagnostics")
        warmup(kk, GPUS[0]["index"], ncpu)
        state["phase"] = "failed"
        s5("cp", f"{LOG}/*", f"{S3_URL}/logs/{INSTANCE}/", check=False)
        time.sleep(70)
        return 3
    state["phase"] = "done" if state["successes"] or not state["failed"] else "failed"
    log(f"WORKER END done={len(state['done'])} failed={len(state['failed'])} frames={state['frames_done']}")
    time.sleep(70)
    return 0 if state["phase"] == "done" else 3


def run_sweep(sweep, kk):
    """ISG_SWEEP=1,2,3,4 (processes per GPU, one phase each, on GPU 0): every phase runs P replay processes at once, each one unit
    (the sweep run's units are copies of one workload), and records runs/<run>/sweep/<instance>_p<P>.json: phase wall, frames, GPU
    frames/s (all processes, incl. load), per-process steady fps (REPLAY_SUMMARY env_steps_per_s), GPU memory peak."""
    global SLOTS, SLOT_CPUS, PER_GPU
    g = GPUS[0]["index"]
    for P in sweep:
        PER_GPU = P
        SLOTS = [(k, g) for k in range(P)]
        SLOT_CPUS = {k: slot_cpus(k, P) for k in range(P)}
        with lock:
            done0, frames0 = len(state["done"]), state["frames_done"]
            state["sweep_phase"] = P
        peak = {"gpu_mem_mb": 0, "gpu_util": []}
        stop = threading.Event()

        def sample():
            while not stop.is_set():
                u = utilization()
                if u.get("gpu_mem_mb"):
                    peak["gpu_mem_mb"] = max(peak["gpu_mem_mb"], u["gpu_mem_mb"][0])
                    peak["gpu_util"].append(u["gpu_util"][0])
                time.sleep(10)
        th_s = threading.Thread(target=sample, daemon=True)
        th_s.start()
        t0 = time.time()
        threads = [threading.Thread(target=slot_loop, args=(k, g, kk, SLOT_CPUS[k], 1)) for k in range(P)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        wall = time.time() - t0
        stop.set()
        recs = []
        for uid in state["done"][done0:]:
            d = get_json(f"done/{uid}.json") or {}
            sm = d.get("summary") or {}
            recs.append({"unit": uid, "frames": d.get("frames"), "seconds": d.get("seconds"), "load_s": sm.get("load_s"),
                         "replay_wall_s": sm.get("replay_wall_s"), "env_steps_per_s": sm.get("env_steps_per_s"),
                         "ms_per_tick": sm.get("ms_per_tick"), "gpu_mib_peak": sm.get("gpu_mib_peak"), "cpu_cores_busy": sm.get("cpu_cores_busy")})
        frames = state["frames_done"] - frames0
        util = peak["gpu_util"]
        out = {"instance": INSTANCE, "gpu": GPUS[0]["name"], "driver": GPUS[0]["driver"], "procs": P, "units": len(recs),
               "failed_units": P - len(recs), "wall_s": round(wall, 1), "frames": frames, "gpu_fps_incl_load": round(frames / wall, 2),
               "proc_fps": [r["env_steps_per_s"] for r in recs],
               "gpu_fps_steady": round(sum(r["env_steps_per_s"] or 0 for r in recs), 2),
               "gpu_mem_peak_mb": peak["gpu_mem_mb"], "gpu_util_mean": round(sum(util) / len(util), 1) if util else None,
               "Q": Q, "cpus_per_proc": len(SLOT_CPUS[0]), "mem_gb": round(MEM, 1), "records": recs, "time": time.time()}
        put_json(f"sweep/{INSTANCE}_p{P}.json", out)
        log(f"SWEEP P={P}: {len(recs)}/{P} units, {frames} frames in {wall:.0f}s = {out['gpu_fps_incl_load']} fps/GPU incl. load; "
            f"steady {out['proc_fps']} -> {out['gpu_fps_steady']} fps/GPU; mem peak {peak['gpu_mem_mb']} MB")
    state["phase"] = "done" if state["successes"] else "failed"
    time.sleep(70)
    return 0 if state["successes"] else 3


if __name__ == "__main__":
    sys.exit(main())
