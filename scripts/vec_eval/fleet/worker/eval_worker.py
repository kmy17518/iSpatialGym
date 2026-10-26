"""Vectorized-eval worker on one rented GPU instance (started by boot.sh once the bundle is unpacked).

Claims units (``units.json``: one per (run, task) - that task's eval instance + val instances, every checkpoint of the run,
latest first) through conditional puts of ``claims/<unit>.json``, runs each with ``ispatialgym.oracle.run_isg_vec --mode policy
--policy-sweep`` (one world per unit; per checkpoint an ACT server started by the driver on this GPU), uploads every finished
checkpoint's outputs to ``out/<unit>/<tag>/`` as they complete and writes ``done/<unit>.json``. Exits 0 when nothing is left
to claim (boot.sh then destroys the instance), 3 when the instance looks unusable.

One process per GPU (``ISG_NUM_ENVS`` clones, default 11 = a task's instances), one Kit app-data folder per process seeded
from the instance's template (the replay worker's rules: kit_tpl/<GPU + driver + Behavior-1K commit>/ on the bundle prefix,
the texture store linked into every process's texture cache).
"""

import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path

import boto3
import botocore

S3_URL = os.environ["ISG_S3"]                      # the run prefix (units, claims, out, done, hb, logs)
BUNDLE_URL = os.environ.get("ISG_BUNDLE_S3", "s3://isg-demos/isg-1020-b1k")  # the replay bundle (env, b1k code, assets, textures, kit templates)
BUCKET, PREFIX = S3_URL[len("s3://"):].split("/", 1)
INSTANCE = os.environ["ISG_INSTANCE"]
TAG = os.environ.get("ISG_TAG", "")
ENVP = os.environ["ENVP"]
# the roots: the Vast image's defaults; ISG_* overrides let the same worker run on a workstation (the lambda 4090 box)
WORK = Path(os.environ.get("ISG_WORK", "/work"))
LOG = Path(os.environ.get("ISG_LOG", "/var/log/isg"))
B1K_COMMIT = "6a6194600"
S5 = ["s5cmd", "--endpoint-url", os.environ["S3_ENDPOINT_URL"], "--numworkers", "32"]
NUM_ENVS = int(os.environ.get("ISG_NUM_ENVS", "11"))
MAX_UNIT_FAILS = 3
#: the eval bundle's roots (absolute cluster paths recreated on the instance)
EVAL_ROOT = Path(os.environ.get("ISG_EVAL_ROOT", "/vision/u/minyeong/projects/ispatialgym-eval-vec"))
VEC_ROOT = Path(os.environ.get("ISG_VEC_ROOT", "/vision/u/minyeong/projects/isg_home/vec_eval85"))
CODE_ROOTS = {"act-todo1": Path(os.environ.get("ISG_CODE_ACT_TODO1", "/vision/u/minyeong/projects/act-todo1")),
              "act-side": Path(os.environ.get("ISG_CODE_ACT_SIDE", "/vision/u/minyeong/projects/act-side")),
              # runs with `image_augment` in their model config (relq-piaug): the checkout that knows the key (campaign c2)
              "act-piaug": Path(os.environ.get("ISG_CODE_ACT_PIAUG", "/vision/u/minyeong/projects/act-piaug")),
              # Diffusion Policy (kmy17518/diffusion_policy isg-todo1-goal): units of kind "dp"; served by its own venv
              "dp-isg": Path(os.environ.get("ISG_CODE_DP_ISG", "/vision/u/minyeong/projects/diffusion_policy"))}
DP_PYTHON = os.environ.get("ISG_DP_PYTHON", str(CODE_ROOTS["dp-isg"] / ".venv/bin/python"))  # DP's serving interpreter
B1K_DIR = os.environ.get("ISG_B1K_DIR", "/work/b1k")  # the Behavior-1K checkout the overlay is applied to
TEMPLATE_EXCLUDE = ("local/logs", "tmp", "local/data/documents/Kit/shared/screenshots", "global/cache/texturecache")

s3 = boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"], region_name="auto",
                  config=botocore.config.Config(retries={"max_attempts": 10, "mode": "adaptive"}))
t_start = time.time()
lock = threading.Lock()
stop_claiming = threading.Event()
state = {"phase": "start", "running": {}, "done": [], "failed": [], "episodes_done": 0, "successes": 0, "startup_fails": 0}


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
    try:
        q, p = open("/sys/fs/cgroup/cpu.max").read().split()
        return None if q == "max" else int(q) / int(p)
    except (OSError, ValueError):
        return None


QUOTA = cpu_quota()
Q = min(len(ALLOWED), QUOTA) if QUOTA else float(len(ALLOWED))


def slot_cpus(k, n_slots):
    c = max(2, int((Q - 1) // n_slots))
    return [ALLOWED[(k * c + i) % len(ALLOWED)] for i in range(c)]


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


# ---------------------------------------------------------------- Kit app data: template + texture store (the replay worker's)
TPL = WORK / "kit_template"
TEX = WORK / "texstore"
tpl_lock = threading.Lock()


def template_ready():
    return (TPL / ".complete").exists()


def bundle_exists(k):
    r = subprocess.run(S5 + ["ls", f"{BUNDLE_URL}/{k}"], capture_output=True, text=True)
    return r.returncode == 0 and bool(r.stdout.strip())


def fetch_template(kk):
    if (TPL / ".complete").exists():  # a workstation worker restarted with its template in place
        log("KIT_TEMPLATE already present")
        return "local"
    shutil.rmtree(TPL, ignore_errors=True)
    for k in (f"kit_tpl/{kk}/unit.tar.zst", f"kit_tpl/{kk}/warm.tar.zst", f"bundle/kit_templates/{kk}.tar.zst"):
        if bundle_exists(k):
            tmp = WORK / "kit_template.dl"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            r = subprocess.run(f"{' '.join(S5)} cat '{BUNDLE_URL}/{k}' | zstd -d -q | tar -x --no-same-owner -C {tmp}", shell=True)
            if r.returncode == 0:
                (tmp / ".complete").touch()
                tmp.rename(TPL)
                log(f"KIT_TEMPLATE fetched {k}")
                return k
            shutil.rmtree(tmp, ignore_errors=True)
    return None


def save_template(appdata, kk):
    """This instance's template from a finished process's app data (not uploaded: the bundle prefix's templates are the replay's)."""
    with tpl_lock:
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
        old = WORK / "kit_template.old"
        shutil.rmtree(old, ignore_errors=True)
        if TPL.exists():
            TPL.rename(old)
        tmp.rename(TPL)
        shutil.rmtree(old, ignore_errors=True)
        log(f"KIT_TEMPLATE saved from {appdata.name}")
    k = f"kit_tpl/{kk}/unit.tar.zst"
    if not bundle_exists(k):
        r = subprocess.run(f"tar -C {TPL} --exclude=./.complete -cf - . | zstd -q -T4 -3 | {' '.join(S5)} pipe '{BUNDLE_URL}/{k}'", shell=True, capture_output=True, text=True)
        log(f"KIT_TEMPLATE upload {k}: rc={r.returncode}")


def rm_tree(path):
    """rmtree that also removes Kit's mode-0 folders (local/data/documents/Kit/shared/screenshots): as a non-root user
    (the cluster) rmtree(ignore_errors=True) silently left them behind and the next mkdir failed with FileExistsError."""
    if not os.path.lexists(path):
        return
    subprocess.run(["chmod", "-R", "u+rwX", str(path)], capture_output=True)
    shutil.rmtree(path, ignore_errors=True)
    if os.path.lexists(path):
        subprocess.run(["rm", "-rf", str(path)], capture_output=True)


def make_appdata(path):
    rm_tree(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if template_ready():
        with tpl_lock:
            path.mkdir(parents=True, exist_ok=True)
            subprocess.run(["cp", "-a", f"{TPL}/.", str(path)], check=True)
        (path / ".complete").unlink(missing_ok=True)
        warm = True
    else:
        path.mkdir(parents=True, exist_ok=True)
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
    marker = TEX / ".scenes" / f"{scene}.complete"
    with lock:
        sl = scene_locks.setdefault(scene, threading.Lock())
    with sl:
        if marker.exists():
            return True
        r = subprocess.run(S5 + ["cp", f"{BUNDLE_URL}/bundle/textures/{scene}/parts.txt", str(WORK / f"tex_{scene}.parts")], capture_output=True, text=True)
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
                r = subprocess.run(["timeout", "300"] + S5 + ["--retry-count", "20", "cp", "-c", "6", "-p", "64", f"{BUNDLE_URL}/bundle/textures/{scene}/{part}", str(f)],
                                   capture_output=True)
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


def save_textures(appdata):
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
                except OSError:
                    # the store on another filesystem (the cluster's NFS store linked into ISG_WORK): copy under a temporary
                    # name, then rename into place, so the store only ever shows complete files
                    tmp = TEX / f".{name}.{os.getpid()}.tmp"
                    shutil.copyfile(tex / name, tmp)
                    os.chmod(tmp, 0o444)
                    os.replace(tmp, TEX / name)
        n += 1
    return n


# ---------------------------------------------------------------- openpi (pi05) serving environment
OPENPI_DIR = Path(os.environ.get("ISG_OPENPI_DIR", "/vision/u/minyeong/projects/openpi-todo1"))
PI_VENV = Path(os.environ.get("ISG_PI_VENV", "/work/pi/venv"))
PI_PY = Path("/work/pi/uv-python")
pi_lock = threading.Lock()
# cluster workers (campaign c2): the extracted NFS steps `<ISG_PI_CKPT_ROOT>/<run>/step-XXXXXXXX/` instead of pi_serve.sh's
# HF download into /work/pi, pi_serve.sh's scratch under ISG_WORK, the NFS venv taken as ready, the NFS openpi data dir
PI_CKPT_ROOT = os.environ.get("ISG_PI_CKPT_ROOT", "")
PI_WORK = Path(os.environ.get("ISG_PI_WORK", "/work/pi"))
PI_VENV_READY = os.environ.get("ISG_PI_VENV_READY") == "1"


PI_DATA = Path(os.environ.get("ISG_OPENPI_DATA") or (WORK / "pi" / "openpi_data"))


def ensure_openpi_data():
    """openpi's data cache (OPENPI_DATA_HOME) pre-seeded with the PaliGemma tokenizer: the server would fetch it from
    storage.googleapis.com, where the venv's python fails the TLS certificate check (every pi05 server died that way)."""
    tok = PI_DATA / "big_vision" / "paligemma_tokenizer.model"
    if tok.exists() and tok.stat().st_size > 4_000_000:
        return
    tok.parent.mkdir(parents=True, exist_ok=True)
    r = s5("cp", f"{S3_URL}/bundle/openpi_data/big_vision/paligemma_tokenizer.model", str(tok), check=False)
    log(f"OPENPI tokenizer {'fetched' if r.returncode == 0 else 'fetch FAILED: ' + r.stderr.strip()[-120:]}")


def ensure_openpi_venv():
    """The openpi uv environment at /work/pi/venv (+ uv's CPython under /work/pi/uv-python): fetched from the run prefix's
    bundle/openpi_venv.tar.zst when it exists, else built with `uv sync --frozen` in the checkout and uploaded for the others."""
    with pi_lock:
        ensure_openpi_data()
        if (PI_VENV / ".ready").exists() or (PI_VENV_READY and (PI_VENV / "bin/python").exists()):
            return PI_VENV
        t0 = time.time()
        Path("/work/pi").mkdir(parents=True, exist_ok=True)
        r = subprocess.run(S5 + ["ls", f"{S3_URL}/bundle/openpi_venv.tar.zst"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            r = subprocess.run(f"{' '.join(S5)} cat '{S3_URL}/bundle/openpi_venv.tar.zst' | zstd -d -q | tar -x --no-same-owner -C /", shell=True, capture_output=True, text=True)
            if r.returncode == 0 and (PI_VENV / "bin/python").exists():
                (PI_VENV / ".ready").touch()
                log(f"OPENPI venv fetched in {time.time() - t0:.0f}s")
                return PI_VENV
            log(f"OPENPI venv fetch failed ({r.stderr[-200:]}); building")
        env = dict(os.environ, PATH=f"{ENVP}/bin:" + os.environ.get("PATH", ""), UV_PYTHON_INSTALL_DIR=str(PI_PY), UV_CACHE_DIR="/work/pi/uv-cache",
                   UV_PROJECT_ENVIRONMENT=str(PI_VENV), UV_LINK_MODE="copy", HOME="/work/pi/home")
        Path("/work/pi/home").mkdir(exist_ok=True)
        if not shutil.which("uv", path=env["PATH"]):
            subprocess.run([f"{ENVP}/bin/pip", "install", "-q", "uv"], check=True, capture_output=True)
        r = subprocess.run(["uv", "sync", "--frozen", "--no-dev"], cwd=str(OPENPI_DIR), env=env, capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(f"uv sync failed: {r.stderr[-800:]}")
        (PI_VENV / ".ready").touch()
        log(f"OPENPI venv built in {time.time() - t0:.0f}s")
        # the tarball for the other instances (venv + uv's python; absolute paths under /work/pi)
        r = subprocess.run(f"tar -C / -cf - work/pi/venv work/pi/uv-python | zstd -q -T4 -3 | {' '.join(S5)} pipe '{S3_URL}/bundle/openpi_venv.tar.zst'", shell=True,
                           capture_output=True, text=True)
        log(f"OPENPI venv upload rc={r.returncode}")
        shutil.rmtree("/work/pi/uv-cache", ignore_errors=True)
        return PI_VENV


# ---------------------------------------------------------------- units
UNITS = {u["unit"]: u for u in json.load(open(WORK / "units.json"))["units"]}
UNITS_SHA = {"value": hashlib.sha1((WORK / "units.json").read_bytes()).hexdigest()}


def refresh_units():
    """``units.json`` of the run prefix when it differs from the loaded one (a re-split / re-prioritization while this worker
    runs): the new units become claimable at the next claim; running units are unaffected."""
    try:
        r = s5("cat", f"{S3_URL}/units.json", check=False)
        if r.returncode != 0 or not r.stdout.strip():
            return
        sha = hashlib.sha1(r.stdout.encode()).hexdigest()
        if sha == UNITS_SHA["value"]:
            return
        units = {u["unit"]: u for u in json.loads(r.stdout)["units"]}
        (WORK / "units.json").write_text(r.stdout)
        UNITS.clear()
        UNITS.update(units)
        UNITS_SHA["value"] = sha
        log(f"units refreshed: {len(units)} units ({sha[:8]})")
    except Exception as e:  # noqa: BLE001
        log(f"units refresh failed: {e}")
PREFER = [u for u in os.environ.get("ISG_PREFER", "").split(",") if u]
ONLY_RUNS = [u for u in os.environ.get("ISG_RUNS", "").split(",") if u]
local_fails = {}
scenes_here = set()
# a SLURM worker's allocation end (epoch seconds; 0 = none): a unit is claimed only when its rough wall-time estimate fits
# before it, so a job's last unit is not killed half-way by the time limit (its finished episodes are only uploaded per tag)
DEADLINE = float(os.environ.get("ISG_DEADLINE", "0") or 0)


def unit_eta_s(u, n):
    """Rough wall time of a unit at n clones on a cluster GPU (2026-10-24: world build ~110 s + ~240 s per clone, episodes
    at ~3 env-steps/s per slot; a batch runs to its longest budget), with a 1.3x margin."""
    import math
    n = max(1, int(n))
    build = 110 + 240 * n
    batches = math.ceil(int(u.get("n_instances") or 11) / n)
    steps = 0.85 * int(u.get("max_budget_steps") or 1000) * batches * len(u.get("steps") or [0])
    return 1.3 * (build + steps / 3.0 + (len(u.get("steps") or [0]) - 1) * 240)


VERIFIED = {}


def verification(task):
    """``verify/<task>.json`` of the run prefix (cached once it passes): {"pass", "num_envs", ...} or None."""
    v = VERIFIED.get(task)
    if v is not None and v.get("pass"):
        return v
    v = get_json(f"verify/{task}.json")
    if v is not None:
        VERIFIED[task] = v
    return v


def claim_next():
    """Claim the next unit, or return "wait" when nothing is claimable yet but policy units are still waiting for their task's
    replay verification (another instance is running it). Order: the replay-verification units (priority 0, longest first),
    then the policy units whose task passed (longest first; a scene this instance already has textures for among the longest)."""
    if stop_claiming.is_set():
        return None
    with lock:
        if exists("ctl/stop") or exists(f"ctl/stop_{INSTANCE}"):
            log("CTL stop: no new claims")
            stop_claiming.set()
            return None
        refresh_units()
        done = set(n[:-5] for n in list_names("done/"))
        taken = done | set(n[:-5] for n in list_names("claims/")) | set(n[:-5] for n in list_names("dead/"))
        release_all = exists("ctl/release_all")
        candidates = [u for u in UNITS.values() if u["unit"] not in taken and local_fails.get(u["unit"], 0) < 2 and (not ONLY_RUNS or u["run"] in ONLY_RUNS)
                      and (u["kind"] in ("act", "dp", "replay_verify") or (PI_OK and os.environ.get("ISG_NO_PI") != "1"))]
        if not candidates:
            return None
        verify_open = [u for u in candidates if u["kind"] == "replay_verify"]
        policy_open = []
        waiting = 0
        too_long = False
        for u in candidates:
            if u["kind"] == "replay_verify":
                continue
            v = verification(u.get("requires_verify") or u["task"]) if not release_all else {"pass": True, "num_envs": NUM_ENVS}
            if not (v and v.get("pass")):
                waiting += 1
                continue
            gates = u.get("requires_hf_step")
            if gates and not all(hf_has_step(u["hf_repo"], int(s)) for s in (gates if isinstance(gates, list) else [gates])):
                waiting += 1  # a pi05 step not uploaded yet (training still running): claimable once it lands (a list: all of them)
                continue
            n_here = min(NUM_ENVS, int(v.get("num_envs") or NUM_ENVS), 6 if GPUS[0]["mem_mb"] < 30000 else 11, int(u.get("max_num_envs") or 99))
            if DEADLINE and time.time() + unit_eta_s(u, n_here) > DEADLINE:
                too_long = True  # would not finish before this allocation ends: left for a fresher worker
                continue
            policy_open.append(dict(u, verified_num_envs=int(v.get("num_envs") or NUM_ENVS)))
        for pool in (verify_open, policy_open):
            if not pool:
                continue
            # wave priority (1: the latest checkpoint of every pair; 2.x: older chunks, latest first) before anything else; the
            # >= 40 GB cards are the only ones that can serve pi05: they take the openpi units of a wave first; then longest first
            pool.sort(key=lambda u: (int(u.get("wave") or 9), float(u.get("priority", 1)), 0 if (PI_OK and u["kind"] == "openpi") else 1, -u["frames"]))
            head = pool[:8]
            local = [u for u in head if u["scene"] in scenes_here]
            # a preferred unit keeps the pool entry's verified N (it used to be given NUM_ENVS, above the task's verified clone count)
            order = [x for p in PREFER for x in pool if x["unit"] == p] + local + pool
            seen = set()
            for u in order:
                if u["unit"] in seen:
                    continue
                seen.add(u["unit"])
                if put_json(f"claims/{u['unit']}.json", {"instance": INSTANCE, "tag": TAG, "time": time.time(), "kind": u["kind"]}, if_none_match=True):
                    scenes_here.add(u["scene"])
                    return u
        if too_long and not waiting:
            log(f"nothing claimable fits before the allocation ends in {(DEADLINE - time.time()) / 60:.0f} min; exiting")
            return None
        return "wait" if waiting else None


def fail_count(uid):
    return len(list_names(f"fail/{uid}/"))


# ---------------------------------------------------------------- checkpoints (Hugging Face, public repos)
HF_META = {}
HF_META_TIME = {}


def hf_meta(repo):
    if repo not in HF_META:
        url = f"https://huggingface.co/{repo}/resolve/main/.b1k-uploader.json"
        with urllib.request.urlopen(url, timeout=60) as r:
            HF_META[repo] = json.loads(r.read())
    return HF_META[repo]


HF_SEEN = {}


def hf_has_step(repo, step):
    """The repo's uploader record lists an eval checkpoint at ``step`` (re-read every 10 min while it is missing)."""
    key = (repo, int(step))
    if HF_SEEN.get(key):
        return True
    if time.time() - HF_META_TIME.get(repo, 0) > 600:
        try:
            url = f"https://huggingface.co/{repo}/resolve/main/.b1k-uploader.json"
            with urllib.request.urlopen(url, timeout=60) as r:
                HF_META[repo] = json.loads(r.read())
            HF_META_TIME[repo] = time.time()
        except Exception as e:  # noqa: BLE001
            log(f"hf meta {repo}: {e}")
            HF_META_TIME[repo] = time.time() - 300
    ok = str(int(step)) in ((HF_META.get(repo) or {}).get("eval") or {})
    if ok:
        HF_SEEN[key] = True
    return ok


def sha256_of(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_checkpoint(repo, step, dest):
    """``eval/step-XXXXXXXX.pt`` of the repo into ``dest`` (curl, resumable), its sha256 checked against the uploader record."""
    meta = hf_meta(repo)
    rec = (meta.get("eval") or {}).get(str(step))
    name = rec["path"] if rec else f"eval/step-{step:08d}.pt"
    dest.parent.mkdir(parents=True, exist_ok=True)

    def verified():
        return dest.exists() and (not rec or sha256_of(dest) == rec["sha256"])

    if verified():
        return dest
    url = f"https://huggingface.co/{repo}/resolve/main/{name}"
    part = dest.with_name(dest.name + ".part")  # dest appears only complete and verified (a sweep's server waits for it)
    # the slots of one instance share the checkpoint dir: one downloader at a time (two slots curl-ing the same .part
    # corrupted it and the loser's os.replace found it gone, 2026-10-25)
    with open(dest.with_name(dest.name + ".lock"), "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if verified():
            return dest
        for attempt in range(6):
            r = subprocess.run(["curl", "-sSL", "--retry", "5", "--retry-delay", "5", "-C", "-", "-o", str(part), url], capture_output=True, text=True)
            if r.returncode == 0 and part.exists() and (not rec or sha256_of(part) == rec["sha256"]):
                os.replace(part, dest)
                return dest
            log(f"checkpoint {repo} {name}: attempt {attempt + 1} failed ({r.stderr.strip()[-200:]})")
            part.unlink(missing_ok=True)
            time.sleep(10 * (attempt + 1))
    raise RuntimeError(f"checkpoint {repo} {name} not downloadable")


# ---------------------------------------------------------------- the unit
KEEP_ENV = ("PATH", "LANG", "LC_ALL", "NVIDIA_VISIBLE_DEVICES", "NVIDIA_DRIVER_CAPABILITIES", "VK_DRIVER_FILES", "TERM")


def proc_env(appdata, home, gpu, ncpu):
    env = {k: os.environ[k] for k in KEEP_ENV if k in os.environ}
    env.update({
        "OMNI_CRASHREPORTER_ENABLED": "0",
        "PATH": f"{ENVP}/bin:" + env.get("PATH", "/usr/bin:/bin"), "CONDA_PREFIX": ENVP,
        "PYTHONPATH": f"{EVAL_ROOT}:{EVAL_ROOT}/Behavior-1K/OmniGibson:{EVAL_ROOT}/Behavior-1K/bddl3",
        # ISG_OG_DATA_PATH: a datasets dir with the assets staged on node-local disk (the cluster job; NFS asset reads made a
        # clone's scene import ~3x slower there); default the worktree's datasets symlinks
        "OMNIGIBSON_DATA_PATH": os.environ.get("ISG_OG_DATA_PATH") or f"{EVAL_ROOT}/Behavior-1K/datasets", "OMNIGIBSON_APPDATA_PATH": str(appdata), "TMPDIR": str(appdata / "tmp"),
        "OMNIGIBSON_HEADLESS": "1", "OMNI_KIT_ACCEPT_EULA": "YES", "HDF5_USE_FILE_LOCKING": "FALSE", "HF_HUB_OFFLINE": "1",
        "OMNIGIBSON_TRACK_STATE_UPDATES": "1", "OMP_NUM_THREADS": "1", "OMNIGIBSON_CHANGE_DETECTION_NUMPY": "1",
        "ISPATIALGYM_SHARED_GOAL_EVAL": "1", "ISPATIALGYM_FAST_GEOMETRY": "1", "ISPATIALGYM_FAST_ASSIGN": "1", "NUMPY_MADVISE_HUGEPAGE": "0", "ISPATIALGYM_CUROBO_WARMUP_ITERS": "0", "PYTHONHASHSEED": "0",
        "PYTHONDONTWRITEBYTECODE": "1", "HOME": str(home), "XDG_CACHE_HOME": f"{home}/.cache", "TORCHINDUCTOR_CACHE_DIR": str(WORK / "inductor_cache"),
        "TRITON_CACHE_DIR": str(WORK / "triton_cache"), "NUMBA_CACHE_DIR": f"{home}/numba", "MKL_NUM_THREADS": str(ncpu), "OPENBLAS_NUM_THREADS": str(ncpu),
        "CUDA_VISIBLE_DEVICES": str(gpu), "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_CACHE_PATH": str(WORK / "cuda_cache"), "CUDA_CACHE_MAXSIZE": str(4 * 2**30),
        "ISG_TEXTURE_WAIT_MAX_S": os.environ.get("ISG_TEXTURE_WAIT_MAX_S", "240"),
    })
    for d in ("numba", ".cache"):
        Path(home, d).mkdir(parents=True, exist_ok=True)
    for d in ("inductor_cache", "triton_cache", "cuda_cache"):
        (WORK / d).mkdir(exist_ok=True)
    return env


def server_launch(unit, gpu, ncpu, step=None):
    """The policy server command of a sweep entry (``run_isg_vec`` fills {ckpt} / {port}): ACT's b1k_server with temporal
    aggregation, or openpi's serve_b1k through pi_serve.sh (which fetches + extracts the step's tar first)."""
    if unit["kind"] == "openpi":
        return {"launch": ["bash", str(Path(__file__).resolve().parent / "pi_serve.sh"), unit["hf_repo"], str(step), "{port}", unit["task"], os.environ.get("ISG_PI_ROBOT", "b1k/R1Pro")],
                "cwd": str(OPENPI_DIR), "clear_pythonpath": True,
                "env": {"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": str(max(1, min(4, ncpu // 2))), "OPENPI_DIR": str(OPENPI_DIR), "OPENPI_VENV": str(PI_VENV),
                        "XLA_PYTHON_CLIENT_PREALLOCATE": "false", "XLA_PYTHON_CLIENT_MEM_FRACTION": os.environ.get("ISG_PI_MEM_FRACTION", "0.35"),
                        "OPENPI_DATA_HOME": str(PI_DATA), "ISG_PI_CKPT_ROOT": PI_CKPT_ROOT, "ISG_PI_WORK": str(PI_WORK),
                        **({"SSL_CERT_FILE": "/etc/ssl/certs/ca-certificates.crt"} if os.path.exists("/etc/ssl/certs/ca-certificates.crt") else {})}}
    code = CODE_ROOTS[unit["code_dir"]]
    if unit["kind"] == "dp":
        # Diffusion Policy's serve_b1k.py: the same per-step protocol as ACT's server (its handshake says so); it executes
        # its own plan of ``--action-horizon`` actions (default: the checkpoint's n_action_steps) -- no temporal aggregation
        server_args = [str(a) for a in (unit.get("server_args") or [])]
        return {"launch": ["bash", "-c", 'until [ -s "$1" ]; do sleep 5; done; shift; exec "$@"', "wait_ckpt", "{ckpt}",
                           DP_PYTHON, str(code / "scripts/b1k/serve_b1k.py"), "--model-path", "{ckpt}", "--host", "127.0.0.1", "--port", "{port}", *server_args],
                "cwd": str(code), "clear_pythonpath": True,
                "env": {"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": str(max(1, min(4, ncpu // 2))), "PYTHONPATH": str(code), "HF_HUB_OFFLINE": "1"}}
    # the serving flags: the unit's ``server_args`` when given (an inference-variant campaign, e.g. ["--temporal-agg",
    # "--temporal-agg-steps", "16"] or ["--action-horizon", "16"]), else the protocol default (RUNBOOK §2: temporal aggregation)
    server_args = [str(a) for a in (unit.get("server_args") or ["--temporal-agg"])]
    # the checkpoint may still be prefetching when the sweep reaches it: the server starts once the (complete) file is there
    return {"launch": ["bash", "-c", 'until [ -s "$1" ]; do sleep 5; done; shift; exec "$@"', "wait_ckpt", "{ckpt}",
                       f"{ENVP}/bin/python", str(code / "scripts/b1k/serve_b1k.py"), "--model-path", "{ckpt}", "--host", "127.0.0.1", "--port", "{port}", *server_args],
            "cwd": str(code), "clear_pythonpath": True,
            "env": {"CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": str(max(1, min(4, ncpu // 2))), "PYTHONPATH": str(code), "HF_HUB_OFFLINE": "1"}}


def tag_episodes(src):
    """(episode dir name -> meta.json dict) of a tag directory (``_batches`` excluded; an unreadable meta counts as an error)."""
    out = {}
    for d in sorted(src.iterdir()):
        if d.is_dir() and d.name != "_batches" and (d / "meta.json").exists():
            try:
                out[d.name] = json.load(open(d / "meta.json"))
            except Exception:  # noqa: BLE001
                out[d.name] = {"status": "error", "unreadable": True}
    return out


def upload_tag(uid, out_dir, tag, status, okey=None, need=11):
    """One finished checkpoint's outputs to out/<out_key>/<tag>/ (videos included), then a marker: ``_done.json`` when every
    one of the ``need`` instances has an error-free episode, ``_partial.json`` otherwise (a crash-time upload of a partial tag
    used to write _done.json and later units skipped the tag for good: 47 tags on 2026-10-24)."""
    okey = okey or uid
    src = out_dir / tag
    if not src.is_dir():
        return
    for attempt in range(4):
        r = s5("cp", f"{src}/", f"{S3_URL}/out/{okey}/{tag}/", check=False)
        if r.returncode == 0:
            break
        log(f"upload {uid}/{tag}: retry {attempt + 1} ({r.stderr.strip()[-200:]})")
        time.sleep(15)
    else:
        status.setdefault("upload_failed", []).append(tag)
        return
    rows = {}
    rp = src / "results.jsonl"
    if rp.exists():
        for l in rp.read_text().splitlines():
            if l.strip():
                row = json.loads(l)
                rows[row.get("instance")] = row  # an instance's latest row
    rows = list(rows.values())
    metas = tag_episodes(src)
    errors = sorted({name for name, m in metas.items() if m.get("status") == "error"} | {r.get("instance") for r in rows if r.get("status") == "error"})
    complete = len(metas) >= need and not errors
    summary = {"unit": uid, "out_key": okey, "tag": tag, "instance": INSTANCE, "episodes": len(metas), "rows": len(rows), "needed": need, "complete": complete,
               "successes": sum(1 for r in rows if r.get("success")), "errors": len(errors), "error_episodes": errors[:11], "time": time.time(),
               "val_successes": sum(1 for r in rows if r.get("success") and r.get("episode_index") is not None),
               "eval_successes": sum(1 for r in rows if r.get("success") and r.get("episode_index") is None),
               "budget_factor": 1.5, "max_steps": sorted({r.get("max_steps") for r in rows if r.get("max_steps")})[-3:]}
    put_json(f"out/{okey}/{tag}/{'_done' if complete else '_partial'}.json", summary)
    delete(f"out/{okey}/{tag}/{'_partial' if complete else '_done'}.json")
    with lock:
        status.setdefault("tags_done" if complete else "tags_partial", []).append(summary)
        state["episodes_done"] += len(rows)
    log(f"UPLOADED {uid}/{tag}: {summary['successes']}/{summary['episodes']} successes ({summary['errors']} errors)"
        + ("" if complete else f" PARTIAL: {len(metas)}/{need} episodes, errors {errors[:3]}"))


def run_unit(unit, gpu, slot, kk, cpus):
    uid = unit["unit"]
    okey = unit.get("out_key") or uid  # out/<out_key>/<tag>/: shared by the units of one (run, task) pair
    udir = WORK / "units" / uid
    shutil.rmtree(udir, ignore_errors=True)
    udir.mkdir(parents=True)
    status = {"unit": uid, "out_key": okey, "gpu": gpu, "slot": slot, "started": time.time(), "phase": "setup", "steps_done": 0, "n_steps": len(unit["steps"]),
              "steps": list(unit["steps"]), "wave": unit.get("wave"), "priority": unit.get("priority")}
    with lock:
        state["running"][slot] = status
    t0 = time.time()
    appdata = WORK / "appdata" / f"{hashlib.sha1(uid.encode()).hexdigest()[:10]}.{slot}"  # short: NVRTC refuses a long TMPDIR ("name of directory for temporary files is too long")
    out = udir / "out"
    ckpt_dir = WORK / "ckpt" / unit["run"]
    try:
        if unit["kind"] not in ("act", "dp", "openpi"):
            raise RuntimeError(f"unit kind {unit['kind']} is not served by this worker")
        status["phase"] = "resume"
        out.mkdir(parents=True)
        # a previous worker's finished checkpoints of this pair (any unit sharing the output key): their meta.json files make
        # run_isg_vec skip them; a unit whose every checkpoint is done is complete at once
        wanted = [f"step-{st:08d}" for st in unit["steps"]]
        need = int(unit["n_instances"])
        names = list_names(f"out/{okey}/")
        remote_eps = {}
        markers = {}
        for n in names:
            parts = n.split("/")
            if len(parts) == 3 and parts[2] == "meta.json" and parts[1] != "_batches":
                remote_eps.setdefault(parts[0], set()).add(parts[1])
            elif len(parts) == 2 and parts[1] in ("_done.json", "_partial.json"):
                markers.setdefault(parts[0], set()).add(parts[1])
        # done: a _done.json that says complete (new markers) or, for the markers of before 2026-10-24, one whose episode count
        # reaches the pair's instances with no error, and every instance's meta.json on R2; anything else with episodes is
        # partial: its error-free episodes are kept (meta.json + their results rows) and run_isg_vec runs the missing ones
        done_tags, partial_tags = [], []
        for tag in wanted:
            if tag not in markers and tag not in remote_eps:
                continue
            rec = get_json(f"out/{okey}/{tag}/_done.json") if "_done.json" in markers.get(tag, ()) else None
            complete = bool(rec) and len(remote_eps.get(tag, ())) >= need and (
                rec.get("complete") is True or ("complete" not in rec and int(rec.get("episodes") or 0) >= need and not rec.get("errors")))
            (done_tags if complete else partial_tags).append(tag)
        if all(t in done_tags for t in wanted):
            put_json(f"done/{uid}.json", {"unit": uid, "out_key": okey, "instance": INSTANCE, "tag": TAG, "slot": slot, "seconds": 0.0, "num_envs": None,
                                          "kind": unit["kind"], "tags": [], "resumed_tags": wanted, "rc": 0, "time": time.time(), "note": "every checkpoint was done already"})
            with lock:
                state["done"].append(uid)
                state["successes"] += 1
            log(f"DONE {uid}: every checkpoint {wanted} was done already")
            return True
        for tag in done_tags:
            s5("cp", f"{S3_URL}/out/{okey}/{tag}/*/meta.json", f"{out}/{tag}/", check=False)
        for tag in partial_tags:
            s5("cp", f"{S3_URL}/out/{okey}/{tag}/*/meta.json", f"{out}/{tag}/", check=False)
            s5("cp", f"{S3_URL}/out/{okey}/{tag}/results.jsonl", f"{out}/{tag}/results.jsonl", check=False)
            kept, dropped = [], []
            for name, m in tag_episodes(out / tag).items():
                if m.get("status") == "error" or m.get("success") is None:
                    shutil.rmtree(out / tag / name, ignore_errors=True)  # runs again
                    dropped.append(name)
                else:
                    kept.append(name)
            rp = out / tag / "results.jsonl"
            if rp.exists():
                rows = [l for l in rp.read_text().splitlines() if l.strip()]
                keep_rows = [l for l in rows if json.loads(l).get("instance") in kept and json.loads(l).get("status") != "error"]
                rp.write_text("".join(x + "\n" for x in keep_rows))
            log(f"{uid}: partial tag {tag}: {len(kept)} error-free episodes kept, {len(dropped)} errored dropped, {need - len(kept)} to run")
        if done_tags:
            log(f"{uid}: {len(done_tags)} checkpoints already done elsewhere: {done_tags[:5]}")
        status["phase"] = "checkpoints"
        steps = list(unit["steps"])
        first = steps[0]
        num_envs = min(NUM_ENVS, int(unit.get("verified_num_envs") or NUM_ENVS))
        if GPUS[gpu]["mem_mb"] < 30000:
            num_envs = min(num_envs, 6)  # a 24 GB card (4090 / 3090 / A5000): 8 house_single_floor clones took 18 GB, plus the ACT server
        if GPUS[gpu]["mem_mb"] < 22000:
            num_envs = min(num_envs, 4)  # a 20 GB card (A4500 / 4000 Ada)
        if unit.get("max_num_envs"):
            num_envs = min(num_envs, int(unit["max_num_envs"]))  # a per-unit cap (re-runs of tags that failed at the verified N)
        if unit["kind"] == "openpi":
            # the pi server shares the GPU with the world: 11 clones + JAX only on >= 44 GB, else two batches of <= 6
            ensure_openpi_venv()
            num_envs = min(num_envs, NUM_ENVS if GPUS[gpu]["mem_mb"] >= 44000 else 6)  # never above the task's verified N
            image_size = [0, 0]  # native frames: openpi's wrapper does its own resize_with_pad
            task_id = int(unit["task_id"])
            ckinfo = {"kind": "openpi"}
            sweep = [{"tag": f"step-{st:08d}", "ckpt": "", "step": st, "run": unit["run"], **server_launch(unit, gpu, len(cpus), step=st)} for st in steps]
        else:
            fetch_checkpoint(unit["hf_repo"], first, ckpt_dir / f"step-{first:08d}.pt")
            # the checkpoint's training image size: the client resizes + quantizes the frames with the training path at that size
            if unit["kind"] == "dp":
                # a DP checkpoint: config.image_size (square); only a checkpoint trained in ACT's resize mode (`act`) takes
                # frames resized with ACT's path on the client -- others get the native frames and resize on the server
                inspect = ("import json,sys,torch; c=torch.load(sys.argv[1], map_location='cpu', weights_only=True); cfg=c['config']; "
                           "s=int(cfg['image_size']); act=cfg.get('resize_mode', 'dp') == 'act'; "
                           "print(json.dumps({'image_size': [s, s] if act else [0, 0], 'goal_views': list(cfg.get('goal_views') or []), "
                           "'resize_mode': cfg.get('resize_mode', 'dp'), 'task_map': {int(k): v for k, v in c['task_map'].items()}, 'step': c.get('step')}))")
            else:
                inspect = ("import json,sys,torch; c=torch.load(sys.argv[1], map_location='cpu', weights_only=True); "
                           "print(json.dumps({'image_size': c['adapter_config']['image_size'], 'goal_views': c['adapter_config'].get('goal_views'), "
                           "'task_map': {int(k): v for k, v in c['task_map'].items()}, 'step': c.get('step')}))")
            ck = subprocess.run([DP_PYTHON if unit["kind"] == "dp" else f"{ENVP}/bin/python", "-c", inspect, str(ckpt_dir / f"step-{first:08d}.pt")],
                                capture_output=True, text=True, timeout=600)
            if ck.returncode:
                raise RuntimeError(f"checkpoint inspection failed: {ck.stderr[-400:]}")
            ckinfo = json.loads(ck.stdout.strip().splitlines()[-1])
            ckinfo["task_map"] = {int(k): v for k, v in ckinfo["task_map"].items()}  # JSON keys come back as strings
            image_size = [int(v) for v in ckinfo["image_size"]]
            by_name = {v: k for k, v in ckinfo["task_map"].items()}
            if unit["task"] not in by_name:
                raise RuntimeError(f"task {unit['task']} is not in the checkpoint's task map {sorted(by_name)[:5]}...")
            task_id = int(by_name[unit["task"]])  # the checkpoint's own id of the task (the dataset's task_index for every run so far)
            if task_id != int(unit["task_id"]):
                log(f"{uid}: checkpoint task id {task_id} differs from the dataset's {unit['task_id']}; the checkpoint's is sent")

            def prefetch():
                for st in steps[1:]:
                    try:
                        fetch_checkpoint(unit["hf_repo"], st, ckpt_dir / f"step-{st:08d}.pt")
                    except Exception as e:  # noqa: BLE001
                        log(f"{uid}: prefetch of step {st} failed: {e}")

            threading.Thread(target=prefetch, daemon=True).start()
            sweep = [{"tag": f"step-{st:08d}", "ckpt": str(ckpt_dir / f"step-{st:08d}.pt"), "step": st, "run": unit["run"], **server_launch(unit, gpu, len(cpus))} for st in steps]
        status["image_size"] = image_size
        status["task_id"] = task_id
        status["num_envs"] = num_envs
        (udir / "sweep.json").write_text(json.dumps(sweep, indent=1))
        status["phase"] = "textures"
        ensure_scene_textures("common")
        ensure_scene_textures(unit["scene"])
        warm, ntex = make_appdata(appdata)
        env = proc_env(appdata, udir / "home", gpu, len(cpus))
        inst_file = VEC_ROOT / unit["instances"]
        cmd = [f"{ENVP}/bin/python", "-u", "-m", "ispatialgym.oracle.run_isg_vec", "--mode", "policy", "--settle-eval", "on", "--instances", str(inst_file), "--num-envs", str(num_envs),
               "--policy-sweep", str(udir / "sweep.json"), "--out", str(out), "--tag", f"{TAG}:{INSTANCE}", "--record-view", "tiled", "--tile-px", "240",
               "--external-size", os.environ.get("ISG_EXTERNAL_SIZE", "240"), "--grid-video", "--server-start-timeout", "3600" if unit["kind"] == "openpi" else "900",
               "--client-image-size", str(image_size[0]), str(image_size[1]), "--task-id", str(task_id), "--policy-timeout", "1200" if unit["kind"] == "openpi" else "300"]
        if os.environ.get("ISG_NO_EXTERNAL") == "1":
            cmd += ["--no-external-camera"]
        if os.environ.get("ISG_RUN_ISG_VEC_ARGS"):
            cmd += os.environ["ISG_RUN_ISG_VEC_ARGS"].split()
        if cpus:
            cmd = ["taskset", "-c", ",".join(map(str, cpus))] + cmd
        logf = udir / "run.log"
        status["phase"] = "run"
        status["warm"] = warm
        log(f"START {uid} slot={slot} gpu={gpu} task={unit['task']} instances={unit['n_instances']} steps={len(steps)} warm={warm} tex_links={ntex}")
        uploads = []
        total_s = 3600 * 2 + unit["frames"] * 0.6 / max(1, min(num_envs, unit["n_instances"])) * 1.5 + len(steps) * (1500 if unit["kind"] == "openpi" else 300)
        def launch(extra):
            """One run_isg_vec process (+ the sweep servers it starts), its log followed; a Kit start that never reached
            OmniGibson is retried once on fresh app data. Returns the exit code."""
            nonlocal warm, ntex, env
            rc, seen_start = None, False
            for attempt in range(2):
                if attempt:
                    log(f"RETRY {uid}: the process died before OmniGibson started (rc={rc}); fresh app data")
                    rm_tree(appdata)
                    warm, ntex = make_appdata(appdata)
                    env = proc_env(appdata, udir / "home", gpu, len(cpus))
                t_launch = time.time()
                with open(logf, "ab") as fh:
                    p = subprocess.Popen(cmd + list(extra), env=env, cwd=str(EVAL_ROOT), stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
                status["pid"] = p.pid
                # follow the log: a finished sweep entry is uploaded at once; a stall kills the process
                size, pending, last_growth, seen_start, gpu_crash, seen_done = os.path.getsize(logf) if logf.exists() else 0, "", time.time(), False, False, False
                while True:
                    rc = p.poll()
                    try:
                        cur = os.path.getsize(logf)
                    except OSError:
                        cur = size
                    if cur != size:
                        with open(logf, "rb") as fh:
                            fh.seek(size)
                            chunk = fh.read().decode(errors="replace")
                        size = cur
                        last_growth = time.time()
                        lines = (pending + chunk).split("\n")
                        pending = lines.pop()
                        for line in lines:
                            if "Welcome to OmniGibson" in line:
                                seen_start = True
                            if line.startswith("RUN_ISG_VEC_DONE"):
                                seen_done = True  # the driver finished; Kit often segfaults on shutdown afterwards (rc -11 / -6)
                            if "GPU crash is detected" in line or "ERROR_DEVICE_LOST" in line:
                                gpu_crash = True
                            m = re.match(r"RUN_ISG_VEC sweep (\d+)/(\d+) (\S+): (\d+) episodes", line)
                            if m:
                                tag = m.group(3)
                                status["steps_done"] = int(m.group(1))
                                t = threading.Thread(target=upload_tag, args=(uid, out, tag, status, okey, int(unit['n_instances'])), daemon=True)
                                t.start()
                                uploads.append(t)
                            if line.startswith("RUN_ISG_VEC_BATCH"):
                                status["last_batch"] = line[:300]
                            m2 = re.search(r"policy server (step-\d{8}):", line)
                            if m2:
                                status["current_tag"] = m2.group(1)
                    if rc is not None:
                        break
                    now = time.time()
                    reason = ""
                    if gpu_crash:
                        reason = "vulkan GPU crash"  # Kit hangs in vkWaitForFences after ERROR_DEVICE_LOST: no point waiting for the stall timer
                    elif not seen_start and now - t_launch > (2400 if not warm else 1500):
                        reason = "no start"
                    elif now - last_growth > 2700:
                        reason = "log stalled 45 min"
                    elif now - t_launch > total_s:
                        reason = f"timeout {total_s:.0f}s"
                    if reason:
                        log(f"KILL {uid}: {reason}")
                        try:
                            os.killpg(p.pid, 9)
                        except ProcessLookupError:
                            pass
                        p.wait()
                        rc = -9
                        break
                    time.sleep(10)
                if seen_done and rc != 0:
                    log(f"{uid}: the driver finished (RUN_ISG_VEC_DONE) but the process exited {rc} (Kit shutdown crash): counted as 0")
                    rc = 0
                if rc in (0, 3) or seen_start or rc == -9:  # 3: several worlds (handled below); -9: killed by the stall / timeout watchdog above
                    break
            return rc

        rc = launch([])
        if rc == 3:
            # the unit's instances span several worlds (templates that differ beyond the task objects): one process per world group
            m = re.search(r"RUN_ISG_VEC_WORLDS (\d+) sizes=(\[[\d, ]*\])", logf.read_text(errors="replace"))
            n_worlds, sizes = (int(m.group(1)), json.loads(m.group(2))) if m else (0, [])
            log(f"{uid}: {n_worlds} worlds {sizes}; one process per world group")
            for k in range(n_worlds):
                rc = launch(["--world-group", str(k), "--num-envs", str(max(1, min(num_envs, sizes[k])))])
                if rc != 0:
                    break
        for t in uploads:
            t.join(timeout=1800)
        # anything finished but not announced (a crash after the last batch): upload the tags with results
        announced = {s["tag"] for s in status.get("tags_done", [])}
        for tag_dir in sorted(out.glob("step-*")):
            if tag_dir.name not in announced and (tag_dir / "results.jsonl").exists() and tag_dir.name not in done_tags:
                upload_tag(uid, out, tag_dir.name, status, okey, int(unit['n_instances']))
        s5("cp", str(logf), f"{S3_URL}/logs/units/{uid}/{INSTANCE}-run.log", check=False)
        if (out / "_servers").is_dir():
            s5("cp", f"{out}/_servers/", f"{S3_URL}/logs/units/{uid}/{INSTANCE}-servers/", check=False)
        done_now = {s["tag"] for s in status.get("tags_done", [])} | set(done_tags)
        elapsed = time.time() - t0
        if rc == 0 and all(f"step-{st:08d}" in done_now for st in steps):
            put_json(f"done/{uid}.json", {"unit": uid, "instance": INSTANCE, "tag": TAG, "gpu_name": GPUS[gpu]["name"], "driver": GPUS[gpu]["driver"], "slot": slot,
                                          "seconds": elapsed, "warm": warm, "num_envs": num_envs, "kind": unit["kind"], "tags": status.get("tags_done"), "resumed_tags": done_tags,
                                          "image_size": image_size, "checkpoint_info": {k: v for k, v in ckinfo.items() if k != "task_map"},
                                          "rc": rc, "time": time.time()})
            with lock:
                state["done"].append(uid)
                state["successes"] += 1
                first_ok = state["successes"] == 1
            log(f"DONE {uid} in {elapsed:.0f}s ({len(steps)} checkpoints)")
            try:
                save_textures(appdata)
                if first_ok:
                    save_template(appdata, kk)
            except Exception as e:  # noqa: BLE001  (the unit is done; a cache save must not fail it)
                log(f"{uid}: cache save after the unit failed: {type(e).__name__}: {e}")
            return True
        raise RuntimeError(f"rc={rc} checkpoints done {len(done_now)}/{len(steps)}")
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        log(f"FAIL {uid} slot={slot}: {msg}")
        with lock:
            state["failed"].append(uid)
            local_fails[uid] = local_fails.get(uid, 0) + 1
            if not status.get("tags_done"):
                state["startup_fails"] += 1
        try:
            n = fail_count(uid)
            (udir / "fail.txt").write_text(f"instance {INSTANCE} slot {slot} gpu {GPUS[gpu]['name']} {GPUS[gpu]['driver']}\n{msg}\n{traceback.format_exc()}\n")
            s5("cp", str(udir / "fail.txt"), f"{S3_URL}/fail/{uid}/{INSTANCE}-{n}.txt", check=False)
            if (udir / "run.log").exists():
                s5("cp", str(udir / "run.log"), f"{S3_URL}/logs/units/{uid}/{INSTANCE}-{n}-run.log", check=False)
            if (appdata / "local/logs").exists():
                s5("cp", f"{appdata}/local/logs/", f"{S3_URL}/logs/units/{uid}/{INSTANCE}-{n}-kitlogs/", check=False)
            with lock:
                proven = state["successes"] > 0 or bool(status.get("tags_done"))
            if n + 1 >= MAX_UNIT_FAILS and proven:
                put_json(f"dead/{uid}.json", {"unit": uid, "fails": n + 1, "last": msg}, if_none_match=True)
            delete(f"claims/{uid}.json")
        except Exception as e2:  # noqa: BLE001
            log(f"fail bookkeeping for {uid} failed: {e2}")
        return False
    finally:
        with lock:
            state["running"].pop(slot, None)
        rm_tree(appdata)
        shutil.rmtree(udir / "out", ignore_errors=True)
        shutil.rmtree(udir / "home", ignore_errors=True)


# ---------------------------------------------------------------- the replay verification unit
def run_vec_process(cmd, env, cwd, logf, status, startup_s=2400, stall_s=2700, total_s=4 * 3600, cpus=None):
    """One run_isg_vec process: killed on a stall / timeout; returns (rc, reason)."""
    if cpus:
        cmd = ["taskset", "-c", ",".join(map(str, cpus))] + cmd
    with open(logf, "ab") as fh:
        p = subprocess.Popen(cmd, env=env, cwd=cwd, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)
    status["pid"] = p.pid
    t0 = last = time.time()
    size, seen, gpu_crash = 0, False, False
    while True:
        rc = p.poll()
        try:
            cur = os.path.getsize(logf)
        except OSError:
            cur = size
        if cur != size:
            with open(logf, "rb") as fh:
                fh.seek(size)
                chunk = fh.read()
            seen = seen or b"Welcome to OmniGibson" in chunk
            gpu_crash = gpu_crash or b"GPU crash is detected" in chunk or b"ERROR_DEVICE_LOST" in chunk
            size, last = cur, time.time()
        if rc is not None:
            return rc, ""
        now = time.time()
        reason = ""
        if gpu_crash:
            reason = "vulkan GPU crash"  # Kit hangs in vkWaitForFences after ERROR_DEVICE_LOST: killed at once
        elif not seen and now - t0 > startup_s:
            reason = "no start"
        elif now - last > stall_s:
            reason = f"log stalled {stall_s}s"
        elif now - t0 > total_s:
            reason = f"timeout {total_s}s"
        if reason:
            try:
                os.killpg(p.pid, 9)
            except ProcessLookupError:
                pass
            p.wait()
            return -9, reason
        time.sleep(10)


def run_verify_unit(unit, gpu, slot, kk, cpus):
    """``verify__<task>``: the task's eval-dataset instance replayed open loop at N=1, then the same demo in every slot at
    N=8 (the ladder 8 -> 4 -> 2 while the comparison fails); outputs (per-slot tiled videos, grid videos, trajectories,
    compare.json, verify.json) to out/<unit>/, the verdict to verify/<task>.json (which releases the task's policy units)."""
    uid = unit["unit"]
    udir = WORK / "units" / uid
    shutil.rmtree(udir, ignore_errors=True)
    udir.mkdir(parents=True)
    status = {"unit": uid, "gpu": gpu, "slot": slot, "started": time.time(), "phase": "setup", "kind": "replay_verify", "task": unit["task"]}
    with lock:
        state["running"][slot] = status
    t0 = time.time()
    appdata = WORK / "appdata" / f"{hashlib.sha1(uid.encode()).hexdigest()[:10]}.{slot}"  # short: NVRTC refuses a long TMPDIR ("name of directory for temporary files is too long")
    out = udir / "out"
    out.mkdir(parents=True)
    inst_file = VEC_ROOT / unit["instances"]
    verdict = {"task": unit["task"], "unit": uid, "instance": INSTANCE, "pass": False, "num_envs": None, "runs": {}, "time": None}
    try:
        status["phase"] = "textures"
        ensure_scene_textures("common")
        ensure_scene_textures(unit["scene"])
        base = [f"{ENVP}/bin/python", "-u", "-m", "ispatialgym.oracle.run_isg_vec", "--mode", "replay", "--instances", str(inst_file), "--record-view", "tiled",
                "--tile-px", "240", "--external-size", "240", "--grid-video", "--tag", f"{TAG}:{INSTANCE}"]
        L = int(unit.get("demo_length") or 1500)
        per_run_s = 3600 + L * 1.5

        def run(n, same):
            name = f"n{n}same" if same else f"n{n}"
            for attempt in range(2):
                warm, ntex = make_appdata(appdata)
                env = proc_env(appdata, udir / "home", gpu, len(cpus))
                cmd = base + ["--num-envs", str(n), "--out", str(out / name)] + (["--same-instance-all-slots"] if same else [])
                status["phase"] = name
                log(f"VERIFY {uid}: {name} start (warm={warm}, attempt {attempt + 1})")
                shutil.rmtree(out / name, ignore_errors=True)
                rc, reason = run_vec_process(cmd, env, str(EVAL_ROOT), udir / f"{name}.log", status, total_s=per_run_s + 600 * n, cpus=cpus, stall_s=2700 if n < 4 else 5400)
                s5("cp", str(udir / f"{name}.log"), f"{S3_URL}/logs/units/{uid}/{INSTANCE}-{name}{'' if attempt == 0 else '-retry'}.log", check=False)
                ok = rc == 0 and any((out / name).glob("*/meta.json"))
                verdict["runs"][name] = {"rc": rc, "reason": reason, "ok": ok, "seconds": round(time.time() - t0, 1), "attempts": attempt + 1}
                rm_tree(appdata)
                if ok:
                    return out / name
                # a Kit start that never reached the world build (extension solver / cold app-data races) is retried once
                try:
                    started = "RUN_ISG_VEC world built" in (udir / f"{name}.log").read_text(errors="replace")
                except Exception:  # noqa: BLE001
                    started = True
                if started or attempt == 1:
                    raise RuntimeError(f"{name}: rc={rc} {reason}")
                log(f"VERIFY {uid}: {name} died before the world build (rc={rc}); retrying once")
            raise RuntimeError(f"{name}: no result")

        ref = run(1, False)
        ladder = [int(n) for n in unit.get("num_envs_ladder") or [8, 4, 2]]
        passed_n = None
        for n in ladder:
            run_dir = run(n, True)
            cmp_cmd = [f"{ENVP}/bin/python", str(EVAL_ROOT / "scripts/vec_eval/compare_slots.py"), str(run_dir), "--ref", str(ref)]
            r = subprocess.run(cmp_cmd, capture_output=True, text=True, cwd=str(EVAL_ROOT), env=dict(os.environ, PYTHONPATH=str(EVAL_ROOT)))
            (run_dir / "compare.txt").write_text(r.stdout + r.stderr)
            r = subprocess.run([f"{ENVP}/bin/python", str(EVAL_ROOT / "scripts/vec_eval/verify_rules.py"), str(run_dir), "--ref", str(ref)], capture_output=True, text=True,
                               cwd=str(EVAL_ROOT), env=dict(os.environ, PYTHONPATH=str(EVAL_ROOT)))
            dec = json.load(open(run_dir / "verify.json")) if (run_dir / "verify.json").exists() else {"pass": False, "reasons": [f"verify_rules failed: {r.stderr[-300:]}"]}
            verdict["runs"][f"n{n}same"].update({"pass": dec.get("pass"), "reasons": dec.get("reasons"), "instances": dec.get("instances")})
            log(f"VERIFY {uid}: N={n} {'PASS' if dec.get('pass') else 'FAIL'} {dec.get('reasons')[:3] if dec.get('reasons') else ''}")
            if dec.get("pass"):
                passed_n = n
                break
        # released either way: at the largest N whose slots reproduced the N=1 replay, else at N=1 (bit-identical to the single-env evaluator)
        verdict.update({"pass": True, "num_envs": passed_n or 1, "verified_at": passed_n, "fallback_single_env": passed_n is None, "time": time.time(),
                        "seconds": round(time.time() - t0, 1), "gpu_name": GPUS[gpu]["name"]})
        status["phase"] = "upload"
        for attempt in range(4):
            r = s5("cp", f"{out}/", f"{S3_URL}/out/{uid}/", check=False)
            if r.returncode == 0:
                break
            time.sleep(15)
        put_json(f"verify/{unit['task']}.json", verdict)
        put_json(f"done/{uid}.json", {**verdict, "slot": slot})
        with lock:
            state["done"].append(uid)
            state["successes"] += 1
        log(f"VERIFY DONE {uid}: policy N={verdict['num_envs']} (verified at N={passed_n}) in {time.time() - t0:.0f}s")
        return True
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        log(f"FAIL {uid} slot={slot}: {msg}")
        with lock:
            state["failed"].append(uid)
            local_fails[uid] = local_fails.get(uid, 0) + 1
            if not verdict["runs"]:
                state["startup_fails"] += 1
        try:
            n = fail_count(uid)
            (udir / "fail.txt").write_text(f"instance {INSTANCE} slot {slot} gpu {GPUS[gpu]['name']}\n{msg}\n{traceback.format_exc()}\n{json.dumps(verdict, default=str)}\n")
            s5("cp", str(udir / "fail.txt"), f"{S3_URL}/fail/{uid}/{INSTANCE}-{n}.txt", check=False)
            s5("cp", f"{out}/", f"{S3_URL}/out/{uid}/attempt-{INSTANCE}-{n}/", check=False)
            if n + 1 >= MAX_UNIT_FAILS:
                put_json(f"dead/{uid}.json", {"unit": uid, "fails": n + 1, "last": msg}, if_none_match=True)
            delete(f"claims/{uid}.json")
        except Exception as e2:  # noqa: BLE001
            log(f"fail bookkeeping for {uid} failed: {e2}")
        return False
    finally:
        with lock:
            state["running"].pop(slot, None)
        rm_tree(appdata)
        shutil.rmtree(udir / "out", ignore_errors=True)
        shutil.rmtree(udir / "home", ignore_errors=True)


# ---------------------------------------------------------------- heartbeat / loops
def utilization():
    out = {}
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=30)
        out["gpu_util"] = [int(x.split(",")[0]) for x in r.stdout.strip().splitlines()]
        out["gpu_mem_mb"] = [int(x.split(",")[1]) for x in r.stdout.strip().splitlines()]
    except Exception:  # noqa: BLE001
        pass
    mi = open("/proc/meminfo").read()
    tot = int(re.search(r"MemTotal:\s+(\d+)", mi).group(1))
    avail = int(re.search(r"MemAvailable:\s+(\d+)", mi).group(1))
    out["mem_used_gb"] = round((tot - avail) / 2**20, 1)
    return out


def heartbeat_loop(kk):
    n = 0
    while True:
        with lock:
            running = [{k: v for k, v in r.items() if k != "pid"} for r in state["running"].values()]
            hb = {"instance": INSTANCE, "tag": TAG, "phase": state["phase"], "time": time.time(), "uptime": time.time() - t_start, "gpus": [g["name"] for g in GPUS],
                  "driver": GPUS[0]["driver"], "cpus": len(ALLOWED), "quota": QUOTA, "mem_gb": round(MEM, 1), "slots": SLOTS, "kit_key": kk, "template": template_ready(),
                  "num_envs": NUM_ENVS, "running": running, "done": len(state["done"]), "failed": len(state["failed"]), "episodes_done": state["episodes_done"],
                  "load1": os.getloadavg()[0], "disk_free_gb": shutil.disk_usage(str(WORK)).free / 2**30, **utilization()}
        try:
            put_json(f"hb/{INSTANCE}.json", hb)
            if n % 5 == 0:
                s5("cp", f"{LOG}/*", f"{S3_URL}/logs/{INSTANCE}/", check=False)
            if n % 3 == 0:
                for r in hb["running"]:
                    lf = WORK / "units" / r["unit"] / "run.log"
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


CODE_SHA = {"value": None}
code_lock = threading.Lock()


def refresh_code():
    """Before a unit: fetch ``bundle/code.tar.zst`` (the eval worktree's python code, absolute paths like the bundle)
    when its ``bundle/code.sha`` changed since the last unit, so a fix reaches running instances without a reboot."""
    if os.environ.get("ISG_NO_CODE_REFRESH") == "1":  # a workstation worker: its checkout is rsynced, the tar's absolute paths do not apply
        return
    with code_lock:
        try:
            r = s5("cat", f"{S3_URL}/bundle/code.sha", check=False)
            sha = r.stdout.strip() if r.returncode == 0 else ""
        except Exception as e:  # noqa: BLE001
            log(f"code refresh: sha lookup failed ({e})")
            return
        if not sha or sha == CODE_SHA["value"]:
            return
        tar = WORK / "code.tar.zst"
        r = s5("cp", f"{S3_URL}/bundle/code.tar.zst", str(tar), check=False)
        if r.returncode != 0 or not tar.exists():
            log(f"code refresh: download failed ({r.stderr.strip()[-200:]})")
            return
        r = subprocess.run(["bash", "-c", f"zstd -dc {tar} | tar -xf - -C /"], capture_output=True, text=True)
        if r.returncode != 0:
            log(f"code refresh: extract failed ({r.stderr.strip()[-200:]})")
            return
        subprocess.run(["bash", "-c", f"find {EVAL_ROOT}/ispatialgym {EVAL_ROOT}/scripts -name __pycache__ -type d -prune -exec rm -rf {{}} +"], capture_output=True)
        # the Behavior-1K overlay (scene placement, attachment callbacks) rides in scripts/: re-apply it to the worker's copy
        r = subprocess.run(["bash", f"{EVAL_ROOT}/scripts/vec_eval/apply_b1k_overlay.sh", B1K_DIR], capture_output=True, text=True)
        if r.returncode != 0:
            log(f"code refresh: overlay failed ({r.stderr.strip()[-200:]})")
        log(f"code refresh: {CODE_SHA['value'] or 'boot'} -> {sha}")
        CODE_SHA["value"] = sha
        tar.unlink(missing_ok=True)


MAX_UNITS = int(os.environ.get("ISG_MAX_UNITS", "0"))  # > 0: the slot exits after this many units (a smoke test)


def slot_loop(slot, gpu, kk, cpus):
    time.sleep(20 * slot)
    waited = 0
    n_units = 0
    while not stop_claiming.is_set():
        if MAX_UNITS and n_units >= MAX_UNITS:
            log(f"slot {slot}: ISG_MAX_UNITS={MAX_UNITS} reached")
            break
        unit = claim_next()
        if unit is None:
            break
        if unit == "wait":
            if waited == 0:
                log(f"slot {slot}: nothing claimable yet (policy units wait for their task's replay verification); polling")
            waited += 1
            time.sleep(60)
            if waited > 240:  # 4 h without anything to claim: give the GPU back
                log(f"slot {slot}: waited 4 h for a claimable unit; exiting")
                break
            continue
        waited = 0
        n_units += 1
        refresh_code()
        if unit["kind"] == "replay_verify":
            run_verify_unit(unit, gpu, slot, kk, cpus)
        else:
            run_unit(unit, gpu, slot, kk, cpus)
        with lock:
            bad = state["successes"] == 0 and state["startup_fails"] >= max(2, len(SLOTS))
        if bad:
            log("instance unusable: units fail at startup and none succeeded")
            stop_claiming.set()
    log(f"slot {slot} exits")


PI_OK = False


def main():
    global GPUS, MEM, SLOTS, PI_OK
    GPUS = gpus()
    MEM = mem_gb()
    # pi05 needs >= 40 GB of GPU memory and room for the 12 GB step tar + its extraction (20 GB when the steps are served
    # from the extracted ISG_PI_CKPT_ROOT instead)
    PI_OK = (min(g["mem_mb"] for g in GPUS) >= 40000 and shutil.disk_usage(str(WORK)).free > (20 if PI_CKPT_ROOT else 60) * 2**30
             and os.environ.get("ISG_NO_PI") != "1")
    SLOTS = [(i, g["index"]) for i, g in enumerate(GPUS)]  # one process per GPU
    cpus = {s: slot_cpus(k, len(SLOTS)) for k, (s, g) in enumerate(SLOTS)}
    kk = kit_key(GPUS[0])
    log(f"WORKER instance={INSTANCE} gpus={[g['name'] for g in GPUS]} driver={GPUS[0]['driver']} cpus={len(ALLOWED)} quota={QUOTA} mem={MEM:.0f}GB "
        f"num_envs={NUM_ENVS} units={len(UNITS)} kit_key={kk} cpus/slot={len(cpus[0])}")
    hb = threading.Thread(target=heartbeat_loop, args=(kk,), daemon=True)
    state["phase"] = "template"
    hb.start()
    if not fetch_template(kk):
        log("no Kit template for this GPU + driver on the bundle prefix: the first process starts cold")
    state["phase"] = "run"
    threads = [threading.Thread(target=slot_loop, args=(s, g, kk, cpus[s])) for s, g in SLOTS]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    state["phase"] = "done" if state["successes"] or not state["failed"] else "failed"
    log(f"WORKER END done={len(state['done'])} failed={len(state['failed'])} episodes={state['episodes_done']}")
    time.sleep(70)
    return 0 if state["phase"] == "done" else 3


if __name__ == "__main__":
    sys.exit(main())
