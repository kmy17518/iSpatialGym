"""i7-vast coordinator (cluster side; run with the replay tools venv after creds.sh): the ONE controller of the round-4 fast-flow Vast
backend. It drains $VJ/queue (jobs submit_fast.py queued through vsubmit's sbatch), stages each job's inputs to R2, rents Vast
instances (label og-isgof-vast-*, four accounts), assigns jobs to them, syncs every finished job dir back into its cluster job dir
(and its log into <root>/slurm/, its new cache files into the campaign caches), queues the batch's job_memory.sh on slurm, and
destroys instances that are idle, stuck or over budget.

    vastctl.py loop                  the coordinator (job_coord.sh runs it as a slurm CPU job)
    vastctl.py status                queue, instances, spend
    vastctl.py offers [--gpu G]      fitting offers now (cheapest per job slot first)
    vastctl.py stage <id>            stage one queued job (test)
    vastctl.py destroy <instance>    destroy one of this job's instances (owning account)
    vastctl.py stop-all              ctl/stop for every worker + STOP_LAUNCH (drain)

Config $VJ/config.json (re-read every tick): gpus (allowed models), default_gpu, max_instances, budget_usd, max_dph_gpu, disk_gb,
idle_destroy_s, boot_max_s, launch_per_tick. $VJ/STOP_LAUNCH stops new rentals. Ledger $VJ/ledger/events.jsonl (create / destroy
with account variable name, instance, offer, machine, $/h); per-job rows $VJ/ledger/jobs.tsv.
"""
import argparse
import configparser
import concurrent.futures
import glob
import json

import os
import re
import shlex
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import uuid
from collections import Counter
from pathlib import Path

import boto3
import botocore
from vastai.api import instances as vi
from vastai.api import offers as vo
from vastai.api.client import VastClient
try:
    from .campaign_inputs import prepare_environment, prepare_cache_environment, prepare_queue, prepare_pools, dispatch_order, runtime_scenes, assignment_plan, staging_record, production_decision, preserved_instances, collector_prebuild_patterns
    from .bounded_process import run_bounded
    from .worker.resource_policy import machine_reservation, duplicate_worker_gpus, bad_machines, host_startup_failure
    from .campaign_inputs import ordered_gpu_offers, compatible_gpus, gpu_matches, uncovered_jobs, drained_workers, rental_accounts, drained_worker_finished, worker_can_run, template_customs_inputs, register_exact_template_customs_inputs, camera_context_inputs
    from .async_staging import StagingReservations, staging_parallelism, staging_compression_threads
    from .async_results import ResultFetchReservations, CoordinatorStateCache, typed_preparation, execute_reserved_fetch
    from .lifecycle import retain_attempt
    from .cost_provenance import scope as cost_scope, rental_intent, retain as retain_cost, assignment_identity
    from .funding import read_snapshot as read_credit_snapshot, remaining_credit
    from .fleet_accounting import observe as observe_rentals
    from .fleet_policy import rental_ceiling, activated_queue_reserve
    from . import ik_transport
    from . import cluster_return
except ImportError:
    from campaign_inputs import prepare_environment, prepare_cache_environment, prepare_queue, prepare_pools, dispatch_order, runtime_scenes, assignment_plan, staging_record, production_decision, preserved_instances, collector_prebuild_patterns
    from bounded_process import run_bounded
    from worker.resource_policy import machine_reservation, duplicate_worker_gpus, bad_machines, host_startup_failure
    from campaign_inputs import ordered_gpu_offers, compatible_gpus, gpu_matches, uncovered_jobs, drained_workers, rental_accounts, drained_worker_finished, worker_can_run, template_customs_inputs, register_exact_template_customs_inputs, camera_context_inputs
    from async_staging import StagingReservations, staging_parallelism, staging_compression_threads
    from async_results import ResultFetchReservations, CoordinatorStateCache, typed_preparation, execute_reserved_fetch
    from lifecycle import retain_attempt
    from cost_provenance import scope as cost_scope, rental_intent, retain as retain_cost, assignment_identity
    from funding import read_snapshot as read_credit_snapshot, remaining_credit
    from fleet_accounting import observe as observe_rentals
    from fleet_policy import rental_ceiling, activated_queue_reserve
    import ik_transport
    import cluster_return

OFH = "/vision/u/minyeong/projects/isg_home/campaign_20261110"
F = "/vision/u/minyeong/projects/isg_home/campaign_20261110"
T85 = "/vision/u/minyeong/projects/isg_home/train85"
SY = "/vision/u/shiyuc/minyeong"
VJ = os.environ.get("VJ", os.path.join(OFH, "vast"))
JOB_ID = os.environ["JOB_ID"]
JOB_S3 = os.environ["JOB_S3"]
REPLAY_S3 = os.environ["REPLAY_S3"]
BUCKET = os.environ["S3_BUCKET"]
PREFIX = JOB_S3.split(f"s3://{BUCKET}/", 1)[1].rstrip("/")
LABEL = "og-isg261110"
IMAGE = "ghcr.io/kmy17518/isg-replay:v3"
LEDGER = os.path.join(VJ, "ledger", "events.jsonl")
JOBS_TSV = os.path.join(VJ, "ledger", "jobs.tsv")
STATE = os.path.join(VJ, "state", "jobs")
QUEUE = os.path.join(VJ, "queue")
ACCOUNTS = ("VAST_API_KEY", "VAST_QUERY_API_KEY_1", "VAST_QUERY_API_KEY_2", "VAST_QUERY_API_KEY_3")
SHORT = {"VAST_API_KEY": "main", "VAST_QUERY_API_KEY_1": "q1", "VAST_QUERY_API_KEY_2": "q2", "VAST_QUERY_API_KEY_3": "q3"}
ACCOUNT_ERRORS = ("spend_limit", "credit", "balance", "fund", "billing", "payment", "verif")
#: the cache the jobs read and write: shipped as base + deltas at boot, new files synced back (never overwritten)
CACHES = ()  # Each validation job starts with its own empty sampling cache.
#: never shipped: the asset-geometry cache holds link-local collision points of the decrypted BEHAVIOR meshes (no decrypted asset data
#: goes to R2 or back); the instances fill their own copy (demos bit-identical with or without it)
NOT_SHIPPED = (os.path.join(OFH, "asset_cache"),)
#: read stores shipped in stores.tar.zst (pack_static.sh)
STORES = ("/vision/u/minyeong/projects/ispatialgym-lab-vec-eval-cache/reach_bands", "/vision/u/minyeong/projects/isg_home/eval85/bands_eval",
          "/vision/u/minyeong/projects/isg_home/.cache/ispatialgym-vecenv/reach_volume", os.path.join(T85, "bands_train"))
#: job.env keys that are not inputs of job_fast.sh (or are shipped otherwise)
SKIP_KEYS = {"ISGE_CODE", "ISGF_SCRIPTS", "ISGF_ROOT", "COUNTED_RUN", "COUNTED_SNAPSHOT", "ISGF_PREDEMO_ROOT", "ISGF_SAMPLING_CACHE",
             "ISGF_ASSET_CACHE", "ISGF_JB_NFS", "KIT_TPL_NFS_ROOT", "KIT_TEX_STORE", "ISGF_EXP_PATH", "EXP_PATH",
             "ISGF_REPLAY_SOURCE_JOB", "ISGF_REPLAY_SOURCE_SNAPSHOT", "ISGF_REPLAY_HELPER_SNAPSHOT",
             "ISPATIALGYM_SAMPLING_CACHE", "ISPATIALGYM_CACHE_DIR", "ISPATIALGYM_HELD_HOLD_CACHE"}
SKIP_KEYS.update({'ISPATIALGYM_EXACT_IK_LOCAL_CACHE', 'ISPATIALGYM_EXACT_IK_METRICS'})
PATH_MAX_BYTES = 3 * 2**30
DEFAULT_CONFIG = {"gpus": ["RTX 4090"], "default_gpu": "RTX 4090", "max_instances": 1, "budget_usd": 300.0, "max_dph_gpu": 1.5,
                  "disk_gb": 280, "idle_destroy_s": 420, "boot_max_s": 3300, "launch_per_tick": 3, "min_inet_down": 500,
                  "vram_per_job_gb": 20, "ram_per_job_gb": 40, "cpus_per_job": 9, "max_jobs_per_instance": 2, "retries": 2,
                  "search_every_s": 300, "max_contracts_per_machine": 4, "market_limit": 1000}
STAGER = None
RESULT_FETCHER = None
JOB_STATE_CACHE = None
STAGE_COMPRESSION_THREADS = 4


def now():
    return time.time()


def cfg():
    p = os.path.join(VJ, "config.json")
    c = dict(DEFAULT_CONFIG)
    if os.path.exists(p):
        c.update(json.load(open(p)))
    return c


def ledger(ev, **kw):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"ev": ev, "time": now(), **kw}) + "\n")


def cost_record(kind, identity, **payload):
    # Observational metadata must not interrupt collection if its disk is unavailable.
    try:
        return retain_cost(os.path.join(VJ, 'state', 'cost_provenance'), kind, identity, payload)
    except Exception as error:
        log('COST_PROVENANCE_UNAVAILABLE', kind, str(identity), type(error).__name__)
        return {'status': 'unavailable', 'error_type': type(error).__name__}


def events():
    if not os.path.exists(LEDGER):
        return []
    return [json.loads(l) for l in open(LEDGER) if l.strip()]


def log(*a):
    print(f"[{time.strftime('%m-%d %H:%M:%S')}]", *a, flush=True)


# ---------------------------------------------------------------------------------------------------------------- R2


def r2():
    return boto3.client("s3", endpoint_url=os.environ["S3_ENDPOINT_URL"], region_name="auto")


def r2_key():
    c = configparser.ConfigParser()
    c.read(os.environ["AWS_SHARED_CREDENTIALS_FILE"])
    p = c[os.environ["AWS_PROFILE"]]
    return p["aws_access_key_id"], p["aws_secret_access_key"]


def r2_list(s3, sub):
    out, token = {}, None
    while True:
        kw = {"Bucket": BUCKET, "Prefix": f"{PREFIX}/{sub}"}
        if token:
            kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        for o in r.get("Contents", []):
            out[o["Key"][len(f"{PREFIX}/{sub}"):]] = o
        if not r.get("IsTruncated"):
            return out
        token = r["NextContinuationToken"]


def r2_json(s3, key):
    try:
        return json.loads(s3.get_object(Bucket=BUCKET, Key=f"{PREFIX}/{key}")["Body"].read())
    except botocore.exceptions.ClientError:
        return None


def read_heartbeats(s3, instance_ids=None):
    keys = [key for key in r2_list(s3, 'hb/')
            if key.endswith('.json') and (instance_ids is None or key[:-5] in instance_ids)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
        values = pool.map(lambda key: (key[:-5], r2_json(s3, 'hb/' + key)), keys)
        return {iid:value for iid,value in values if value}


def r2_put(s3, key, body):
    s3.put_object(Bucket=BUCKET, Key=f"{PREFIX}/{key}", Body=body.encode() if isinstance(body, str) else body)


def r2_exists(s3, key):
    try:
        s3.head_object(Bucket=BUCKET, Key=f"{PREFIX}/{key}")
        return True
    except botocore.exceptions.ClientError:
        return False


def s5(*args, check=True, timeout=7200):
    r = subprocess.run(["s5cmd", "--endpoint-url", os.environ["S3_ENDPOINT_URL"], "--retry-count", "10", *args], capture_output=True,
                       text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"s5cmd {args[0]}: {r.stderr.strip()[:300]}")
    return r


def tar_put(listfile, key, level=3):
    """Ship the fixed closure, allowing only parent-directory timestamp races.

    Other jobs may create siblings while GNU tar records an explicitly listed
    parent directory. Its exit 1 then says nothing about this archive's payload.
    Retain directory entries/mtimes, verify every non-directory before and after
    packing, and accept only that precise diagnostic. Payload read errors,
    mutations, removals and compressor/uploader failures still fail staging.
    """
    import stat

    def fingerprint(path):
        s = os.lstat('/' + path)
        identity = (s.st_dev, s.st_ino, s.st_mode, s.st_uid, s.st_gid)
        # A directory's children are never recursively included here.
        if stat.S_ISDIR(s.st_mode):
            return True, identity
        return False, identity + (s.st_size, s.st_mtime_ns, s.st_ctime_ns)

    names = Path(listfile).read_text().splitlines()
    if not names or any(not name or name.startswith('/') or '..' in Path(name).parts for name in names):
        raise ValueError('archive closure must contain nonempty paths relative to /')
    before = {name: fingerprint(name) for name in names}
    with tempfile.TemporaryDirectory(prefix='isg-vast-pack-') as temporary:
        directory = Path(temporary)
        listing, errors, statuses = (directory / name for name in ('files', 'stderr', 'statuses'))
        listing.write_bytes(b'\0'.join(os.fsencode(name) for name in names) + b'\0')
        cmd = (f"LC_ALL=C tar -C / --format=posix --quoting-style=literal --no-recursion --null --verbatim-files-from "
               f"-cf - -T {shlex.quote(str(listing))} 2>{shlex.quote(str(errors))} | "
               f"zstd -q -T{STAGE_COMPRESSION_THREADS} -{level} | "
               f"s5cmd --endpoint-url {shlex.quote(os.environ['S3_ENDPOINT_URL'])} pipe {shlex.quote(JOB_S3 + '/' + key)}\n"
               f"printf '%s\\n' \"${{PIPESTATUS[@]}}\" > {shlex.quote(str(statuses))}\n")
        subprocess.run(['bash', '-o', 'pipefail', '-c', cmd], check=True, timeout=7200, stdout=subprocess.DEVNULL)
        changed = [name for name, original in before.items() if fingerprint(name) != original]
        if changed:
            raise RuntimeError('archive input changed during packing: ' + repr(changed[:5]))
        codes = [int(value) for value in statuses.read_text().splitlines()]
        diagnostics = errors.read_text()
        allowed = {f'tar: {name}: file changed as we read it' for name, (is_dir, _) in before.items() if is_dir}
        directory_race = bool(diagnostics) and all(line in allowed for line in diagnostics.splitlines())
        if len(codes) != 3 or codes[1:] != [0, 0] or not (codes[0] == 0 or (codes[0] == 1 and directory_race)):
            raise RuntimeError(f'archive pipeline failed {codes}: {diagnostics[-2000:]}')


# ---------------------------------------------------------------------------------------------------------------- staging


def read_env(job_dir):
    env = {}
    for line in open(os.path.join(job_dir, "job.env")):
        line = line.rstrip("\n")
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            try:
                env[k] = " ".join(shlex.split(v)) if v else ""
            except ValueError:
                env[k] = v
    return env


def under(p, roots):
    p = os.path.abspath(p)
    return any(p == r or p.startswith(r.rstrip("/") + "/") for r in roots)


class Lister:
    """The file list of a job's input archive (paths relative to /, dirs listed so their mtimes come back)."""

    def __init__(self, exclude_roots):
        self.items = set()
        self.exclude = list(exclude_roots)
        self.bytes = 0

    def _parents(self, p):
        d = os.path.dirname(p)
        while d not in ("/", ""):
            self.items.add(d.lstrip("/"))
            d = os.path.dirname(d)

    def add_file(self, p):
        if under(p, self.exclude) or not os.path.lexists(p):
            return
        self._parents(p)
        self.items.add(p.lstrip("/"))
        try:
            self.bytes += os.lstat(p).st_size
        except OSError:
            pass

    def add_tree(self, root, skip_runs=False, guard=PATH_MAX_BYTES):
        root = os.path.abspath(root)
        if under(root, self.exclude):
            return
        if os.path.isfile(root) or os.path.islink(root):
            return self.add_file(root)
        tot = 0
        self._parents(root)
        for d, dns, fns in os.walk(root):
            if skip_runs and "runs" in dns and os.path.exists(os.path.join(d, "template.json")):
                dns.remove("runs")
            self.items.add(d.lstrip("/"))
            for fn in fns:
                p = os.path.join(d, fn)
                self.items.add(p.lstrip("/"))
                try:
                    tot += os.lstat(p).st_size
                except OSError:
                    pass
            for dn in list(dns):
                if os.path.islink(os.path.join(d, dn)):
                    self.items.add(os.path.join(d, dn).lstrip("/"))
            if tot > guard:
                raise RuntimeError(f"input {root} is over {guard / 2**30:.0f} GB: not shipped per job (add it to the static bundles)")
        self.bytes += tot

    def write(self, path):
        with open(path, "w") as f:
            f.write("\n".join(sorted(self.items)) + "\n")


def ensure_code(s3, d, state):
    """bundle/code/<basename>.tar.zst of a frozen code dir (a trunk snapshot or its Behavior-1K copy), uploaded once."""
    key = f"bundle/code/{os.path.basename(d.rstrip('/'))}.tar.zst"
    if key in state.setdefault("code", {}) or r2_exists(s3, key):
        state["code"][key] = True
        return key
    lst = os.path.join(VJ, "bundle", f"code_{os.path.basename(d)}_{uuid.uuid4().hex[:12]}.txt")
    L = Lister([])
    for dd, dns, fns in os.walk(d):
        dns[:] = [x for x in dns if x not in ("__pycache__", ".pytest_cache")]
        L.items.add(dd.lstrip("/"))
        for fn in fns:
            if not fn.endswith(".pyc"):
                L.items.add(os.path.join(dd, fn).lstrip("/"))
        for dn in dns:
            if os.path.islink(os.path.join(dd, dn)):
                L.items.add(os.path.join(dd, dn).lstrip("/"))
    L._parents(d)
    L.write(lst)
    t = now()
    tar_put(lst, key)
    os.replace(lst, os.path.join(VJ, "bundle", f"code_{os.path.basename(d)}.txt"))
    log(f"CODE {key} ({len(L.items)} entries, {now() - t:.0f} s)")
    state["code"][key] = True
    return key


def git_full(d):
    r = subprocess.run(["git", "-C", d, "rev-parse", "HEAD"], capture_output=True, text=True)
    return r.stdout.strip() if r.returncode == 0 else None


def stage(s3, rec, state):
    """Upload the job's inputs: jobs/<id>/jobdata.tar.zst (the job dir + every input path job_fast.sh reads: job.env's paths, the eval
    index's instances, the prebuild candidates, camera relocalization's earlier train instances) and the code archives. -> spec."""
    if rec.get('job_kind') == 'native_dependency':
        return stage_native_dependency(s3, rec, state)
    jd = rec["job_dir"]
    env = read_env(jd)
    replay = rec.get('job_kind') == 'physics_replay'
    if replay:
        # Replay transports immutable source artifacts, not a fresh sampler
        # queue. The dedicated wrapper sets private image-native caches.
        pool_inputs = []
    else:
        if env.get("ISGF_JB_NFS", "").startswith(SY) or env.get("ISPATIALGYM_FAST_BAND_SHARE", "0") == "1":
            raise RuntimeError("shiyuc-mode / band-sharing jobs stay on the cluster")
        prepare_environment(jd, OFH)
        prepare_cache_environment(jd, OFH)
        prepare_queue(jd, env, OFH)
        env = read_env(jd)
        pool_inputs = prepare_pools(jd, env, OFH)
    code = os.path.abspath(env["ISGE_CODE"])
    b1k = os.path.dirname(os.path.realpath(os.path.join(code, "Behavior-1K", "OmniGibson")))
    # tools the job scripts call that the image lacks (rsync: kit_appdata.sh, job_fast.sh's collector app-data copies and prebuild dir)
    code_keys = ["bundle/tools/rsync.tar.zst", ensure_code(s3, code, state)]
    exact_ik = None
    if not replay and env.get('ISPATIALGYM_EXACT_IK_MANIFEST'):
        try:
            helper_code = str(Path(__file__).resolve().parents[3])
            exact_ik = ik_transport.prepare(OFH, jd, env, s3, BUCKET, PREFIX, helper_code)
            helper_key = ensure_code(s3, helper_code, state)
            if helper_key not in code_keys:
                code_keys.append(helper_key)
        except Exception as exc:
            # Ordinary native solving is always available when sharing fails.
            log('EXACT_IK_SEED_UNAVAILABLE ' + str(rec['id']) + ' ' + type(exc).__name__)
    if not under(b1k, [code]):
        code_keys.append(ensure_code(s3, b1k, state))
    shipped = [code, b1k] + list(STORES) + list(CACHES) + list(NOT_SHIPPED)
    L = Lister(shipped)
    L.add_tree(jd)
    if replay:
        sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
        from scripts.campaign_20261110.vast_replay import transport_inputs, sha as replay_sha
        if replay_sha(Path(jd)/'replay_inputs.json') != rec.get('replay_inputs_sha256'):
            raise ValueError('replay transport manifest changed after enqueue')
        for path in transport_inputs(jd, OFH):
            L.add_tree(path, skip_runs=True)
    for path in pool_inputs:
        L.add_file(path)
    # These metadata customs are sampler inputs. Replay uses the pinned TRO
    # and the original native snapshot's own customs/tolerance definitions.
    if not replay:
        register_exact_template_customs_inputs(env, OFH)
    customs_inputs = [] if replay else template_customs_inputs(env, OFH)
    camera_inputs = camera_context_inputs(env, OFH)
    for dependency in customs_inputs + camera_inputs:
        L.add_file(dependency['path'])
    scripts = env.get("ISGF_SCRIPTS") or os.path.join(F, "scripts")
    if not under(scripts, [code]):
        L.add_tree(scripts)
    # every absolute path job.env names (TEMPLATE, COUNTED_TEMPLATE, EVAL_INDEX, the v2 glib / maps / anchors, the alias-memory version,
    # the map-overlay root, BAND_STORES, --extra-env paths)
    for k, v in env.items():
        if k in SKIP_KEYS:
            continue
        for tok in re.split(r"[\s,:=]+", v):
            if tok.startswith("/vision/") and os.path.exists(tok) and not under(tok, shipped):
                L.add_tree(tok, skip_runs=True)
    # the eval index's instances (instance_separation.load_instances reads their TRO files) with their template dirs
    ei = env.get("EVAL_INDEX")
    if ei and os.path.isfile(ei):
        try:
            idx = json.load(open(ei))
            for ins in idx.get("instances") or []:
                p = ins.get("path") if isinstance(ins, dict) else None
                if p and os.path.exists(p):
                    td = os.path.dirname(os.path.dirname(p)) if os.path.isfile(p) else p
                    L.add_tree(td, skip_runs=True)
        except (ValueError, OSError) as e:
            log(f"stage {rec['id']}: eval index {ei}: {e!r}")
    # COLLECTOR_PREBUILD candidates (job_fast.sh picks the newest fitting one of these on the instance, as on the cluster)
    pats = collector_prebuild_patterns(env, replay=replay, primary_root=F, legacy_root=T85)
    n_pre = 0
    for pat in pats:
        for d in glob.glob(pat):
            if env.get("ISPATIALGYM_FAST_COLLECTOR_PREBUILD") == "0" or under(d, [jd]):
                continue
            L.add_tree(d, skip_runs=True)
            for p in (os.path.dirname(os.path.dirname(os.path.dirname(d))), ):  # the source job's job.env (prebuild_crop.py reads it)
                if os.path.exists(os.path.join(p, "job.env")):
                    L.add_file(os.path.join(p, "job.env"))
            n_pre += 1
    # camera relocalization: train_predemo.py compares with the earlier train instances under <predemo root>/out/<key>/*
    if not replay and env.get("FAMILY") == "camera_relocalization":
        key = env["ALIAS_KEY"]
        tr = env.get("ISGF_PREDEMO_ROOT") or F
        for other in glob.glob(os.path.join(tr, "out", key, "*")):
            if under(other, [jd]) or not os.path.isdir(other):
                continue
            for p in [os.path.join(other, "instances.tsv")] + glob.glob(os.path.join(other, "claims", "*")) + \
                    glob.glob(os.path.join(other, "inst", "*", "*", "instances", "*-tro_state.json")) + \
                    glob.glob(os.path.join(other, "inst", "*", "*", "instances", "*.summary.json")):
                L.add_file(p)
    lst = os.path.join(VJ, "bundle", f"jobdata_{rec['id']}.txt")
    L.write(lst)
    t = now()
    tar_put(lst, f"jobs/{rec['id']}/jobdata.tar.zst")
    sha = git_full(os.path.join(code, "Behavior-1K"))
    roots = []
    if sha:
        por = os.path.join(b1k, ".snapshot_patches")
        roots = [{"dir": os.path.join(code, "Behavior-1K"), "sha": sha, "porcelain": por if os.path.exists(por) else "-"},
                 {"dir": b1k, "sha": sha, "porcelain": por if os.path.exists(por) else "-"}]
    spec = {k: rec[k] for k in ("id", "name", "job_dir", "script", "time_s", "cpus", "mem", "log", "cwd", "acct", "part")}
    spec.update({"code": [{"key": k} for k in code_keys], "jobdata": f"jobs/{rec['id']}/jobdata.tar.zst", "scenes": runtime_scenes(env, OFH),
                 "git_roots": roots, "staged": now(), "inputs": {"entries": len(L.items), "bytes": L.bytes, "prebuild_candidates": n_pre,
                 "template_customs": customs_inputs, "camera_context": camera_inputs}})
    if exact_ik:
        spec['exact_ik_exchange'] = exact_ik
    log(f"STAGED {rec['id']} {rec['name']} {len(L.items)} entries {L.bytes / 2**20:.1f} MB, {n_pre} prebuild candidates ({now() - t:.0f} s)")
    return spec


def library_prebuild_reader():
    """Load a controller-pinned reader independently of other helper imports."""
    import hashlib
    import importlib.util
    path = os.environ.get('ISG_VAST_LIBRARY_READER')
    if not path:
        from scripts.campaign_20261110 import vast_prebuild
        return vast_prebuild
    source = Path(path)
    if hashlib.sha256(source.read_bytes()).hexdigest() != os.environ.get('ISG_VAST_LIBRARY_READER_SHA256'):
        raise ValueError('Pinned consumed-library reader changed')
    spec = importlib.util.spec_from_file_location('isg_pinned_library_reader', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def map_prebuild_reader():
    """Use the reviewed consumed-map reader even if older helpers are cached."""
    import hashlib
    import importlib.util
    path = os.environ.get('ISG_VAST_MAP_READER')
    if not path:
        from scripts.campaign_20261110 import vast_map_prebuild
        return vast_map_prebuild
    source = Path(path)
    if hashlib.sha256(source.read_bytes()).hexdigest() != os.environ.get('ISG_VAST_MAP_READER_SHA256'):
        raise ValueError('Pinned consumed-map reader changed')
    spec = importlib.util.spec_from_file_location('isg_pinned_map_reader', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage_native_dependency(s3, rec, state):
    """Use the existing private worker transport for an exact native canary."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    typed=json.loads((Path(rec['job_dir'])/'native_inputs.json').read_text())
    diagnostic=typed.get('schema')=='isg-vast-ik-native-probe-v1'
    prebuild=typed.get('schema')=='isg-vast-library-prebuild-v1'
    map_prebuild=typed.get('schema')=='isg-vast-map-prebuild-v1'
    contact_map=typed.get('schema')=='isg-vast-contact-map-prebuild-v1'
    if diagnostic:
        from scripts.campaign_20261110 import vast_ik_probe as native
    elif prebuild:
        native = library_prebuild_reader()
    elif contact_map:
        from scripts.campaign_20261110 import vast_contact_map as native
    elif map_prebuild:
        native = map_prebuild_reader()
    else:
        from scripts.campaign_20261110 import vast_native_dependency as native
    jd=rec['job_dir'];value=native.verify(jd,OFH)
    if native.sha(Path(jd)/'native_inputs.json') != rec['native_inputs_sha256']:
        raise ValueError('native dependency manifest changed after enqueue')
    code=value['snapshot'];b1k=os.path.dirname(os.path.realpath(os.path.join(code,'Behavior-1K','OmniGibson')))
    code_keys=['bundle/tools/rsync.tar.zst',ensure_code(s3,code,state)]
    exact_ik=None
    if diagnostic or prebuild or map_prebuild or contact_map:
        helper_code=value['helper_snapshot']
        code_keys.append(ensure_code(s3,helper_code,state))
    if diagnostic:
        exact_ik=ik_transport.prepare(OFH,jd,read_env(jd),s3,BUCKET,PREFIX,helper_code)
        if not exact_ik:raise ValueError('native cache probe requires typed exchange context')
    if not under(b1k,[code]):code_keys.append(ensure_code(s3,b1k,state))
    listing=Lister([code,b1k]+list(STORES)+list(CACHES)+list(NOT_SHIPPED));listing.add_tree(jd)
    for name in value['files']:listing.add_file(name)
    for name in value.get('library_files',{}):listing.add_file(name)
    for name in native.library_environment(OFH,value.get('v2_release_root')).values():
        if value.get('v2_release_root') and name.startswith(value['v2_release_root']+'/'):continue
        if name.startswith('/vision/') and os.path.exists(name):listing.add_tree(name,skip_runs=True)
    path=os.path.join(VJ,'bundle',f"jobdata_{rec['id']}.txt");listing.write(path)
    tar_put(path,f"jobs/{rec['id']}/jobdata.tar.zst")
    commit=git_full(os.path.join(code,'Behavior-1K'));roots=[]
    if commit:
        patches=os.path.join(b1k,'.snapshot_patches')
        roots=[{'dir':os.path.join(code,'Behavior-1K'),'sha':commit,'porcelain':patches if os.path.exists(patches) else '-'},
               {'dir':b1k,'sha':commit,'porcelain':patches if os.path.exists(patches) else '-'}]
    spec={key:rec[key] for key in ('id','name','job_dir','script','time_s','cpus','mem','log','cwd','acct','part')}
    spec.update(code=[{'key':key} for key in code_keys],jobdata=f"jobs/{rec['id']}/jobdata.tar.zst",
                scenes=value['scenes'],git_roots=roots,staged=now(),
                inputs={'entries':len(listing.items),'bytes':listing.bytes,'native_manifest_sha256':rec['native_inputs_sha256']})
    if exact_ik:spec['exact_ik_exchange']=exact_ik
    log(f"STAGED_NATIVE {rec['id']} {listing.bytes / 2**20:.1f} MB; original runtime {os.path.basename(code)}")
    return spec


# ---------------------------------------------------------------------------------------------------------------- results


def safe_extract(tf, dest_root, allow):
    """Extract members whose path is under one of allow (absolute) only; regular files / dirs / symlinks inside."""
    n = 0
    for m in tf:
        p = "/" + m.name.lstrip("./").lstrip("/")
        if ".." in m.name.split("/") or not under(p, allow) or m.isdev():
            continue
        if hasattr(tarfile, "fully_trusted_filter"):
            tf.extract(m, dest_root, filter="fully_trusted")  # paths checked above; keep modes / mtimes as the worker wrote them
        else:
            tf.extract(m, dest_root)
        n += 1
    return n


def sync_back(s3, st):
    """out/<id>/ -> the cluster: the job dir (into its existing dir), slurm.log -> the sbatch -o path, new cache files (only names the
    cluster cache lacks; written to a temporary name and renamed, so a cluster reader never sees half a file)."""
    jid = str(st["id"])
    tmp = os.path.join(VJ, "results", jid)
    os.makedirs(tmp, exist_ok=True)
    have = [n for n in r2_list(s3, f"out/{jid}/") if n in ("jobdir.tar.zst", "slurm.log", "cache_new.tar.zst")]
    for name in have:
        s5("cp", f"{JOB_S3}/out/{jid}/{name}", os.path.join(tmp, name))
    jd = st["job_dir"]
    if "jobdir.tar.zst" in have:
        subprocess.run(["bash", "-o", "pipefail", "-c", f"zstd -d -q -c {tmp}/jobdir.tar.zst > {tmp}/jobdir.tar"], check=True)
        with tarfile.open(f"{tmp}/jobdir.tar") as tf:
            n = safe_extract(tf, "/", [jd])
        os.remove(f"{tmp}/jobdir.tar")
        log(f"SYNC {jid} job dir {jd}: {n} entries")
    if "slurm.log" in have:
        os.makedirs(os.path.dirname(st["log"]), exist_ok=True)
        shutil.copyfile(os.path.join(tmp, "slurm.log"), st["log"])
    added = 0
    if "cache_new.tar.zst" in have:
        subprocess.run(["bash", "-o", "pipefail", "-c", f"zstd -d -q -c {tmp}/cache_new.tar.zst > {tmp}/cache_new.tar"], check=True)
        with tarfile.open(f"{tmp}/cache_new.tar") as tf:
            for m in tf:
                p = "/" + m.name.lstrip("./").lstrip("/")
                if not m.isfile() or ".." in m.name.split("/") or os.path.dirname(p) not in CACHES or os.path.exists(p):
                    continue
                t = os.path.join(os.path.dirname(p), f".vast_{jid}_{os.path.basename(p)}.tmp")
                with tf.extractfile(m) as src, open(t, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                os.utime(t, (m.mtime, m.mtime))
                try:
                    os.link(t, p)  # never replaces a file a cluster job wrote meanwhile
                    added += 1
                except FileExistsError:
                    pass
                os.unlink(t)
        os.remove(f"{tmp}/cache_new.tar")
    shutil.rmtree(tmp, ignore_errors=True)
    return added


# ---------------------------------------------------------------------------------------------------------------- Vast


def vclients():
    return {k: VastClient(api_key=os.environ[k]) for k in ACCOUNTS if os.environ.get(k)}


def vast_error(e):
    try:
        return e.response.json().get("error")
    except Exception:  # noqa: BLE001
        return None


def my_instances(c):
    """This job's instances (label og-isgof-vast*) on every account with _key_name; ok=False when an account's list failed."""
    out, ok = [], True
    for name, cl in c.items():
        try:
            for i in vi.show_instances(cl):
                if str(i.get("label", "")).startswith(LABEL + "-"):
                    i["_key_name"] = name
                    out.append(i)
        except Exception as e:  # noqa: BLE001
            ok = False
            log(f"show instances ({SHORT[name]}) failed: {vast_error(e) or str(e)[:160]}")
    return out, ok


def owner_of(iid):
    for e in events():
        if e["ev"] == "create" and str(e.get("instance")) == str(iid):
            return e.get("account")
    return None


def destroy(c, iid, reason, key_name=None):
    if str(iid) in preserved_instances(F):
        log(f"PRESERVE {iid}: stopped training disk awaits verified artifact harvest")
        return False
    owner = key_name or owner_of(iid)
    order = ([owner] if owner in c else []) + [k for k in c if k != owner]
    ok, used = False, None
    request_started = now()
    for name in order:
        try:
            r = vi.destroy_instance(c[name], iid)
            if isinstance(r, dict) and r.get("success") is False:
                raise RuntimeError(r.get("error") or r.get("msg") or "success: false")
            ok, used = True, name
            break
        except Exception as e:  # noqa: BLE001
            log(f"destroy {iid} with {SHORT[name]} failed: {vast_error(e) or str(e)[:160]}")
    ledger("destroy", instance=iid, reason=reason, ok=ok, account=used,
           request_started_unix=request_started, response_completed_unix=now())
    log(f"DESTROY {iid} [{SHORT.get(used, used)}]: {reason} ({'ok' if ok else 'FAILED'})")
    return ok


def slots_of(o, C):
    n = max(1, int(o.get("num_gpus") or 1))
    vram = float(o.get("gpu_ram") or 0) / 1024
    cpus = float(o.get("cpu_cores_effective") or 0)
    ram = float(o.get("cpu_ram") or 0) / 1024
    return int(max(0, min(n * (vram // C["vram_per_job_gb"]), (cpus - 1) // C["cpus_per_job"], ram // C["ram_per_job_gb"], C["max_jobs_per_instance"])))


def search(c, C, gpu=None, force=False):
    """Offers for one single-GPU host of an allowed model (driver 570-580, verified, reliability >= 0.95), priced with the disk;
    cached for search_every_s (market/fresh_*.json)."""
    files = sorted(glob.glob(os.path.join(VJ, "market", "fresh_*.json")))
    if files and not force and now() - os.path.getmtime(files[-1]) < C["search_every_s"]:
        rows = json.load(open(files[-1]))["offers"]
    else:
        q = {"rentable": {"eq": True}, "rented": {"eq": False}, "external": {"eq": False}, "verified": {"eq": True},
             "gpu_name": {"in": list(C["gpus"])}, "num_gpus": {"eq": 1}, "gpu_ram": {"gte": 23000},
             "inet_down": {"gte": C["min_inet_down"]}, "reliability2": {"gte": 0.95}, "cpu_ram": {"gte": 46000},
             "cpu_cores_effective": {"gte": 10}, "disk_space": {"gte": C["disk_gb"]},
             "driver_version": {"gte": 570000000, "lte": 580999999}, "dph_total": {"lte": C["max_dph_gpu"]}, "inet_down_cost": {"lte": 0.01}, "inet_up_cost": {"lte": 0.01}}
        rows = None
        for name in ("VAST_QUERY_API_KEY_1", "VAST_QUERY_API_KEY_2", "VAST_QUERY_API_KEY_3", "VAST_API_KEY"):
            if not os.environ.get(name):
                continue
            try:
                rows = vo.search_offers(c[name], query=dict(q), order=[["dph_total", "asc"]], limit=int(C.get('market_limit',1000)), storage=C["disk_gb"], no_default=True)
                break
            except Exception as e:  # noqa: BLE001
                log(f"search ({SHORT[name]}) failed: {vast_error(e) or str(e)[:160]}")
        if rows is None:
            return []
        p = os.path.join(VJ, "market", f"fresh_{time.strftime('%Y%m%d_%H%M%S')}.json")
        json.dump({"time": now(), "query": q, "offers": rows}, open(p, "w"))
    bad = bad_machines(events())
    tried = {e.get("offer") for e in events() if e["ev"] == "tried" and now() - e["time"] < 3600}
    out = []
    for o in rows:
        if (gpu and o.get("gpu_name") != gpu) or int(o.get("num_gpus") or 0) != 1:  # the worker runs every job on GPU 0
            continue
        if o.get("machine_id") in bad or o["id"] in tried or str(o.get("geolocation") or "").endswith("CN"):
            continue
        try:
            drv = int(str(o.get("driver_version")).split(".")[0])
        except ValueError:
            continue
        if not 570 <= drv <= 580 or float(o.get("dph_total") or 9) > C["max_dph_gpu"]:
            continue
        s = slots_of(o, C)
        if s < 1:
            continue
        o = dict(o)
        o["_slots"] = s
        o["_per_job"] = float(o["dph_total"]) / s
        out.append(o)
    out.sort(key=lambda o: (o["_per_job"], -float(o.get("cpu_cores_effective") or 0)))
    return out


def fmt_offer(o):
    return (f"{o['id']} {o.get('gpu_name')} ${float(o['dph_total']):.3f}/h slots {o.get('_slots')} (${o.get('_per_job', 0):.3f}/job-h) "
            f"cpu {o.get('cpu_cores_effective')} ram {float(o.get('cpu_ram') or 0) / 1024:.0f}G drv {o.get('driver_version')} "
            f"net {float(o.get('inet_down') or 0):.0f} rel {float(o.get('reliability2') or 0):.3f} {o.get('geolocation')} m{o.get('machine_id')}")


def launch(c, o, C, live_by_acct):
    key_id, secret = r2_key()
    tag = uuid.uuid4().hex[:8]
    env = {"ISG_S3": JOB_S3, "REPLAY_S3": REPLAY_S3, "S3_ENDPOINT_URL": os.environ["S3_ENDPOINT_URL"], "AWS_ACCESS_KEY_ID": key_id,
           "AWS_SECRET_ACCESS_KEY": secret, "AWS_REGION": "auto", "AWS_DEFAULT_REGION": "auto", "NVIDIA_VISIBLE_DEVICES": "all",
           "NVIDIA_DRIVER_CAPABILITIES": "all", "ISG_IDLE_S": str(int(C["idle_destroy_s"]) + 900),
           "ISG_VRAM_PER_JOB_GB": str(C["vram_per_job_gb"]), "ISG_RAM_PER_JOB_GB": str(C["ram_per_job_gb"]),
           "ISG_CPUS_PER_JOB": str(C["cpus_per_job"]), "ISG_MAX_JOBS": str(C["max_jobs_per_instance"]), "ISG_MAX_LIFETIME_S": str(C.get("max_worker_lifetime_s", 21600))}
    env['ISG_CPU_AFFINITY'] = ','.join(map(str, o.get('_reservation',{}).get('cpu_affinity',[])))
    login = f"-u {os.environ['GHCR_USERNAME']} -p {os.environ['GHCR_PULL_TOKEN']} ghcr.io"
    order = rental_accounts(tuple(c), C, live_by_acct)
    for name in order:
        lease_cost = float(o['dph_total']) * float(C.get('max_worker_lifetime_s',21600)) / 3600
        if C.get('_rental_credit_remaining', {}).get(name, 0) < lease_cost:
            continue
        request_started = now()
        cost_request = cost_record('rental_request', tag + '_' + name,
            request_started_unix=request_started, offer_id=o['id'], machine_id=o.get('machine_id'),
            gpu_count=int(o['num_gpus']), account=name, work_intent=C.get('_rental_work_intent',
                {'purpose':'unassigned_shared_campaign_capacity','exclusive_cost_assignment':False}))
        try:
            r = vi.create_instance(c[name], o["id"], image=IMAGE, disk=C["disk_gb"], env=env, label=f"{LABEL}-{tag}", login=login,
                                   runtype="args", args=[], cancel_unavail=True)
        except Exception as e:  # noqa: BLE001
            err = vast_error(e) or type(e).__name__
            cost_record('rental_response',tag+'_'+name,request=cost_request,
                response_completed_unix=now(),instance=None,outcome='create_error',error_type=type(e).__name__)
            if any(w in err for w in ACCOUNT_ERRORS):
                log(f"account {SHORT[name]}: {err} (offer {o['id']}); next account")
                continue
            ledger("tried", offer=o["id"], error=err, account=name)
            log(f"create {o['id']} failed: {err}")
            return None
        iid = r.get("new_contract")
        cost_record('rental_response',tag+'_'+name,request=cost_request,
            response_completed_unix=now(),instance=iid,outcome='created' if iid else 'no_contract_returned')
        if not iid:
            ledger("tried", offer=o["id"], error=str(r)[:80], account=name)
            return None
        ledger("create", instance=iid, offer=o["id"], machine_id=o.get("machine_id"), gpu=o.get("gpu_name"), dph_total=o["dph_total"],
               num_gpus=int(o['num_gpus']),request_started_unix=request_started,response_completed_unix=now(),
               cost_request=cost_request,
               disk=C["disk_gb"], max_lifetime_s=C.get("max_worker_lifetime_s",21600), inet_up_cost=o.get("inet_up_cost"), inet_down_cost=o.get("inet_down_cost"), tag=tag, label=f"{LABEL}-{tag}", account=name, driver=o.get("driver_version"),
               cpus=o.get("cpu_cores_effective"), ram_gb=round(float(o.get("cpu_ram") or 0) / 1024), slots=o["_slots"],
               **o.get('_reservation', {}),
               geo=o.get("geolocation"), reliability=o.get("reliability2"), inet_down=o.get("inet_down"))
        log(f"LAUNCHED {iid} [{SHORT[name]}] {fmt_offer(o)}")
        live_by_acct[name] = live_by_acct.get(name, 0) + 1
        C['_rental_credit_remaining'][name] -= lease_cost
        return iid
    return None


def spend():
    created, ended = {}, {}
    for e in events():
        if e["ev"] == "create" and e.get("instance"):
            created[str(e["instance"])] = e
        elif e['ev'] == 'gpu_identity_backfill' and str(e.get('instance')) in created:
            original = created[str(e['instance'])]
            if e.get('offer') == original.get('offer') and e.get('machine_id') == original.get('machine_id'):
                original['gpu_ids'] = e['gpu_ids']
        elif e["ev"] == "destroy" and e.get("ok"):
            ended.setdefault(str(e["instance"]), e["time"])
    tot = sum(float(e["dph_total"]) * (ended.get(i, now()) - e["time"]) / 3600 for i, e in created.items())
    live = [i for i in created if i not in ended]
    return tot, live, created


# ---------------------------------------------------------------------------------------------------------------- job state


def load_states():
    os.makedirs(STATE, exist_ok=True)
    out = {}
    for p in glob.glob(os.path.join(STATE, "*.json")):
        s = json.load(open(p))
        out[str(s["id"])] = s
    return out


def save_state(s):
    p = os.path.join(STATE, f"{s['id']}.json")
    temporary = p + '.' + uuid.uuid4().hex + '.tmp'
    try:
        with open(temporary, "x") as stream:
            json.dump(s, stream, indent=1); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, p)
    except BaseException:
        if JOB_STATE_CACHE is not None:JOB_STATE_CACHE.invalidate()
        raise


def ingest(states, C):
    for p in sorted(glob.glob(os.path.join(QUEUE, "9*.json"))):
        if p.endswith(".memory.json"):
            continue
        if Path(p).stem in states:
            continue  # Immutable queue entries are already owned by this coordinator.
        rec = json.load(open(p))
        jid = str(rec["id"])
        if jid != Path(p).stem:raise ValueError('Queue filename identity differs')
        if jid not in states:
            s = {**rec, "id": jid, "status": "queued", "tries": 0, "history": []}
            s["gpu"] = rec.get("gpu") or C.get("default_gpu")
            save_state(s)
            states[jid] = s
            log(f"QUEUED {jid} {rec['name']} {rec['job_dir']} gpu={s['gpu']}")


def tag_of(job_dir):
    p = job_dir.rstrip("/").split("/")
    return "/".join(p[:-2]), p[-3]  # (<root>/out/<tag>, tag)


def jobs_row(s, done):
    new = not os.path.exists(JOBS_TSV)
    cols = ["id", "name", "job_dir", "status", "instance", "account", "gpu", "dph", "rc", "fetch_s", "elapsed_s", "attempts", "written",
            "demos", "successes", "start", "end"]
    with open(JOBS_TSV, "a") as f:
        if new:
            f.write("\t".join(cols) + "\n")
        row = {"id": s["id"], "name": s["name"], "job_dir": s["job_dir"], "status": s["status"], "instance": s.get("instance"),
               "account": s.get("account"), "gpu": done.get("gpu"), "dph": s.get("dph"), "rc": done.get("rc"), "fetch_s": done.get("fetch_s"),
               "elapsed_s": done.get("elapsed_s"), "attempts": done.get("attempts"), "written": done.get("written"), "demos": done.get("demos"),
               "successes": done.get("successes"), "start": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(done.get("start") or 0)),
               "end": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(done.get("end") or 0))}
        f.write("\t".join(str(row[k]) for k in cols) + "\n")


def real_sbatch():
    for d in os.environ.get("PATH", "").split(":"):
        p = os.path.join(d, "sbatch")
        if os.access(p, os.X_OK) and "vast_fast" not in p and not p.startswith(VJ):
            return p
    return "/usr/bin/sbatch"


def memory_jobs(states):
    """job_memory.sh of a Vast job, once its tag has no Vast job still open (the last one aggregates, as --if-batch-done does)."""
    for p in sorted(glob.glob(os.path.join(QUEUE, "*.memory.json"))):
        m = json.load(open(p))
        if m.get("status"):
            continue
        s = states.get(str(m["after"]))
        if not s or s["status"] not in ("synced", "failed"):
            continue
        _, tag = tag_of(s["job_dir"])
        root = s["job_dir"].split("/out/")[0]
        open_ = [x for x in states.values() if x["status"] not in ("synced", "failed", "cancelled") and x["job_dir"].startswith(f"{root}/out/{tag}/")]
        if open_:
            continue
        later = [q for q in glob.glob(os.path.join(QUEUE, "*.memory.json")) if q != p and json.load(open(q)).get("status") is None
                 and str(json.load(open(q))["after"]) in states and states[str(json.load(open(q))["after"])]["job_dir"].startswith(f"{root}/out/{tag}/")
                 and int(json.load(open(q))["after"]) > int(m["after"])]
        if later:
            m["status"] = "superseded"
        else:
            r = subprocess.run([real_sbatch()] + m["argv"], capture_output=True, text=True)
            m["status"] = f"submitted {r.stdout.strip()}" if r.returncode == 0 else f"failed {r.stderr.strip()[:100]}"
            log(f"MEMORY {tag}: {m['status']}")
        json.dump(m, open(p, "w"), indent=1)


# ---------------------------------------------------------------------------------------------------------------- loop


def rental_needs(queued, live, hbs, created, assigned_to, config, timestamp):
    """Uncovered queued job counts, separately for every requested GPU model."""
    needs = {}
    for job in uncovered_jobs(queued, live, hbs, created, assigned_to, timestamp):
        gpu = job.get("gpu") or config["default_gpu"]
        needs[gpu] = needs.get(gpu, 0) + 1
    return {gpu: count for gpu, count in needs.items() if count > 0}


def coordinator_progress(phase, **fields):
    path = os.path.join(VJ, "coordinator_progress.json")
    temporary = path + "." + uuid.uuid4().hex + ".tmp"
    with open(temporary, "w") as stream:
        json.dump({"observed_at_unix": now(), "coordinator_job_id": os.environ.get("SLURM_JOB_ID"),
                   "phase": phase, **fields}, stream, indent=2)
    os.replace(temporary, path)


def stage_bounded(rec, C):
    attempt = uuid.uuid4().hex[:12]
    base = os.path.join(VJ, "logs", f"stage_{rec['id']}_{attempt}")
    result = base + ".json"
    timeout = min(900, max(30, int(C.get("stage_timeout_s", 600))))
    command = [sys.executable, os.path.abspath(__file__), "stage", rec['id'], "--output", result,
               "--compression-threads", str(staging_compression_threads(C))]
    coordinator_progress("staging", job_id=rec['id'], log=base + ".log", timeout_seconds=timeout)
    try:
        run_bounded(command, base + ".log", timeout=timeout,
                    progress=lambda elapsed: coordinator_progress("staging", job_id=rec['id'],
                        log=base + ".log", elapsed_seconds=elapsed, timeout_seconds=timeout))
    except (subprocess.TimeoutExpired, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"staging failed or timed out; preserved log {base}.log: {exc}") from exc
    with open(result) as stream:
        return json.load(stream)


def result_fetch_bounded(row, reservation):
    log_path = os.path.join(VJ, 'logs', 'result_fetch_'+row['id']+'_'+row['token']+'.log')
    command = [sys.executable, os.path.abspath(__file__), 'result-fetch', '--reservation', reservation]
    run_bounded(command, log_path, timeout=row['timeout_seconds'], interval=10)
    with open(row['result']) as stream:
        return json.load(stream)


def publish_metrics(C, states, live, hbs, assigned_to, t, list_ok, tot, committed, hourly, phase,
                    eligibility=None):
    try:
        try:
            from .heartbeat_mirror import mirror_heartbeats
        except ImportError:
            from heartbeat_mirror import mirror_heartbeats
        mirror_heartbeats(os.path.join(VJ, 'worker_heartbeats'), hbs, now())
    except Exception as error:
        log('HEARTBEAT_MIRROR_UNAVAILABLE ' + type(error).__name__)
    staging = STAGER.reservations() if STAGER else {}
    n = {k: sum(1 for s in states.values() if s["status"] == k) for k in ("queued", "assigned", "synced", "failed")}
    # Reporting must not re-run every authorization and input hash. The old
    # observer did this twice per publication (four extra full scans per tick),
    # delaying ready assignments behind read-only metrics. Actual planning and
    # each assignment boundary still perform their normal authorization checks.
    eligible = (None if eligibility is None else sum(
        s['status'] == 'queued' and str(s['id']) in eligibility['job_ids'] for s in states.values()))
    # A recent mirror write does not make the original worker heartbeat fresh.
    # Preserve stopped providers and their assignments for recovery, but never
    # report their retained heartbeat phases as usable worker capacity.
    heartbeat_max_age_seconds = 300
    instance_metrics = []
    for iid, instance in live.items():
        heartbeat = hbs.get(iid) or {}
        heartbeat_time = heartbeat.get('time')
        age = (t - heartbeat_time if type(heartbeat_time) in (int, float)
               and heartbeat_time > 0 else None)
        fresh = age is not None and 0 <= age <= heartbeat_max_age_seconds
        instance_metrics.append({
            **{key: instance.get(key) for key in
               ('id', 'label', 'gpu_name', 'num_gpus', 'actual_status', 'dph_total')},
            'account': SHORT.get(instance.get('_key_name'), 'unknown'),
            'worker_phase': heartbeat.get('phase'),
            'heartbeat_age_seconds': age,
            'heartbeat_observation_status': ('fresh' if fresh else 'missing' if age is None
                                             else 'future' if age < 0 else 'stale'),
            'worker_observation_current': instance.get('actual_status') == 'running' and fresh,
            'heartbeat_running_jobs': sorted((heartbeat.get('running') or {}).keys()),
            'assigned_jobs': assigned_to.get(iid, []), 'staging_jobs': staging.get(iid, [])})
    current_workers = [instance for instance in instance_metrics if instance['worker_observation_current']]
    metrics = {"observed_at_unix": t, "assignments_observed_at_unix": now(), "coordinator_phase": phase, "account_listing_complete": list_ok, "running_gpus": sum(int(i.get("num_gpus") or 1) for i in live.values() if i.get("actual_status") == "running") if list_ok else None,
               "instances": instance_metrics,
               "result_fetch_scheduler": RESULT_FETCHER.metrics() if RESULT_FETCHER else None,
               "staging_scheduler": {**STAGER.metrics(now()),
                    'compression_threads_per_stage': staging_compression_threads(C)} if STAGER else None,
               'worker_heartbeat_max_age_seconds': heartbeat_max_age_seconds,
               "worker_booting_gpus":sum(int(i.get('num_gpus') or 1) for i in current_workers if i['worker_phase']=='boot') if list_ok else None,
               "worker_ready_gpus":sum(int(i.get('num_gpus') or 1) for i in current_workers if i['worker_phase'] in ('ready','run')) if list_ok else None,
               "worker_running_gpus":sum(int(i.get('num_gpus') or 1) for i in current_workers if i['assigned_jobs']) if list_ok else None,
               "worker_idle_ready_gpus":sum(int(i.get('num_gpus') or 1) for i in current_workers if i['worker_phase']=='ready' and not i['assigned_jobs'] and not i['staging_jobs'] and not i['heartbeat_running_jobs']) if list_ok else None,
               'worker_staging_gpus':sum(int(i.get('num_gpus') or 1) for i in current_workers if i['staging_jobs']) if list_ok else None,
               "worker_unknown_gpus":sum(int(i.get('num_gpus') or 1) for i in instance_metrics if i['actual_status']=='running' and (not i['worker_observation_current'] or i['worker_phase'] is None)) if list_ok else None,
               "rental_pending_gpus":sum(int(i.get('num_gpus') or 1) for i in instance_metrics if i['actual_status'] not in ('running','stopped','exited')) if list_ok else None,
               "preserved_stopped_instances": sorted(preserved_instances(F)),
               "max_instances":C['max_instances'],
               "dispatch_eligible_queued":eligible,
               "dispatch_excluded_queued":None if eligible is None else n['queued']-eligible,
               "dispatch_eligibility_observed_at_unix":None if eligibility is None else eligibility['observed_at'],
               "dispatch_eligibility_scope":"Last planning check, filtered by current queued state; not execution authorization",
               "spend_estimate_usd": tot, "future_compute_commitment_usd": committed, "transfer_reserve_usd": C.get("transfer_reserve_usd",300), "budget_usd": min(C["budget_usd"],3000), "hourly_usd": hourly, "queue": n}
    mp = os.path.join(VJ, "metrics.json")
    with open(mp + ".tmp", "w") as handle:
        json.dump(metrics, handle, indent=2)
    os.replace(mp + ".tmp", mp)


def tick(c, s3, state, states=None):
    global STAGER, RESULT_FETCHER
    C = cfg()
    protected = preserved_instances(F)
    if os.path.isdir(os.path.join(F, 'ik_exchange', 'manifests')):
        try:
            ik_transport.reconcile(F, s3, BUCKET, PREFIX, max_descriptors=4, deadline_seconds=10)
        except Exception as exc:
            log('EXACT_IK_MERGE_DEFERRED ' + type(exc).__name__)
    states = load_states() if states is None else states
    ingest(states, C)
    presence_started = now()
    inst, list_ok = my_instances(c)
    observe_rentals(VJ, inst, list_ok, presence_started, now())
    live = {str(i["id"]): i for i in inst}
    _, previously_live, _ = spend()
    hbs = read_heartbeats(s3, set(live) | set(previously_live))
    dones = set(k[:-5] for k in r2_list(s3, "done/"))
    t = now()
    tot, live_ids, created = spend()
    if STAGER is None:
        STAGER = StagingReservations(os.path.join(VJ,'state','staging'),os.environ.get('SLURM_JOB_ID','pid'+str(os.getpid())),
                                    staging_parallelism(C))
    else:
        STAGER.resize(staging_parallelism(C))
    def owner_alive(owner):
        if not str(owner).isdigit():return True  # unknown former local owner requires explicit reconciliation
        result=subprocess.run(['squeue','-h','-j',str(owner),'-o','%T'],capture_output=True,text=True,timeout=30,check=True)
        return bool(result.stdout.strip())
    if RESULT_FETCHER is None:
        RESULT_FETCHER = ResultFetchReservations(os.path.join(VJ, 'state', 'result_fetches'),
            os.environ.get('SLURM_JOB_ID', 'pid'+str(os.getpid())), 8)
    RESULT_FETCHER.recover(owner_alive)
    RESULT_FETCHER.poll()
    RESULT_FETCHER.reconcile_committed(states)
    def assignment(iid,jid):
        try:return json.loads(s3.get_object(Bucket=BUCKET,Key=f'{PREFIX}/assign/{iid}/{jid}.json')['Body'].read())
        except botocore.exceptions.ClientError as error:
            if str(error.response.get('Error',{}).get('Code')) in ('NoSuchKey','404','NotFound'):return None
            raise
    def reconcile_assignment(reservation):
        job=states[reservation['id']];worker=reservation['worker'];iid=reservation['instance']
        if job['status']=='assigned':
            if str(job['instance']) != iid:raise ValueError('assigned worker conflicts with staging reservation')
            return
        job.update(status='assigned',instance=iid,account=worker.get('_key_name'),machine_id=worker.get('machine_id'),
                   dph=worker.get('dph_total'),tries=int(reservation['job']['tries'])+1,
                   assigned=reservation.get('assignment_started_at',now()),gpu_actual=worker.get('gpu_name'),inputs=reservation['spec']['inputs'])
        attempt_id=reservation['spec'].get('attempt_id')
        job['cost_assignment_identity']=assignment_identity(job,attempt_id)
        job['cost_assignment_begin']=cost_record('assignment_begin',job['cost_assignment_identity'],
            instance=iid, job_id=job['id'], assignment_started_unix=job['assigned'],
            attempt_id=attempt_id, try_number=job['tries'], staging_started_unix=reservation.get('started_at'),
            scope=cost_scope(job), boundary='Assignment publish begin; worker runtime timestamps arrive separately.')
        save_state(job)
    STAGER.recover(owner_alive,assignment,reconcile_assignment)
    for reservation in list(STAGER.active().values()):
        if reservation['status']=='assigning' and reservation['instance'] in live:
            if not production_decision(states[reservation['id']], C)['allowed']:
                # A previous publish might have completed; do not republish or
                # discard its reservation without reconciliation.
                log('PRODUCTION_GATE blocked assignment recovery ' + reservation['id'])
                continue
            # Recovery retries the same job/worker/object only. Never free a
            # reservation while a previous remote PUT could still complete.
            r2_put(s3,f"assign/{reservation['instance']}/{reservation['id']}.json",json.dumps(reservation['spec']))
            reconcile_assignment(reservation);STAGER.assigned(reservation['id'])
    for failed in STAGER.poll():
        job=states[failed['id']]
        job.update(status='failed',error='stage: '+failed['error']);save_state(job)
        cost_record('staging_failure',job['id']+'_'+str(failed.get('started_at')),
            job_id=job['id'],instance=failed.get('instance'),scope=cost_scope(job),
            started_unix=failed.get('started_at'),failed_unix=failed.get('failed_at'),
            gpu_assignment_started=False,reason=failed.get('error'))
        log(f"STAGE_FAILED {job['id']}: {failed['error']}")
    if list_ok:  # an instance that destroyed itself: its rental ends now in the ledger (no later than the truth)
        for iid in live_ids:
            if iid not in live and t - created[iid]["time"] > 120:
                prior_heartbeat = hbs.get(iid) or {}
                detail = str(prior_heartbeat.get("detail") or "")
                if prior_heartbeat.get("phase") in ("unusable", "destroyed") and ("no NVIDIA" in detail or "GLX/EGL" in detail):
                    ledger("bad_machine", machine_id=created[iid].get("machine_id"), instance=iid, reason=detail)
                ledger("destroy", instance=iid, reason="gone (self-destroyed or removed)", ok=True, account=created[iid].get("account"))
                log(f"GONE {iid} [{SHORT.get(created[iid].get('account'), '?')}]")
        tot, live_ids, created = spend()

    # 1. finished jobs -> the cluster
    for s in states.values():
        if s["status"] in ("assigned",) and s["id"] in dones:
            d = r2_json(s3, f"done/{s['id']}.json") or {}
            async_fetch = False
            try:
                async_fetch = typed_preparation(s)
                if async_fetch:
                    added = RESULT_FETCHER.result(s, d)
                    if added is None:
                        RESULT_FETCHER.submit(s, d, result_fetch_bounded,
                                              timeout=C.get('result_fetch_timeout_s', 180))
                        continue  # Preserve assignment and claims until main-thread completion.
            except Exception as error:
                message = repr(error)
                if s.get('sync_error') != message:
                    log(f"SYNC_FAILED {s['id']}: {message}")
                    s['sync_error'] = message; save_state(s)
                continue
            attempt_evidence = retain_attempt(os.path.join(VJ,'state','attempts'),s,d)
            cost_record('assignment_end',s.get('cost_assignment_identity') or assignment_identity(s,d.get('attempt_id')),
                job_id=s['id'],instance=s.get('instance'),attempt_evidence=attempt_evidence,
                assignment_begin=s.get('cost_assignment_begin'),scope=cost_scope(s),end_kind='worker_done',
                observed_done_unix=t,worker_timestamps={key:d.get(key) for key in
                    ('fetch_started_unix','runtime_started_unix','runtime_ended_unix','upload_finished_unix','start','end')},
                rc=d.get('rc'),upload_error=d.get('upload_error'),
                legacy_assignment_begin_unavailable=not bool(s.get('cost_assignment_begin')))
            # a job that made no sampling attempt at all (inputs failed, Kit died at its first frame, ...) runs once more on another
            # instance before its result counts; the instance is kept (a code error fails everywhere: the second result is synced)
            # Replay has no sampling-attempt denominator. Its genuine native
            # gate (including a failed replay) must be synced without guessing
            # that a missing sampler counter means a launch failure.
            host_fail = production_decision(s, C)['allowed'] and (s.get('job_kind') not in ('physics_replay','native_dependency') and not d.get("attempts")) and not s.get("zero_retried") and s["tries"] < C["retries"]
            if host_fail:
                s["history"].append({"instance": s.get("instance"), "rc": d.get("rc"), "error": d.get("error"), "time": t, "zero_attempts": True,
                                     'attempt_evidence':attempt_evidence,'assigned_unix':s.get('assigned'),'done':d})
                ledger("suspect", machine_id=s.get("machine_id"), instance=s.get("instance"), job=s["id"], rc=d.get("rc"), error=d.get("error"))
                log(f"REQUEUE {s['id']}: rc={d.get('rc')} err={d.get('error')} with 0 attempts on {s.get('instance')}")
                s3.delete_object(Bucket=BUCKET, Key=f"{PREFIX}/done/{s['id']}.json")
                s3.delete_object(Bucket=BUCKET, Key=f"{PREFIX}/assign/{s.get('instance')}/{s['id']}.json")
                s.update(status="queued", instance=None, zero_retried=True, avoid=s.get("instance"))
                save_state(s)
                continue
            try:
                if not async_fetch:
                    added = sync_back(s3, s)
                s["status"] = "synced" if d.get("rc") == 0 and not d.get("upload_error") else "failed"
                s["done"] = d
                s['attempt_evidence'] = attempt_evidence
                s["cache_added"] = added
                log(f"DONE {s['id']} {s['name']} rc={d.get('rc')} att={d.get('attempts')} wr={d.get('written')} ok={d.get('successes')} "
                    f"run {d.get('elapsed_s')} s fetch {d.get('fetch_s')} s gpu={d.get('gpu')} cache+{added}")
                jobs_row(s, d)
            except Exception as e:  # noqa: BLE001
                log(f"SYNC_FAILED {s['id']}: {e!r}")
                s["sync_error"] = repr(e)
            save_state(s)
            if async_fetch and s['status'] in ('synced', 'failed'):
                RESULT_FETCHER.committed(s['id'])

    # 2. lost instances -> requeue their jobs
    for s in states.values():
        if s["status"] != "assigned" or s["id"] in dones:
            continue
        if RESULT_FETCHER.retains_assignment(s):
            continue  # Completed GPU work awaiting I/O must never become a GPU retry.
        iid = s.get("instance")
        if iid in protected:
            continue  # deliberately stopped, with on-disk evidence retained
        hb = hbs.get(iid) or {}
        gone = list_ok and iid not in live
        stale = hb.get("time") and t - hb["time"] > 1500
        if gone or stale:
            log(f"LOST {s['id']} on {iid} ({'gone' if gone else 'heartbeat stale'}); requeue")
            cost_record('assignment_end',s.get('cost_assignment_identity') or assignment_identity(s),
                job_id=s['id'],instance=iid,assignment_begin=s.get('cost_assignment_begin'),scope=cost_scope(s),
                end_kind='worker_lost',reason='gone' if gone else 'stale',last_heartbeat_unix=hb.get('time'),
                loss_observed_unix=t,exact_runtime_end_unavailable=True,
                legacy_assignment_begin_unavailable=not bool(s.get('cost_assignment_begin')))
            s["history"].append({"instance": iid, "lost": "gone" if gone else "stale", "time": t})
            s.update(status="queued" if s["tries"] < C["retries"] else "failed", instance=None)
            save_state(s)
            if stale and iid in live:
                destroy(c, iid, "heartbeat stale", live[iid].get("_key_name"))

    # Publish the fresh API observation before potentially slow staging. The
    # separate progress record reports staging liveness without falsifying the
    # timestamp of the provider observation.
    committed_observed = sum(float(created[iid]["dph_total"]) * max(0.0,
        float(created[iid].get("max_lifetime_s",21600)) - (t-created[iid]["time"])) / 3600 for iid in live_ids)
    assigned_observed = {}
    for record in states.values():
        if record["status"] == "assigned":
            assigned_observed.setdefault(record["instance"], []).append(record["id"])
    publish_metrics(C, states, live, hbs, assigned_observed, t, list_ok, tot, committed_observed,
                    sum(float(i.get("dph_total") or 0) for i in live.values()), "dispatching")
    # 3. Prefer explicitly prioritized ready production; otherwise preserve FIFO.
    # Only the coordinator may remove a never-assigned typed map from this queue.
    # Account submitters wait for its durable acknowledgement before submitting.
    if cluster_return.has_pending(F):
        remote_jobs = {Path(key).stem for key in r2_list(s3, 'assign/')}
        cluster_return.reconcile(F, states, STAGER.active(), hbs, remote_jobs, save_state)
        if JOB_STATE_CACHE is not None:JOB_STATE_CACHE.require_durable()
    coordinator_progress('eligibility_check', started_at=now())
    queued = dispatch_order(states.values(), C)
    eligibility = {'job_ids':{str(job['id']) for job in queued}, 'observed_at':now()}
    coordinator_progress('assigning_ready', eligible_queued=len(queued))
    for job in queued:
        job['compatible_gpus'] = compatible_gpus(job.get('gpu') or C['default_gpu'], C)
        job['exact_ik_exchange_required'] = bool(read_env(job['job_dir']).get('ISPATIALGYM_EXACT_IK_MANIFEST'))
    assigned_to = {}
    for s in states.values():
        if s["status"] == "assigned":
            assigned_to.setdefault(s["instance"], []).append(s["id"])
    resource_blocked = duplicate_worker_gpus(live, hbs)
    if resource_blocked:
        log('RESOURCE_GUARD blocked GPU identities: ' + ','.join(sorted(resource_blocked)))
    draining = drained_workers(live, C)
    routing_live = {iid:worker for iid,worker in live.items() if iid not in resource_blocked | draining | protected}
    allowed = {job['id'] for job in queued}
    for reservation in STAGER.ready():
        jid,iid=reservation['id'],reservation['instance'];s=states[jid]
        if jid not in allowed or iid not in routing_live or not production_decision(s, C)['allowed']:
            STAGER.release(jid,'job no longer eligible or worker unavailable');continue
        i=live[iid];spec=dict(reservation['spec'])
        candidate,_=assignment_plan([s],{iid:i},hbs,created,assigned_to,t)
        if not candidate:
            STAGER.release(jid,'worker no longer has a compatible free slot');continue
        spec.update({'label':i.get('label'),'machine_id':i.get('machine_id'),'account':i.get('_key_name'),'dph':i.get('dph_total'),
                     'attempt_id':s['id']+'_'+str(int(s['tries'])+1)+'_'+uuid.uuid4().hex[:12]})
        STAGER.assigning(jid,spec)  # durable write-ahead, then exact remote assignment
        r2_put(s3,f'assign/{iid}/{jid}.json',json.dumps(spec))
        reconcile_assignment(STAGER.records[jid]);STAGER.assigned(jid)
        assigned_to.setdefault(iid,[]).append(jid)
        log(f"ASSIGN {jid} {s['name']} -> {iid} ({i.get('gpu_name')}, {SHORT.get(i.get('_key_name'),'?')})")
    reserved_to={iid:list(jobs) for iid,jobs in assigned_to.items()}
    for iid,jobs in STAGER.reservations().items():reserved_to.setdefault(iid,[]).extend(jobs)
    queued=[job for job in queued if job['status']=='queued' and job['id'] not in STAGER.active()]
    planned,_=assignment_plan(queued,routing_live,hbs,created,reserved_to,t)
    for iid,worker,job in planned:
        if STAGER.submit(job,iid,worker,C,stage_bounded):
            queued.remove(job);reserved_to.setdefault(iid,[]).append(job['id'])
        else:break

    # 4. destroy: failed / stuck / idle instances, any instance over the price limit
    pending = len(queued)
    for iid, i in live.items():
        if iid in protected:
            continue
        hb = hbs.get(iid) or {}
        up = t - float(i.get("start_date") or t)
        st = i.get("actual_status")
        busy = bool(reserved_to.get(iid))
        model_pending = any(worker_can_run(job, iid, i, created.get(iid) or {}, t)
                            and (not job.get('exact_ik_exchange_required') or int(hb.get('exact_ik_exchange_version',0)) >= 1)
                            for job in queued)
        reason = None
        if drained_worker_finished(iid, draining, reserved_to, hb):
            reason = 'account funding drain; worker has no active or reserved jobs'
        elif hb.get("phase") in ("destroyed", "unusable", "stopped", "idle_exit") and t - hb.get("time", t) > 120:
            reason = f"worker {hb.get('phase')} {hb.get('detail', '')}"
        elif not hb and up > C["boot_max_s"]:
            reason = f"no heartbeat after {up / 60:.0f} min ({st})"
        elif hb.get("phase") == "boot" and up > C["boot_max_s"]:
            reason = f"boot not finished after {up / 60:.0f} min"
        elif t-float((created.get(iid) or {}).get("time",t)) >= float((created.get(iid) or {}).get("max_lifetime_s",21600)):
            reason = "rental lifetime cap"
        elif st == "exited" and up > 300:
            reason = "container exited"
        elif hb.get("phase") in ("ready",) and not busy and not model_pending:
            idle_since = state.setdefault("idle_since", {}).setdefault(iid, t)
            if t - idle_since > C["idle_destroy_s"]:
                reason = f"idle {(t - idle_since) / 60:.0f} min, no queued work fits GPU and remaining lease"
        elif hb.get("phase") == "boot" and pending == 0 and not busy and t - state.get("last_pending", t) > 900:
            reason = "booting with an empty queue for 15 min"
        if hb.get("phase") != "ready" or busy or model_pending:
            state.setdefault("idle_since", {}).pop(iid, None)
        if reason:
            if busy:
                for jid in assigned_to.get(iid,[]):
                    s = states[jid]
                    if s["status"] == "assigned":
                        if RESULT_FETCHER.retains_assignment(s):
                            continue
                        s["history"].append({"instance": iid, "lost": reason, "time": t})
                        s.update(status="queued" if s["tries"] < C["retries"] else "failed", instance=None)
                        save_state(s)
            destroy(c, iid, reason, i.get("_key_name"))
            if host_startup_failure(reason):
                ledger("bad_machine", machine_id=i.get("machine_id"), instance=iid, reason=reason)
    if pending:
        state["last_pending"] = t

    # 5. rent for the queued jobs that no instance (ready or booting) will take
    needs_by_gpu = rental_needs(queued, routing_live, hbs, created, reserved_to, C, t)
    need = sum(needs_by_gpu.values())
    fleet_ceiling = rental_ceiling(C, len(live), need)
    stop = os.path.exists(os.path.join(VJ, "STOP_LAUNCH"))
    committed = sum(float(created[iid]["dph_total"]) * max(0.0, float(created[iid].get("max_lifetime_s",21600)) - (t-created[iid]["time"])) / 3600 for iid in live_ids)
    queue_reserve = activated_queue_reserve(C, states)
    # Each live lease and each newly accepted offer is reserved once. Queue
    # planning cost overlaps those leases and is reported separately below.
    budget_left = min(float(C["budget_usd"]),3000.0) - float(C.get("transfer_reserve_usd",300.0)) - tot - committed
    hourly = sum(float(i.get("dph_total") or 0) for i in live.values())
    if need > 0 and list_ok and not stop and len(live) < fleet_ceiling and budget_left > max(10.0, 2 * hourly):
        credit_snapshot = read_credit_snapshot(os.path.join(VJ,'funding_state.json'), tuple(c))
        C['_rental_credit_remaining'] = {name:remaining_credit(credit_snapshot,name,events(),now(),
                                                               reserve=float(C.get('funding_reserve_per_account_usd',25)))
                                          for name in c}
        by_acct = {}
        for i in live.values():
            by_acct[i["_key_name"]] = by_acct.get(i["_key_name"], 0) + 1
        gpus_needed = sorted({gpu for requested in needs_by_gpu for gpu in compatible_gpus(requested,C)})
        offers = []
        for g in gpus_needed:
            offers += search(c, C, gpu=g)
        offers = ordered_gpu_offers(offers)
        n_new = 0
        reserved_live, reserved_created = dict(live), dict(created)
        market_reasons = Counter()
        for o in offers:
            if n_new >= C["launch_per_tick"] or need <= 0 or len(live) + n_new >= fleet_ceiling:
                break
            offer_gpu = o.get("gpu_name")
            demand_gpu = next((requested for requested in sorted(needs_by_gpu,key=lambda x:x != offer_gpu)
                               if needs_by_gpu[requested] > 0 and offer_gpu in compatible_gpus(requested,C)), None)
            if demand_gpu is None:
                continue
            reservation, reason = machine_reservation(o, reserved_live, hbs, reserved_created, C)
            if reason:
                market_reasons[reason] += 1
                continue
            o['_reservation'] = reservation
            reserve = float(o["dph_total"]) * float(C.get("max_worker_lifetime_s",21600)) / 3600
            if reserve > budget_left:
                continue
            if not any(C['_rental_credit_remaining'].get(name,0) >= reserve
                       for name in rental_accounts(tuple(c),C,by_acct)):
                market_reasons['insufficient_funded_credit'] += 1
                continue
            C['_rental_work_intent'] = rental_intent(queued, demand_gpu)
            iid = launch(c, o, C, by_acct)
            if iid:
                n_new += 1
                budget_left -= reserve
                needs_by_gpu[demand_gpu] = max(0, needs_by_gpu[demand_gpu] - o["_slots"])
                need = sum(needs_by_gpu.values())
                reserved_live[str(iid)] = {'machine_id':o.get('machine_id'), 'gpu_ids':reservation['gpu_ids']}
                reserved_created[str(iid)] = reservation
        market_path = os.path.join(VJ, 'market_decision.json')
        market_temporary = market_path + '.' + uuid.uuid4().hex + '.tmp'
        with open(market_temporary, 'w') as stream:
            json.dump({'observed_at_unix':now(), 'offers':len(offers), 'launched':n_new,
                       'remaining_demand':needs_by_gpu, 'rejections':dict(market_reasons),
                       'fleet_scaling_mode':C.get('fleet_scaling_mode','fixed'),
                       'fleet_bound_from_ready_demand':fleet_ceiling,
                       'budget_remaining_after_commitments':budget_left,
                       'queued_work_planning_upper_cost_usd':queue_reserve,
                       'queue_planning_cost_is_additional_commitment':False,
                       'funded_credit_remaining_by_account':{SHORT[name]:value for name,value in C['_rental_credit_remaining'].items()},
                       'max_contracts_per_machine':C.get('max_contracts_per_machine',4)}, stream, indent=2)
        os.replace(market_temporary, market_path)
    elif need > 0 and budget_left <= max(10.0, 2 * hourly):
        log(f"BUDGET: ${tot:.2f} of ${C['budget_usd']:.0f} spent, ${hourly:.2f}/h live: no new rentals")
    memory_jobs(states)
    n = {k: sum(1 for s in states.values() if s["status"] == k) for k in ("queued", "assigned", "synced", "failed")}
    phases = ", ".join(f"{iid}:{(hbs.get(iid) or {}).get('phase', '-')}" for iid in live)
    line = (f"queue {n} | instances {len(live)} ({phases}) | "
            f"spend_est ${tot:.2f} / ${C['budget_usd']:.0f}, committed ${committed:.2f}, transfer_reserve ${C.get('transfer_reserve_usd',300):.2f}, live ${hourly:.2f}/h")
    if line != state.get("last_line") or t - state.get("last_line_t", 0) > 600:
        log(line)
        state["last_line"], state["last_line_t"] = line, t
    open(os.path.join(VJ, "status.txt"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S ") + line + "\n")
    publish_metrics(C, states, live, hbs, assigned_to, t, list_ok, tot, committed, hourly, "complete", eligibility)
    coordinator_progress("idle", queue=n)


def status(c, s3):
    states = load_states()
    tot, live_ids, created = spend()
    inst, _ = my_instances(c)
    hbs = read_heartbeats(s3, {str(i['id']) for i in inst})
    print(f"spend ${tot:.2f} (ledger: {len(created)} created, {len(live_ids)} not destroyed)")
    for i in inst:
        hb = hbs.get(str(i["id"])) or {}
        print(f"  {i['id']} [{SHORT.get(i['_key_name'])}] {i.get('gpu_name')} {i.get('actual_status')} ${float(i.get('dph_total') or 0):.3f}/h "
              f"hb {hb.get('phase')} {hb.get('detail', '')} age {now() - hb.get('time', now()):.0f}s slots {hb.get('slots')} "
              f"running {list((hb.get('running') or {}).keys())} done {hb.get('done')} gpu_mem {hb.get('gpu_used_mb')} load {hb.get('load')} "
              f"mem {hb.get('mem_used_gb')}")
    for s in sorted(states.values(), key=lambda s: int(s["id"])):
        d = s.get("done") or {}
        print(f"  job {s['id']} {s['status']:8s} {s['name']} {s['job_dir'].split('/out/')[-1]} gpu={s.get('gpu')} inst={s.get('instance')} "
              f"tries={s['tries']}" + (f" rc={d.get('rc')} att={d.get('attempts')} wr={d.get('written')} ok={d.get('successes')} run={d.get('elapsed_s')}s"
                                       if d else ""))


def main():
    global JOB_STATE_CACHE
    global STAGE_COMPRESSION_THREADS
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("loop").add_argument("--every", type=int, default=60)
    sub.add_parser("status")
    o = sub.add_parser("offers")
    o.add_argument("--gpu", default=None)
    o.add_argument("--force", action="store_true")
    sg = sub.add_parser("stage")
    sg.add_argument("id")
    sg.add_argument("--output")
    sg.add_argument("--compression-threads", type=int, default=4)
    rf = sub.add_parser('result-fetch')
    rf.add_argument('--reservation', required=True)
    d = sub.add_parser("destroy")
    d.add_argument("instance")
    sub.add_parser("stop-all")
    a = ap.parse_args()
    c = vclients()
    s3 = r2()
    if a.cmd == "loop":
        state = {}
        sp = os.path.join(VJ, "state", "coordinator.json")
        if os.path.exists(sp):
            state = json.load(open(sp))
        lock = os.path.join(VJ, "state", "coordinator.lock")
        mine = f"{os.environ.get('SLURM_JOB_ID', 'pid' + str(os.getpid()))}"
        if os.path.exists(lock):
            other = open(lock).read().strip()
            r = subprocess.run(["squeue", "-h", "-j", other, "-o", "%T"], capture_output=True, text=True)
            if other != mine and r.stdout.strip() in ("RUNNING", "PENDING"):
                raise SystemExit(f"another coordinator is running: slurm job {other}")
        open(lock, "w").write(mine)
        # Audited writer boundary: queue clients never write STATE. Return
        # callbacks receive this same mapping and save_state. Disk is loaded
        # once after ownership and remains the authoritative restart baseline.
        JOB_STATE_CACHE = CoordinatorStateCache(load_states)
        log(f"COORDINATOR {mine} job={JOB_ID} prefix={JOB_S3} config={cfg()}")
        while True:
            try:
                tick(c, s3, state, JOB_STATE_CACHE.get())
            except Exception as e:  # noqa: BLE001
                JOB_STATE_CACHE.invalidate()
                import traceback
                log(f"TICK_FAILED {e!r}\n{traceback.format_exc()}")
                coordinator_progress("error", error=repr(e))
            json.dump(state, open(sp + ".tmp", "w"))
            os.replace(sp + ".tmp", sp)
            time.sleep(a.every)
    if a.cmd == 'result-fetch':
        row = json.load(open(a.reservation))
        job = json.load(open(os.path.join(STATE, row['id']+'.json')))
        execute_reserved_fetch(a.reservation, job, lambda value: sync_back(s3, value))
        return
    if a.cmd == "status":
        return status(c, s3)
    if a.cmd == "offers":
        C = cfg()
        if a.gpu:
            C["gpus"] = [a.gpu]
        for o in search(c, C, gpu=a.gpu, force=a.force)[:25]:
            print(fmt_offer(o))
        return 0
    if a.cmd == "stage":
        STAGE_COMPRESSION_THREADS = staging_compression_threads({'stage_compression_threads': a.compression_threads})
        result = stage(s3, staging_record(STATE, a.id), {})
        if a.output:
            with open(a.output, "x") as stream:
                json.dump(result, stream, indent=2)
        else:
            print(json.dumps(result, indent=1))
        return 0
    if a.cmd == "destroy":
        inst, _ = my_instances(c)
        k = next((i["_key_name"] for i in inst if str(i["id"]) == a.instance), None)
        if k is None:
            raise SystemExit(f"{a.instance} is not one of this job's instances")
        return 0 if destroy(c, int(a.instance), "manual", k) else 1
    if a.cmd == "stop-all":
        open(os.path.join(VJ, "STOP_LAUNCH"), "w").write(time.strftime("%c") + "\n")
        r2_put(s3, "ctl/stop", "stop")
        print("STOP_LAUNCH + ctl/stop written")
        return 0


if __name__ == "__main__":
    sys.exit(main())
