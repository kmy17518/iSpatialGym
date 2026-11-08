#!/usr/bin/env python3
"""trainfast: submit fast-flow jobs for an alias (train85's parameters, this campaign's roots) - production collection AND alias
validation (owner 2026-11-05: one pipeline; validation differs only in its output root, ledger / index range and job tag).

    submit_fast.py submit <key> <n> --flow base|fast --code <frozen snapshot> [--gres titanrtx|l40s] [--acct vision] [--part svl]
                   [--kcollect K] [--sample-limit-s S] [--time-s T] [--tag name] [--customs template|trunk] [--extra-env NAME=VALUE ...]
                   [--root DIR] [--ledger TSV] [--base-index N] [--job-prefix isgf_] [--registry TSV] [--grasps legacy|v2] [--dry-run]
    submit_fast.py recollect <source job dir> --code <frozen snapshot> [--name N] [--iids IID ...] [--root DIR] [--job-prefix P] ...
    submit_fast.py template <alias> <key> --code <frozen snapshot> --seed-index K [--scene S] [--support SUP] [--support-row ROW] [--n N]
                   [--budget-s B] [--root DIR] [--job-prefix P] ...   (sample_isg at an eval seed: the template for register_alias.py)
    submit_fast.py params <key> [<n>]
    submit_fast.py keys [--registry TSV]
    submit_fast.py running

Aliases: train85's counted table (manifests/counted.tsv + counted_costs.tsv: template copy, eval index, limits) plus the trunk's
registry (scripts/trainfast/registry.tsv of the --code snapshot, or --registry): aliases sampled only with job_val.sh KIND=eval, or
re-bound, with a template made by sample_isg at an eval seed (see the registry's header). A registry row wins over a counted row of
the same key.

Train indices come from the ledger (<root>/manifests/allocations.tsv) starting at --base-index per alias (100000 = trainfast's; the
oracle-fix validation uses 300000), so they never collide with isg-1020's or with another root's. The job scripts and their helpers
come from the --code snapshot (<code>/scripts/trainfast, ISGF_SCRIPTS in job.env) when it has them, else from the campaign's live
$F/scripts. Goal occlusion (owner 2026-11-05) is screened in the sampler for mental_rotation and disembedding (ISGF_GOAL_OCCLUSION=on)
and in the replay stage for every other family.
"""
import argparse
import csv
import fcntl
import glob
import importlib.util
import json
import os
import shlex
import subprocess
import sys
import time

F = "/vision/u/minyeong/projects/isg_home/trainfast"
T85 = "/vision/u/minyeong/projects/isg_home/train85"
LEDGER_COLS = ["submitted", "key", "alias", "start", "n", "job_tag", "job_dir", "flow", "code", "jobid", "acct", "part", "gres", "time_s",
               "sample_limit_s", "max_attempts", "demo_limit_s", "kcollect", "tag"]
BASE_INDEX = 100000
#: the families whose sampler screens goal occlusion (owner 2026-11-05: they keep a camera under SAMPLER_CAMERAS; train85 measured
#: 2.0-2.1 s per instance and 6.1-6.3 % rejected); every other family is screened in the replay stage
SAMPLER_OCCLUSION_FAMILIES = ("mental_rotation", "disembedding")
#: i7-blockdesign 2026-11-08 (owner 5m.1): the oracle's action-step budget per alias-key prefix, for the COLLECTORS only (run_isg --max-steps
#: through ISGF_COLLECT_EXTRA in job.env; the shared eval/instances DEFAULT_MAX_STEPS 3000 stays). Block design: every success so far took
#: 3267-4149 steps (the bd_fix 6000-step recollect 4 / 4 at 3267-3512; eval85 3520 / 4149), so at 3000 the alias writes demos that never succeed
COLLECTOR_MAX_STEPS = {"configuration_matching-block_design": 6000}
EXCLUDE = "visionlab-dgx1,viscam7,viscam11,viscam12,viscam-hgx-1,viscam-hgx-2,viscam14,viscam15,viscam16,svl9"

spec = importlib.util.spec_from_file_location("submit_train", os.path.join(T85, "scripts", "submit_train.py"))
st = importlib.util.module_from_spec(spec)
spec.loader.exec_module(st)



#: 2026-11-01 (REPEAT_MEASURED, owner): train85 per-alias eval indexes rebuilt with the measured entries (scripts/build_eval_index_v2.py);
#: an alias without a rebuilt index keeps train85's (the old repeat rule then applies to it)
EVAL_INDEX_V2 = os.path.join(F, "eval_index_v2", "per_alias")

#: of-v2maps 2026-11-06 (--grasps v2, fast_env.sh ISGF_GRASPS): the campaign-owned grasp library v2, reach maps and height anchors
#: (docs/GRASP_BANDS.md "Fast-flow pipeline"); written into job.env, so the collectors and a later recollect use the same library
OFH = "/vision/u/minyeong/projects/isg_home/oracle_fix"
V2_ROOT = os.path.join(OFH, "v2maps_20261106")  # glib/ (the library), maps/ (the reach maps), anchors.json, logs/
GRASPS_V2 = {"ISPATIALGYM_GRASP_LIBRARY_V2": os.path.join(V2_ROOT, "glib"), "ISPATIALGYM_REACH_MAPS": os.path.join(V2_ROOT, "maps"),
             "ISPATIALGYM_REACH_MAP_ANCHORS": os.path.join(V2_ROOT, "anchors.json")}
#: i7-plateqtf2 (owner decision 2026-11-08 02:35 item 5): the campaign's merged reach-map overlay (v2/map_tilt.py; <root>/<map key>/v<N>.json,
#: releases <root>/releases/r<M>.json written by every offline merge). --map-overlay <root> pins the root's LATEST RELEASE into job.env
#: (ISPATIALGYM_MAP_OVERLAY + ISPATIALGYM_MAP_OVERLAY_RELEASE): the job reads exactly those versions, whatever is merged meanwhile.
MAP_OVERLAY_ROOT = os.path.join(OFH, "map_overlay")
MAP_OVERLAY_DEFAULT = MAP_OVERLAY_ROOT  # i7-plateqtf2: ON by default (owner item 5, separate commit); --map-overlay off = none


def overlay_release(root):
    """The newest release number under an overlay root (0: none - the job then reads no overlay). Same rule as map_tilt.latest_release
    (this script runs on the login node's python, without the package)."""
    import glob as _glob

    ns = [int(os.path.basename(p)[1:-5]) for p in _glob.glob(os.path.join(root, "releases", "r*.json")) if os.path.basename(p)[1:-5].isdigit()]
    return max(ns) if ns else 0


#: of-v2maps 2026-11-07 (owner: sampler jobs on shiyuc's GPUs too; shiyuc writes ONLY under this folder): submit --shiyuc - run as
#: shiyuc@svl11, every write of the job (root: out / slurm / manifests, the sampling + exact-IK cache, the asset cache, the per-job HOME
#: fallback, the kit template / texture stores, the EXP_PATH overlay, the predemo root) under SY; the read stores minyeong keeps mode 600
#: come from staged copies under SY/stores; band sharing stays off; indices start at SHIYUC_BASE_INDEX (minyeong's ledgers: 100000 /
#: 300000), so no index is used by both accounts.
SY = "/vision/u/shiyuc/minyeong"
SHIYUC_BASE_INDEX = 600000


def shiyuc_env(a):
    """The job.env entries of shiyuc mode (raises when a write path would leave SY or an input is unreadable)."""
    root = os.path.abspath(a.root)
    if not root.startswith(SY + "/"):
        raise SystemExit(f"--shiyuc: --root {root} is not under {SY}")
    if a.base_index < SHIYUC_BASE_INDEX:
        raise SystemExit(f"--shiyuc: --base-index {a.base_index} < {SHIYUC_BASE_INDEX} (minyeong's ranges)")
    if a.ledger and not os.path.abspath(a.ledger).startswith(SY + "/"):
        raise SystemExit(f"--shiyuc: --ledger {a.ledger} is not under {SY}")
    # the read stores (lab band store, eval85 bands, train85 bands, reach_volume) are world-readable (probe 2026-11-07): read in place, as
    # minyeong's jobs do; env_fast.sh / job_fast.sh take ISGF_LAB_BAND_STORE / ISGF_EVAL_BANDS / ISGF_REACH_VOLUME_SRC / ISGF_T85_BANDS
    # (--extra-env) for a staged copy should one become unreadable
    env = {"ISGF_SAMPLING_CACHE": os.path.join(SY, "cache", "sampling"), "ISGF_ASSET_CACHE": os.path.join(SY, "asset_cache"),
           "ISGF_JB_NFS": os.path.join(SY, "jobs"), "KIT_TPL_NFS_ROOT": os.path.join(SY, "kit_templates"),
           "KIT_TEX_STORE": os.path.join(SY, "kit_textures"), "ISGF_EXP_PATH": os.path.join(SY, "isaacsim_overlay", "apps"),
           "ISPATIALGYM_FAST_BAND_SHARE": "0", "ISGF_PREDEMO_ROOT": root, "ISGF_GIT_SAFE_ALL": "1"}
    for k in ("ISGF_SAMPLING_CACHE", "ISGF_ASSET_CACHE", "ISGF_JB_NFS", "KIT_TPL_NFS_ROOT", "KIT_TEX_STORE", "ISGF_PREDEMO_ROOT"):
        assert env[k].startswith(SY + "/"), k
    if not os.path.isdir(env["ISGF_EXP_PATH"]):
        raise SystemExit(f"--shiyuc: no EXP_PATH overlay at {env['ISGF_EXP_PATH']}")
    return env


#: i7-memory2 (owner 2026-11-08 2.2 / 2.4): the store of each memory mode under --root (alias_memory.STORE_DIRS) - separate stores
#: (<root>/alias_memory stays the legacy store the pre-mode snapshots aggregate every demo into: never pinned by a mode)
MEMORY_STORES = {"production": "alias_memory_production", "validation": "alias_memory_validation"}


def _version_mode(path):
    try:
        meta = json.load(open(os.path.join(path, "meta.json")))
    except (OSError, ValueError):
        return "production"
    return str(meta.get("mode") or "legacy")


def alias_memory_pin(a):
    """i7-memory (owner 2026-11-08 part C): the outcome-memory VERSION the job reads, pinned in job.env (ISPATIALGYM_ALIAS_MEMORY):
    --alias-memory auto (default) = the mode's store's LATEST when one exists, else off; off; or a version dir / a memory root (its LATEST).
    i7-memory2: --memory-mode production (default; store <root>/alias_memory_production, starts empty) | validation (store <root>/alias_memory_validation,
    seeded from the replay; it must exist - its first version is built by `alias_memory aggregate --root R --mode validation`); a pinned
    version of the other mode's store is refused."""
    mode = getattr(a, "memory_mode", "production") or "production"
    v = getattr(a, "alias_memory", "auto") or "auto"
    if v == "off":
        return "off"
    root = os.path.join(os.path.abspath(a.root), MEMORY_STORES[mode]) if v == "auto" else os.path.abspath(v)
    if os.path.exists(os.path.join(root, "meta.json")):
        ver_dir = root
    else:
        try:
            ver = open(os.path.join(root, "LATEST")).read().strip()
        except OSError:
            if v != "auto":
                raise SystemExit(f"--alias-memory {v}: neither a version (meta.json) nor a memory root (LATEST)")
            if mode == "validation":
                raise SystemExit(f"--memory-mode validation: no seeded store at {root} - build it first (CPU): PYTHONPATH=<code> python -m "
                                 f"ispatialgym.sampling.alias_memory aggregate --root {os.path.abspath(a.root)} --mode validation --no-gpu")
            return "off"
        ver_dir = os.path.join(root, ver) if ver and os.path.isdir(os.path.join(root, ver)) else None
        if ver_dir is None:
            return "off"
    if _version_mode(ver_dir) != mode:
        raise SystemExit(f"--memory-mode {mode}: {ver_dir} is a {_version_mode(ver_dir)} memory version (the stores are separate)")
    return ver_dir


def memory_job(a, gpu_jobid, code):
    """The batch's aggregation job (job_memory.sh: CPU, afterany the GPU job; it aggregates only when the tag's last job is done)."""
    script = os.path.join(scripts_of(code), "job_memory.sh")
    if not os.path.exists(script):
        return None
    cmd = ["sbatch", "--parsable", "-J", f"{a.job_prefix}mem", "-A", a.acct, "-p", a.part, "--time=1:00:00", f"--exclude={EXCLUDE}", "-c", "2",
           "--mem=8G", f"--dependency=afterany:{gpu_jobid}", "-o", os.path.join(a.root, "slurm", "%x_%j.log"), script, os.path.abspath(a.root),
           a.tag or a.flow, os.path.abspath(code), getattr(a, "memory_mode", "production") or "production"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    return res.stdout.strip().split(";")[0] if res.returncode == 0 else f"failed ({res.stderr.strip()[:80]})"


def eval_index_of(key):
    p = os.path.join(EVAL_INDEX_V2, key + ".json")
    return p if os.path.exists(p) else os.path.join(st.EVAL_INDEX_DIR, key + ".json")


def collector_max_steps(key):
    """The collector step budget of alias ``key`` (``COLLECTOR_MAX_STEPS``, the longest matching prefix), else None (run_isg's default)."""
    hits = [p for p in COLLECTOR_MAX_STEPS if key.startswith(p)]
    return COLLECTOR_MAX_STEPS[max(hits, key=len)] if hits else None


def with_collector_max_steps(env, key):
    """5m.1: ``--max-steps N`` appended to ``env["ISGF_COLLECT_EXTRA"]`` for an alias with a collector budget, unless the job already
    passes its own ``--max-steps`` (an --extra-env ISGF_COLLECT_EXTRA wins). Returns the budget applied, else None."""
    n = collector_max_steps(key)
    extra = str(env.get("ISGF_COLLECT_EXTRA", "") or "")
    if n is None or "--max-steps" in extra.split():
        return None
    env["ISGF_COLLECT_EXTRA"] = (extra + " " if extra else "") + f"--max-steps {int(n)}"
    return n


def registry_path(a):
    if getattr(a, "registry", None):
        return a.registry
    code = getattr(a, "code", None)
    p = os.path.join(code, "scripts", "trainfast", "registry.tsv") if code else ""
    return p if p and os.path.exists(p) else os.path.join(os.path.dirname(os.path.abspath(__file__)), "registry.tsv")


def load_registry(path):
    """registry.tsv rows as counted rows (the columns st.params / submit read)."""
    if not path or not os.path.exists(path):
        return {}
    out = {}
    for r in csv.DictReader((l for l in open(path) if not l.startswith("#")), delimiter="\t"):
        out[r["key"]] = {**r, "template_dir": r["counted_template"], "run_dir": r.get("source_run", ""), "isg_root": r.get("source_code", ""),
                         "c_attempts": r["attempts"], "c_attempts_s": r["attempts_s"], "c_world_load_s": r["world_load_s"],
                         "c_setup_s": r["setup_s"], "c_bands_missing_s": r.get("bands_missing_s") or "0", "c_demo_load_s": r["demo_load_s"],
                         "_registry": path}
    return out


def aliases(a):
    counted = st.load_counted()
    return {**counted, **load_registry(registry_path(a))}


def scripts_of(code):
    s = os.path.join(os.path.abspath(code), "scripts", "trainfast")
    return s if os.path.exists(os.path.join(s, "job_fast.sh")) else os.path.join(F, "scripts")


def ledger_of(a):
    return a.ledger or os.path.join(a.root, "manifests", "allocations.tsv")


def lock_ledger(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    lock = open(path + ".lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX)  # best effort: this NFS returns EIO for flock
    except OSError:
        pass
    return lock


def unlock(lock):
    try:
        fcntl.flock(lock, fcntl.LOCK_UN)
    except OSError:
        pass


def sbatch(a, name, time_s, script, job_dir):
    if a.cpus is None:  # COLLECTOR_PIN needs (K + 1) x 3 CPUs or it stays off (cpu_split.py)
        a.cpus = max(8, 3 * (int(a.kcollect) + 1))
    gres = f"--gres=gpu:{a.gres}:1" if a.gres else "--gres=gpu:1"
    return (["sbatch", "--parsable", "-J", name, "-A", a.acct, "-p", a.part, f"--time={st.hms(time_s)}", f"--exclude={EXCLUDE}", gres, "-c", str(a.cpus),
             f"--mem={a.mem}", "-o", os.path.join(a.root, "slurm", "%x_%j.log"), script, job_dir], gres)


def submit(a):
    table = aliases(a)
    if a.key not in table:
        raise SystemExit(f"unknown key {a.key} (neither train85's counted table nor the registry {registry_path(a)})")
    c = table[a.key]
    reg = "_registry" in c
    ledger = ledger_of(a)
    lock = lock_ledger(ledger)
    try:
        rows = list(csv.DictReader(open(ledger), delimiter="\t")) if os.path.exists(ledger) else []
        start = max([a.base_index] + [int(r["start"]) + int(r["n"]) for r in rows if r["key"] == a.key])
        n = int(a.n)
        p = st.params(c, n, a.sample_limit_s, a.time_s)
        tag = f"g{start}-{start + n - 1}"
        job_dir = os.path.join(a.root, "out", a.tag or a.flow, a.key, tag)
        if os.path.exists(job_dir):
            raise SystemExit(f"{job_dir} exists")
        tmpl = c["template"] if reg else os.path.join(T85, "templates", a.key, "inst", c["alias"], c["template_id"])
        if not os.path.exists(os.path.join(tmpl, "template_meta.json")):
            raise SystemExit(f"no template copy for {a.key}: {tmpl}")
        fam = c["family"] if reg else st.family_of(c["subtask"], c["tag"])
        spa = int(c.get("start_per_attempt") or 0) if reg else (1 if a.key in st.affected_keys() else 0)
        # one trunk (2026-11-06): the snapshot's isg_customs.json for every alias (the validated settings live there; a counted
        # template's pre-rename variant resolves through customs.entry_key); --customs template = the counted run's copy
        customs = a.customs or (c.get("customs") if reg else None) or "trunk"  # a counted row's "customs" column is its run's path
        env = {"ALIAS": c["alias"], "ALIAS_KEY": a.key, "PICK": c["pick"], "TAG": c["tag"], "COUNTED_RUN": c["run_dir"],
               "ISGE_CODE": os.path.abspath(a.code), "COUNTED_SNAPSHOT": c["isg_root"], "TEMPLATE": tmpl, "TEMPLATE_ID": c["template_id"],
               "COUNTED_TEMPLATE": c["template_dir"], "EVAL_INDEX": c["eval_index"] if reg else eval_index_of(a.key), "FAMILY": fam,
               "KEEP_SCENE": "--keep-template-scene" if fam in ("object_scaling", "mental_rotation") else "",
               "TRUNK_CHECK": "idle-arm" if c["tag"] == "iors_large" else "", "START": start, "N": n, "JOB_TAG": tag, "KIND": a.flow,
               "FLOW": a.flow, "KCOLLECT": a.kcollect, **{k: p[k] for k in ("MAX_ATTEMPTS", "DEMO_LIMIT_S", "SAMPLE_LIMIT_S", "TIME_LIMIT_S")},
               "START_PER_ATTEMPT": spa, "STILL_EXIT": 1, "CODE_VARIANT": a.flow, "CUSTOMS": customs,
               "ISGF_SCRIPTS": scripts_of(a.code), "ISGF_ROOT": os.path.abspath(a.root)}
        if fam in SAMPLER_OCCLUSION_FAMILIES:
            env["ISGF_GOAL_OCCLUSION"] = "on"
        if reg and c.get("band_stores"):
            env["BAND_STORES"] = c["band_stores"]
        for kv in (c.get("extra_env") or "").split() if reg else []:  # the registry's per-alias switches (validated settings)
            k, v = kv.split("=", 1)
            env[k] = v
        # of-v2maps: the grasp / band source; v2 replaces a registry row's own library (the umbrella's entry is in the campaign library)
        env["ISGF_GRASPS"] = "v2" if a.grasps == "v2only" else a.grasps
        if a.grasps in ("v2", "v2only"):
            env.update(GRASPS_V2)
            if a.v2_root:  # of-v2maps 2026-11-07: another campaign data root (glib/, maps/, anchors.json)
                env.update({"ISPATIALGYM_GRASP_LIBRARY_V2": os.path.join(a.v2_root, "glib"), "ISPATIALGYM_REACH_MAPS": os.path.join(a.v2_root, "maps"),
                            "ISPATIALGYM_REACH_MAP_ANCHORS": os.path.join(a.v2_root, "anchors.json")})
        if a.grasps == "v2only":
            # of-v2maps 2026-11-07 (owner: no legacy band, v2 grasps + v2 reach maps only): every legacy band load / compute is refused
            # (reach_bands.block_legacy_band -> legacy_band_blocked), a map miss rejects (missing_map), the support policy is on from the start
            env.update({"ISPATIALGYM_V2_ONLY": "1", "ISPATIALGYM_REACH_MAPS_STRICT": "1", "ISPATIALGYM_SUPPORT_POLICY": "grasp"})
        # i7-memory (owner 2026-11-08 part C, unattended): the per-alias outcome memory version this job's sampler reads (train only)
        env["ISPATIALGYM_ALIAS_MEMORY"] = alias_memory_pin(a)
        env["ISPATIALGYM_ALIAS_MEMORY_ALIAS"] = a.key
        # i7-memory2 (owner 2.2 / 2.4): the job's memory mode - the store it reads (pinned above) and the one its demos are aggregated into
        env["ISPATIALGYM_ALIAS_MEMORY_MODE"] = getattr(a, "memory_mode", "production") or "production"
        ov_root = getattr(a, "map_overlay", None) or "off"
        if a.grasps in ("v2", "v2only") and ov_root != "off":
            # i7-plateqtf2 item 5: the reach maps' merged overlay, pinned to the root's latest release at submission (job.env is the record)
            if not os.path.isdir(ov_root):
                raise SystemExit(f"--map-overlay {ov_root}: no such directory (or --map-overlay off)")
            env.update({"ISPATIALGYM_MAP_OVERLAY": os.path.abspath(ov_root), "ISPATIALGYM_MAP_OVERLAY_RELEASE": str(overlay_release(ov_root))})
        for kv in a.extra_env or []:
            k, v = kv.split("=", 1)
            env[k] = v
        with_collector_max_steps(env, a.key)  # 5m.1: block design's collectors at 6000 steps (job.env is the record)
        if getattr(a, "shiyuc", False):
            env.update(shiyuc_env(a))  # after --extra-env: no override moves a write out of SY
            for k in ("ISPATIALGYM_GRASP_LIBRARY_V2", "ISPATIALGYM_REACH_MAPS", "ISPATIALGYM_REACH_MAP_ANCHORS", "TEMPLATE", "EVAL_INDEX", "ISPATIALGYM_MAP_OVERLAY"):
                if env.get(k) and not os.access(env[k], os.R_OK):
                    raise SystemExit(f"--shiyuc: {k}={env[k]} is not readable by this account (stage a copy under {SY})")
            for k in ("ISPATIALGYM_GRASP_LIBRARY_V2", "ISPATIALGYM_REACH_MAPS"):  # every file of the v2 data (minyeong's are mode 600)
                bad = [os.path.join(d, f) for d, _, fs in os.walk(env.get(k) or "/nonexistent") for f in fs if not os.access(os.path.join(d, f), os.R_OK)]
                if bad:
                    raise SystemExit(f"--shiyuc: {len(bad)} unreadable files under {env[k]} (e.g. {bad[0]}): pass --v2-root <the staged copy under {SY}>")
        script = os.path.join(env["ISGF_SCRIPTS"], f"job_{a.flow}.sh")
        cmd, gres = sbatch(a, f"{a.job_prefix}{st.short_of(a.key)}", p["TIME_LIMIT_S"], script, job_dir)
        if a.dry_run:
            print("DRY", tag, " ".join(shlex.quote(x) for x in cmd))
            print("\n".join(f"  {k}={v}" for k, v in env.items()))
            return 0
        os.makedirs(job_dir)
        os.makedirs(os.path.join(a.root, "slurm"), exist_ok=True)
        with open(os.path.join(job_dir, "job.env"), "w") as fh:
            for k, v in env.items():
                fh.write(f"{k}={shlex.quote(str(v))}\n")
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            os.rename(job_dir, job_dir + ".failed_submit")
            raise SystemExit(f"sbatch failed: {res.stderr.strip()}")
        jobid = res.stdout.strip().split(";")[0]
        row = {"submitted": time.strftime("%Y-%m-%dT%H:%M:%S"), "key": a.key, "alias": c["alias"], "start": start, "n": n, "job_tag": tag,
               "job_dir": job_dir, "flow": a.flow, "code": env["ISGE_CODE"], "jobid": jobid, "acct": a.acct, "part": a.part, "gres": a.gres or "any",
               "time_s": p["TIME_LIMIT_S"], "sample_limit_s": p["SAMPLE_LIMIT_S"], "max_attempts": p["MAX_ATTEMPTS"], "demo_limit_s": p["DEMO_LIMIT_S"],
               "kcollect": a.kcollect, "tag": a.tag or a.flow}
        new = not os.path.exists(ledger)
        with open(ledger, "a") as fh:
            if new:
                fh.write("\t".join(LEDGER_COLS) + "\n")
            fh.write("\t".join(str(row[k]) for k in LEDGER_COLS) + "\n")
        mem_id = memory_job(a, jobid, a.code) if getattr(a, "memory_aggregate", True) else None
        print(f"SUBMITTED {jobid} {a.flow} {a.key} {tag} {a.acct}/{a.part} {gres} time={st.hms(p['TIME_LIMIT_S'])} sample_limit={p['SAMPLE_LIMIT_S']} "
              f"max_attempts={p['MAX_ATTEMPTS']} demo_limit={p['DEMO_LIMIT_S']} spa={env['START_PER_ATTEMPT']} k={a.kcollect} customs={customs} "
              f"occl={env.get('ISGF_GOAL_OCCLUSION', 'off')} grasps={a.grasps} root={a.root} alias_memory={env['ISPATIALGYM_ALIAS_MEMORY']} memory_mode={env['ISPATIALGYM_ALIAS_MEMORY_MODE']}"
              + (f" collect_extra=\"{env['ISGF_COLLECT_EXTRA']}\"" if env.get("ISGF_COLLECT_EXTRA") else "")
              + (f" map_overlay={env['ISPATIALGYM_MAP_OVERLAY']}@r{env.get('ISPATIALGYM_MAP_OVERLAY_RELEASE', '?')}" if env.get("ISPATIALGYM_MAP_OVERLAY") else "")
              + (f" memory_job={mem_id}" if mem_id else ""))
    finally:
        unlock(lock)
    return 0


def recollect(a):
    """A collector-only job (job_recollect.sh) on a finished job's sampled instances: prep_recollect.py restores the TROs the sampler
    wrote (the source collector's goal override reverted); --iids keeps only those instances (a skip claim for every other one)."""
    src = os.path.abspath(a.src)
    env0 = dict(l.rstrip("\n").split("=", 1) for l in open(os.path.join(src, "job.env")) if "=" in l and not l.startswith("#"))
    key = shlex.split(env0["ALIAS_KEY"])[0]
    name = a.name or time.strftime("r%Y%m%d_%H%M%S")
    job_dir = os.path.join(a.root, "out", "recollect", key, f"{os.path.basename(src)}__{name}")
    if os.path.exists(job_dir):
        raise SystemExit(f"{job_dir} exists")
    scripts = scripts_of(a.code)
    cmd0 = [sys.executable, os.path.join(scripts, "prep_recollect.py"), src, job_dir]
    if a.dry_run:
        print("DRY", " ".join(cmd0))
        return 0
    subprocess.run(cmd0, check=True)
    over = {"ISGE_CODE": os.path.abspath(a.code), "ISGF_SCRIPTS": scripts, "ISGF_ROOT": os.path.abspath(a.root), "KCOLLECT": a.kcollect,
            "RECOLLECT_OF": src}
    for kv in a.extra_env or []:
        k, v = kv.split("=", 1)
        over[k] = v
    # 5m.1: a recollect of a job submitted before the collector budget (its job.env has no --max-steps) gets it too
    if "ISGF_COLLECT_EXTRA" not in over:
        cur = {"ISGF_COLLECT_EXTRA": " ".join(shlex.split(env0.get("ISGF_COLLECT_EXTRA", "''")))}
        if with_collector_max_steps(cur, key) is not None:
            over["ISGF_COLLECT_EXTRA"] = cur["ISGF_COLLECT_EXTRA"]
    with open(os.path.join(job_dir, "job.env"), "a") as fh:
        for k, v in over.items():
            fh.write(f"{k}={shlex.quote(str(v))}\n")
    if a.iids:
        keep = set(a.iids)
        for t in glob.glob(os.path.join(job_dir, "inst", "**", "*-tro_state.json"), recursive=True):
            iid = os.path.basename(t)[: -len("-tro_state.json")]
            if iid not in keep and not os.path.exists(os.path.join(job_dir, "claims", iid)):
                open(os.path.join(job_dir, "claims", iid), "w").write("skip not in --iids\n")
    time_s = a.time_s or int(shlex.split(env0.get("TIME_LIMIT_S", "14400"))[0])
    cmd, gres = sbatch(a, f"{a.job_prefix}rc_{st.short_of(key)}", time_s, os.path.join(scripts, "job_recollect.sh"), job_dir)
    os.makedirs(os.path.join(a.root, "slurm"), exist_ok=True)
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise SystemExit(f"sbatch failed: {res.stderr.strip()}")
    jobid = res.stdout.strip().split(";")[0]
    with open(os.path.join(a.root, "manifests", "recollect.tsv"), "a") as fh:
        fh.write("\t".join([time.strftime("%Y-%m-%dT%H:%M:%S"), key, src, job_dir, jobid, os.path.abspath(a.code), " ".join(a.iids or [])]) + "\n")
    print(f"SUBMITTED {jobid} recollect {key} {job_dir} {gres} time={st.hms(time_s)} k={a.kcollect} "
          f"grasps={shlex.split(env0.get('ISGF_GRASPS', 'legacy'))[0]} (the source job's)")
    return 0


def template(a):
    """job_template.sh: the template-making step that stays of job_val.sh KIND=eval (sample_isg at an eval seed, no demo); register the
    result with register_alias.py, then submit it."""
    job_dir = os.path.join(a.root, "out", "template", a.key, f"s{a.seed_index}" + (f"_{a.name}" if a.name else ""))
    if os.path.exists(job_dir):
        raise SystemExit(f"{job_dir} exists")
    env = {"ALIAS": a.alias, "ALIAS_KEY": a.key, "ISGE_CODE": os.path.abspath(a.code), "ISGF_SCRIPTS": scripts_of(a.code), "ISGF_ROOT": os.path.abspath(a.root),
           "SEED_SPLIT": "eval", "SEED_INDEX": a.seed_index, "N": a.n, "BUDGET_S": a.budget_s, "MAX_ATTEMPTS": a.max_attempts, "START_PER_ATTEMPT": 1,
           "SCENE": a.scene or "", "SUPPORT": a.support or "", "SUPPORT_ROW": a.support_row or ""}
    for kv in a.extra_env or []:
        k, v = kv.split("=", 1)
        env[k] = v
    time_s = a.time_s or int(a.budget_s) + 2700
    cmd, gres = sbatch(a, f"{a.job_prefix}tpl_{st.short_of(a.key)}", time_s, os.path.join(env["ISGF_SCRIPTS"], "job_template.sh"), job_dir)
    if a.dry_run:
        print("DRY", " ".join(shlex.quote(x) for x in cmd))
        print("\n".join(f"  {k}={v}" for k, v in env.items()))
        return 0
    os.makedirs(job_dir)
    os.makedirs(os.path.join(a.root, "slurm"), exist_ok=True)
    with open(os.path.join(job_dir, "job.env"), "w") as fh:
        for k, v in env.items():
            fh.write(f"{k}={shlex.quote(str(v))}\n")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        os.rename(job_dir, job_dir + ".failed_submit")
        raise SystemExit(f"sbatch failed: {res.stderr.strip()}")
    print(f"SUBMITTED {res.stdout.strip().split(';')[0]} template {a.alias} {job_dir} {gres} time={st.hms(time_s)} seed=eval:{a.seed_index}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(s):
        s.add_argument("--code", required=True, help="a frozen snapshot of the trunk (never the copy being edited)")
        s.add_argument("--gres", default="titanrtx")
        s.add_argument("--acct", default="vision")
        s.add_argument("--part", default="svl")
        s.add_argument("--cpus", type=int, default=None, help="default: 3 per process (COLLECTOR_PIN: the sampler + K collectors), at least 8")
        s.add_argument("--mem", default="48G")
        s.add_argument("--kcollect", type=int, default=1)
        s.add_argument("--time-s", type=int, default=None)
        s.add_argument("--extra-env", action="append", default=[], help="NAME=VALUE into job.env (ISPATIALGYM_* / ISG_* win over fast_env.sh)")
        s.add_argument("--root", default=F, help="output root: out/, slurm/, manifests/ (default: trainfast; alias validation: $OFH)")
        s.add_argument("--job-prefix", default="isgf_", help="slurm job-name prefix (validation: isgof_)")
        s.add_argument("--registry", default=None, help="registry TSV (default: <code>/scripts/trainfast/registry.tsv)")
        s.add_argument("--dry-run", action="store_true")

    s = sub.add_parser("submit")
    s.add_argument("key")
    s.add_argument("n", type=int)
    s.add_argument("--flow", default="base", choices=["base", "fast"])
    common(s)
    s.add_argument("--sample-limit-s", type=int, default=None)
    s.add_argument("--tag", default=None)
    s.add_argument("--customs", default=None, choices=["template", "trunk"],
                   help="the sampler's customs: the snapshot's isg_customs.json (default) or the template's own copy (the counted run's)")
    s.add_argument("--ledger", default=None, help="default: <root>/manifests/allocations.tsv")
    s.add_argument("--base-index", type=int, default=BASE_INDEX, help="first train index per alias in this ledger")
    s.add_argument("--v2-root", default=None, help="with --grasps v2 / v2only: the v2 data root (default V2_ROOT, v2maps_20261106)")
    s.add_argument("--shiyuc", action="store_true", help="run as shiyuc: every write under " + SY + " (shiyuc_env); --root / --base-index / --v2-root there")
    s.add_argument("--map-overlay", default=MAP_OVERLAY_DEFAULT,
                   help=f"with --grasps v2 / v2only: the merged reach-map overlay root, pinned to its latest release in job.env; 'off' = none "
                        f"(default {MAP_OVERLAY_DEFAULT}; the campaign root is {MAP_OVERLAY_ROOT})")
    s.add_argument("--grasps", default="legacy", choices=["legacy", "v2", "v2only"],
                   help="grasp library + reach bands: legacy (default) or v2 (the campaign's v2 library, reach maps and anchors, GRASPS_V2)")
    s.add_argument("--alias-memory", default="auto",
                   help="i7-memory: the per-alias outcome memory the sampler reads, pinned in job.env - auto (default: the --memory-mode store's LATEST "
                        "if any, else off), off, or a version dir / memory root")
    s.add_argument("--memory-mode", default="production", choices=sorted(MEMORY_STORES),
                   help="i7-memory2 (owner 2026-11-08 2.2 / 2.4): production (default: <root>/alias_memory_production, starts empty, resets on a customs / "
                        "keys change) or validation (<root>/alias_memory_validation: seeded from the replay, carried over at the reduced weight)")
    s.add_argument("--no-memory-aggregate", dest="memory_aggregate", action="store_false",
                   help="i7-memory: do not queue the batch's aggregation job (job_memory.sh, CPU, afterany this job)")
    r = sub.add_parser("recollect")
    r.add_argument("src")
    common(r)
    r.add_argument("--name", default=None)
    r.add_argument("--iids", nargs="+", default=None)
    tp = sub.add_parser("template")
    tp.add_argument("alias")
    tp.add_argument("key")
    common(tp)
    tp.add_argument("--seed-index", type=int, required=True)
    tp.add_argument("--scene", default=None)
    tp.add_argument("--support", default=None)
    tp.add_argument("--support-row", default=None)
    tp.add_argument("--n", type=int, default=1)
    tp.add_argument("--budget-s", type=int, default=3600)
    tp.add_argument("--max-attempts", type=int, default=1000)
    tp.add_argument("--name", default=None)
    pp = sub.add_parser("params")
    pp.add_argument("key")
    pp.add_argument("n", type=int, nargs="?", default=4)
    pp.add_argument("--registry", default=None)
    kk = sub.add_parser("keys")
    kk.add_argument("--registry", default=None)
    sub.add_parser("running")
    a = ap.parse_args()
    if a.cmd == "submit":
        return submit(a)
    if a.cmd == "recollect":
        return recollect(a)
    if a.cmd == "template":
        return template(a)
    if a.cmd == "params":
        print(st.params(aliases(a)[a.key], a.n))
        return 0
    if a.cmd == "keys":
        for k, c in sorted(aliases(a).items()):
            print(k, "registry" if "_registry" in c else "counted", c["alias"])
        return 0
    if a.cmd == "running":
        out = subprocess.run(["squeue", "-u", "minyeong", "-h", "-o", "%i|%j|%T|%P|%M|%l|%N|%R"], capture_output=True, text=True).stdout
        for l in out.splitlines():
            if l.split("|")[1].startswith(("isgf_", "isgof_")):
                print(l)
        return 0


if __name__ == "__main__":
    sys.exit(main())
