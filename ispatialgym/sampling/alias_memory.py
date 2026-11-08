"""Per-alias OUTCOME MEMORY (owner decision 2026-11-08 01:45 UTC, part C; branch i7-memory): down-weight the configurations that keep
failing for an alias, unattended, on top of the sampler's own order.

Five steps (docs/GRASP_BANDS.md "Per-alias outcome memory"):

1. RECORD (:func:`record_job`, run by job_fast.sh after its collectors): one row per collected demo into the job's own shard
   ``<job dir>/alias_memory_shard.jsonl`` (one writer per job - no races): the alias, the configuration keys (PLACE: arm, place stance,
   stand-off bin, trunk translate / tilt, goal yaw slot; PICK: arm, pick stance, stand-off bin, trunk, grasp family, entry key), the
   outcome, the failure type and its STAGE, the context (trunk snapshot, alias customs version = the TRO's task_spec_sha256, the entry /
   map key fingerprint of the alias's models) and the instance's sampling provenance (the memory version and weight it was drawn with).
2. ATTRIBUTE (:func:`attribute`): a failure counts only against the choice that precedes its stage - pick-stage failures (everything up
   to the lift: push-to-pick, grasp, lift; the whole program of a task without a place) go to the PICK configuration and leave the place
   configuration unobserved; carry- and place-stage failures (the carried legs to stance C, the place and everything after it, and the
   end-of-episode verdicts of a placing task: symbolic_only / not_reached / not_settled) go to the PLACE configuration and count as a
   pick pass. A success is a pass for both. Infrastructure failures (rc != 0/2, status error, timeout / wall clock, OOM, a missing
   meta or sidecar) and the documented known bugs (``assets/alias_memory/exclusions.json``) are excluded.
3. AGGREGATE (:func:`aggregate`, an offline CPU job after every batch): the shards (a finished job without one is back-filled) merge
   into a NEW immutable version ``<memory root>/v<NNNN>/`` (``aliases/<alias>.json`` = its observations, ``meta.json``, ``report.json``
   / ``report.md`` = the guardrails); ``LATEST`` names it. submit_fast.py pins the version a job reads in its job.env
   (``ISPATIALGYM_ALIAS_MEMORY``).
4. SAMPLE (:func:`reorder`): per (alias, configuration bin) a Beta posterior; hierarchical keys (alias x arm x trunk, then x stand-off,
   then x yaw slot); the finest level with at least ``min_n`` attributed demos decides (back-off to the coarser one when sparse). A bin
   is DOWN-WEIGHTED when it is credibly worse than the rest of the alias (P(theta_bin < theta_rest) >= p_worse) and its posterior mean
   is <= max_mean. The hook keeps the sampler's PRIOR ORDER (e.g. i7-placeorder's least-trunk-first) and only defers: per attempt a
   down-weighted bin keeps its place with probability max(floor, theta_bin / theta_rest) (Thompson draws), else its candidates move
   to the end of the order (still tried when everything before them fails - never a blacklist). A memory without a down-weighted bin
   changes nothing and draws no random number (bit-identical to memory off). Kit counters ``alias_memory.*``.
5. FORGET (:meth:`AliasMemory.weights`): evidence from another trunk snapshot decays by ``snapshot_decay`` per newer snapshot of the
   alias; evidence under another alias customs version or another entry / map key fingerprint is RESET (weight 0).

Two MODES (owner decisions 2026-11-08 2.2 / 2.4; branch i7-memory2), one switch (``submit_fast.py --memory-mode``, job.env
``ISPATIALGYM_ALIAS_MEMORY_MODE``), two separate stores:

* ``production`` (the default): store ``<root>/alias_memory``; starts EMPTY (only demos of production-mode jobs are aggregated into it);
  an alias's evidence RESETS when its customs version or its entry / map keys change.
* ``validation`` (opt-in): store ``<root>/alias_memory_validation``; SEEDED from the replay (``VALIDATION_SEED``: the seed version's demos,
  and the demos of jobs submitted before the modes existed) and CARRIED OVER a customs / keys change, both at the reduced weight
  ``VALIDATION_WEIGHT`` (1/3: three carried demos count as one fresh one), flat; evidence of the run's own context decays per newer
  snapshot as in production but never below that weight. Only validation-mode jobs (and the seed) enter it; production never reads it.

The PICK hook (owner 2.3): the same :func:`reorder` over the pick (manipulation) stance candidates - IORS large's stance B, IORS small's
and articulation small's manipulation stance - keyed (arm x trunk, then x stand-off): the key whose findings the replay followed by
failure (9 of 10 later demos; the grasp-family key: 6 of 10). Never a block: Thompson draws and the floor, as the place hook.

Guardrails (:func:`guardrails`): per batch and alias - bins down-weighted, the share of the configuration space they hold, the yield
and GPU-h per success before / after; ALERTS only (an alias losing > ``alert_space_share`` of its space, a yield drop, a batch where
everything failed = a bug, not a configuration); a configuration down-weighted in >= ``shared_fix_aliases`` aliases is flagged as a
shared-fix candidate for the owner. The loop never changes a shared rule.

Never for eval-instance generation: the memory is consulted only after :func:`enable_for_split` was called with ``"train"`` (only
``multiply_isg --split train`` does); ``sample_isg`` (eval templates) never enables it, whatever the environment says.

CLI: ``python -m ispatialgym.sampling.alias_memory record <job dir> | aggregate --root R [--memory M] [--batch TAG] |
replay --root R --out DIR | show <version dir> <alias>``.
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

SCHEMA = "isg-alias-memory/1"
SHARD_NAME = "alias_memory_shard.jsonl"
ENV_PATH = "ISPATIALGYM_ALIAS_MEMORY"            # the pinned version dir (submit_fast.py writes it into job.env); "off" / unset = off
ENV_ALIAS = "ISPATIALGYM_ALIAS_MEMORY_ALIAS"     # the alias key the job samples (job.env ALIAS_KEY)
ENV_MODE = "ISPATIALGYM_ALIAS_MEMORY_MODE"       # i7-memory2: production (default) | validation - the store the job reads and writes
MODES = ("production", "validation")
#: the store of each mode under a root (separate: a validation run never writes into the production memory, nor the reverse)
STORE_DIRS = {"production": "alias_memory", "validation": "alias_memory_validation"}
#: i7-memory2 (owner 2.2 / 2.4): the weight of seeded and carried-over evidence in validation mode - the least weight with which the
#: 2026-11-08 replay still flags inside_large_01's 0.85 m / trunk 0.50 bin (0 / 3: mean 1 / (2 + 3 w) <= 0.35 needs w >= 0.286) from the
#: seed at a new snapshot, and with which the chronological replay flags nothing with a healthy record (docs/GRASP_BANDS.md)
VALIDATION_WEIGHT = 1.0 / 3.0
#: the replay the validation store is seeded from (owner 2.4); its LATEST version at seeding time is pinned in the store's seed.json
VALIDATION_SEED = "/vision/u/minyeong/projects/isg_home/oracle_fix/memory_20261108/alias_memory"
EXCLUSIONS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "alias_memory", "exclusions.json")
PLACE_LEVELS = (("arm", "translate", "tilt"), ("arm", "translate", "tilt", "stand_off"), ("arm", "translate", "tilt", "stand_off", "yaw_slot"))
PICK_LEVELS = (("arm", "translate", "tilt"), ("arm", "translate", "tilt", "stand_off"), ("arm", "translate", "tilt", "stand_off", "family"))
#: the levels the place hook decides at (the goal yaw is drawn after stance C: the yaw level is recorded and reported, not used there)
PLACE_HOOK_LEVELS = 2
#: the levels the pick hook decides at (i7-memory2, owner 2.3): arm x trunk, then x stand-off - the grasp family is recorded and reported
#: (the replay: stance-level findings were followed by failure in 9 of 10 later demos, family-level ones in 6 of 10)
PICK_HOOK_LEVELS = 2
YAW_SLOTS = 16  # the sampler's yaw grid (pi / 8): pair_key's bucket


@dataclasses.dataclass
class Thresholds:
    """The owner sets these once (proposed defaults, justified from the 2026-11-08 replay over $OFH/out in docs/GRASP_BANDS.md)."""
    min_n: int = 3                 # raw attributed demos in a bin before it can be down-weighted (at the deciding level)
    p_worse: float = 0.95          # P(theta_bin < theta_rest of the alias) needed
    max_mean: float = 0.35         # the bin's posterior mean (uniform prior) must be at most this
    floor: float = 0.05            # exploration floor: the least probability a down-weighted bin keeps its prior place
    snapshot_decay: float = 0.8    # evidence weight per newer trunk snapshot of the alias
    customs_change: float = 0.0    # evidence weight under another alias customs version (0 = reset; > 0: carried over, flat)
    keys_change: float = 0.0       # evidence weight under another entry / map key fingerprint (0 = reset; > 0: carried over, flat)
    seed_weight: float = 0.0       # i7-memory2: the weight of a seeded demo (validation: the replay's), flat
    weight_floor: float = 0.0      # i7-memory2: the least weight the snapshot decay leaves a demo of the run's own context
    mode: str = "production"       # i7-memory2: the store these thresholds belong to (production | validation)
    alert_space_share: float = 0.5  # alert when the down-weighted bins hold more than this share of the alias's observed space
    alert_yield_drop: float = 0.25  # alert when the batch's yield is this much (absolute) below the alias's earlier yield
    alert_min_demos: int = 4       # demos a batch / an earlier window needs before a yield alert / an all-fail alert
    shared_fix_aliases: int = 3    # a cross-alias configuration down-weighted in this many aliases -> shared-fix candidate

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "Thresholds":
        f = {x.name for x in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in f})

    @classmethod
    def for_mode(cls, mode: str = "production") -> "Thresholds":
        """The owner's thresholds (2.1) of a mode: production resets on a customs / keys change and has no seed; validation carries over
        and seeds at :data:`VALIDATION_WEIGHT` (2.2 / 2.4), its own context's evidence never decayed below that weight."""
        if mode not in MODES:
            raise ValueError(f"alias memory mode {mode!r}: one of {MODES}")
        if mode == "production":
            return cls()
        w = VALIDATION_WEIGHT
        return cls(customs_change=w, keys_change=w, seed_weight=w, weight_floor=w, mode="validation")


# ---------------------------------------------------------------------------------------------------------------- keys / context

def q(x: Optional[float], step: float) -> Optional[float]:
    """Quantize to ``step`` (stand-off 0.05 m, trunk 0.01)."""
    if x is None:
        return None
    return round(round(float(x) / step + 1e-9) * step, 4)


def yaw_slot(yaw: Optional[float], n: int = YAW_SLOTS) -> Optional[int]:
    if yaw is None:
        return None
    return int(round(float(yaw) / (2.0 * math.pi / n))) % n


def stance_key(arm: Optional[str], stance: Optional[Dict[str, Any]], name: str) -> Optional[Dict[str, Any]]:
    if not isinstance(stance, dict):
        return None
    arms = stance.get("arms")
    if arm is None and isinstance(arms, list) and arms:
        arm = arms[0]
    return {"arm": arm or "-", "stance": name, "stand_off": q(stance.get("stand_off"), 0.05),
            "translate": q(stance.get("translate"), 0.01), "tilt": q(stance.get("tilt"), 0.01)}


def grasp_family(name: Optional[str]) -> str:
    """The grasp family of a recorded grasp name: the twin suffix (``~r180``) folded in (a v2 library family groups a grasp with its
    twins); '-' without a grasp."""
    if not name:
        return "-"
    return str(name).split("~")[0]


def config_keys(sidecar: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """(place key, pick key) of a written instance from its sidecar (``*.oracle.json``). The place stance is ``placement`` when the
    task has one, else the single manipulation stance (tabletop tasks place where they pick); the pick stance is ``pick`` (an
    overhang pick's own stance), else ``manipulation``."""
    st = sidecar.get("stances") or {}
    g = (sidecar.get("grasps") or [None])[0] or {}
    arm = g.get("arm")
    place = None
    has_place = bool(g.get("place")) or "placement" in st
    if has_place:
        pn = "placement" if "placement" in st else "manipulation"
        place = stance_key(arm, st.get(pn), pn)
        if place is not None:
            place["yaw_slot"] = yaw_slot(g.get("yaw"))
            place["edge"] = (st.get(pn) or {}).get("edge")
    pn = "pick" if "pick" in st else "manipulation"
    pick = stance_key(arm, st.get(pn), pn)
    if pick is not None:
        pick["family"] = grasp_family(g.get("grasp_name"))
        pick["grasp_key"] = g.get("grasp_key")
    return place, pick


def model_prefixes(entry: Optional[Dict[str, Any]]) -> List[str]:
    """``<category>__<model>__`` of every role's grasp keys / representative in a customs entry (the glib / map directory prefixes)."""
    out = set()
    for role in ((entry or {}).get("roles") or {}).values():
        if not isinstance(role, dict):
            continue
        for v in (role.get("grasp_keys") or {}).values():
            p = str(v).split("__")
            if len(p) >= 2:
                out.add(f"{p[0]}__{p[1]}__")
        if role.get("category") and role.get("representative"):
            out.add(f"{role['category']}__{role['representative']}__")
    return sorted(out)


_FP_CACHE: Dict[Tuple[Any, ...], str] = {}


def keys_fingerprint(glib_root: Optional[str], maps_root: Optional[str], grasps_mode: Optional[str], prefixes: Sequence[str]) -> str:
    """The entry / map key fingerprint of an alias's models: the v2 library content (grasps, families, rests, twins - not the
    provenance) of every model directory with the alias's prefixes, and the reach-map file names (content-hashed) under them. A
    staged copy of the same libraries / maps has the same fingerprint; a rebuilt library or a new map changes it."""
    key = (glib_root, maps_root, grasps_mode, tuple(prefixes))
    if key in _FP_CACHE:
        return _FP_CACHE[key]
    h = hashlib.sha1(f"grasps={grasps_mode or '-'}".encode())
    for p in prefixes:
        if glib_root:
            for d in sorted(glob.glob(os.path.join(glib_root, "models", p + "*"))):
                h.update(os.path.basename(d).encode())
                try:
                    with open(os.path.join(d, "library.json")) as f:
                        lib = json.load(f)
                    h.update(json.dumps({k: lib.get(k) for k in ("grasps", "families", "rests", "twins")}, sort_keys=True, default=str).encode())
                except (OSError, ValueError):
                    h.update(b"-")
        if maps_root:
            for d in sorted(glob.glob(os.path.join(maps_root, p + "*"))):
                h.update(os.path.basename(d).encode())
                try:
                    for f in sorted(os.listdir(d)):
                        h.update(f.encode())
                except OSError:
                    pass
    fp = h.hexdigest()[:12]
    _FP_CACHE[key] = fp
    return fp


def snapshot_name(code_root: Optional[str]) -> str:
    """The trunk snapshot a job ran (``ISGE_CODE`` = $OFH/code/trunk_<sha>): its directory name."""
    return os.path.basename(os.path.normpath(code_root)) if code_root else "-"


# ---------------------------------------------------------------------------------------------------------------- attribution

def _names(meta: Dict[str, Any]) -> List[str]:
    """The action names of the demo's whole program (``meta.actions`` lists only the actions executed up to the failure)."""
    prog = meta.get("program")
    if isinstance(prog, list) and prog:
        return [re.match(r"[a-z_]*", str(a)).group(0) for a in prog]
    return [str(a.get("name")) for a in (meta.get("actions") or []) if isinstance(a, dict)]


def stage_of(meta: Dict[str, Any]) -> str:
    """The stage of a failed demo's verdict: ``pick`` (up to and including the lift after the first grasp; the push-to-pick before
    it), ``carry`` (between the lift and the first place: the carried legs to stance C, a reorient), ``place`` (the place and
    everything after it), ``manipulation`` (a program without a grasp-and-place: articulation, pushes, ...), ``outcome`` (a verdict
    without an action: the end-of-episode evaluation)."""
    v = meta.get("verdict") or {}
    names = _names(meta)
    ai = v.get("action_index")
    act = v.get("action")
    if ai is None and v.get("step") is not None:
        for i, a in enumerate(meta.get("actions") or []):
            if isinstance(a, dict) and a.get("step_start") is not None and a.get("step_end") is not None and a["step_start"] <= v["step"] < a["step_end"]:
                ai, act = i, a.get("name")
                break
    if ai is None and act is None:
        return "outcome"
    if act is not None and (ai is None or ai >= len(names) or names[ai] != act):
        cand = [i for i, n in enumerate(names) if n == act]
        ai = min(cand, key=lambda i: abs(i - (ai if ai is not None else i))) if cand else ai
    if ai is None:
        return "outcome"
    i_grasp = next((i for i, n in enumerate(names) if n == "grasp"), None)
    i_place = next((i for i, n in enumerate(names) if n == "place" and (i_grasp is None or i > i_grasp)), None)
    if i_grasp is None or i_place is None:
        return "manipulation"
    i_lift = i_grasp + 1 if i_grasp + 1 < len(names) and names[i_grasp + 1] == "lift" else i_grasp
    if ai <= i_lift:
        return "pick"
    if ai < i_place:
        return "carry"
    return "place"


def load_exclusions(path: Optional[str] = None) -> List[Dict[str, Any]]:
    try:
        with open(path or EXCLUSIONS_FILE) as f:
            return list(json.load(f).get("exclusions") or [])
    except (OSError, ValueError):
        return []


def _match(rule: Dict[str, Any], row: Dict[str, Any]) -> bool:
    view = {**row, "snapshot": (row.get("ctx") or {}).get("snapshot")}
    for k, want in (rule.get("match") or {}).items():
        have = view.get(k)
        if isinstance(want, list):
            if have not in want:
                return False
        elif isinstance(want, dict):
            if "not_in" in want and have in want["not_in"]:
                return False
            if "in" in want and have not in want["in"]:
                return False
            if "re" in want and not re.search(str(want["re"]), str(have or "")):
                return False
        elif have != want:
            return False
    return True


def exclusion_of(row: Dict[str, Any], rules: Sequence[Dict[str, Any]]) -> Optional[str]:
    """The id of the first exclusion rule a demo row matches (infrastructure first, then the known bugs); None = counted."""
    for r in rules:
        if _match(r, row):
            return str(r.get("id"))
    return None


def attribute(row: Dict[str, Any]) -> Tuple[Optional[int], Optional[int]]:
    """(place observation, pick observation) of a demo row: 1 pass, 0 failure, None unobserved (the stage was not reached, or the
    task has no such configuration, or the row is excluded)."""
    if row.get("excluded") or row.get("outcome") not in ("success", "failure"):
        return None, None
    has_place, has_pick = row.get("place") is not None, row.get("pick") is not None
    if row["outcome"] == "success":
        return (1 if has_place else None), (1 if has_pick else None)
    stage = row.get("stage")
    if stage in ("pick", "manipulation") or (stage == "outcome" and not has_place):
        return None, (0 if has_pick else None)
    if stage in ("carry", "place", "outcome"):
        return (0 if has_place else None), (1 if has_pick else None)
    return None, None


# ---------------------------------------------------------------------------------------------------------------- record

def read_env(path: str) -> Dict[str, str]:
    env: Dict[str, str] = {}
    try:
        for line in open(path):
            if "=" in line and not line.startswith("#"):
                k, v = line.rstrip("\n").split("=", 1)
                env[k.strip()] = v.strip().strip("'\"")
    except OSError:
        pass
    return env


def _json(path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def demo_rows(job_dir: str, rules: Optional[Sequence[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
    """One memory row per results.jsonl line of a job (the job's shard content)."""
    rules = load_exclusions() if rules is None else rules
    env = read_env(os.path.join(job_dir, "job.env"))
    parts = os.path.normpath(job_dir).split(os.sep)
    alias = env.get("ALIAS_KEY") or parts[-2]
    tag = parts[-3] if len(parts) >= 3 else "-"
    jobid = (open(os.path.join(job_dir, "jobid")).read().strip() if os.path.exists(os.path.join(job_dir, "jobid")) else None)
    customs = _json(os.path.join(job_dir, "inst", "task_custom_lists.json"))  # the customs the job's sampler used
    entry = (customs or {}).get(env.get("ALIAS", "")) or (next(iter(customs.values())) if customs and len(customs) == 1 else None)
    glib = env.get("ISPATIALGYM_GRASP_LIBRARY_V2") or None
    maps = env.get("ISPATIALGYM_REACH_MAPS") or None
    keys_fp = keys_fingerprint(glib, maps, None, model_prefixes(entry))
    snap = snapshot_name(env.get("ISGE_CODE"))
    out = []
    res = os.path.join(job_dir, "results.jsonl")
    if not os.path.exists(res):
        return out
    for line in open(res):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        inst = r.get("instance") or ""
        meta = _json(os.path.join(r.get("out") or "", "meta.json")) or {}
        sc_path = meta.get("sidecar") or ((r.get("tro") or "").replace("-tro_state.json", ".oracle.json") if r.get("tro") else None)
        sidecar = _json(sc_path) or {}
        tro = _json(r.get("tro")) or {}
        spec = ((tro.get("ispatialgym") or {}).get("task_spec_sha256") or "-")[:16]
        place, pick = config_keys(sidecar) if sidecar else (None, None)
        v = meta.get("verdict") or {}
        status = r.get("status")
        ok = bool(r.get("success")) and status == "success"
        row = {"schema": SCHEMA, "alias": alias, "variant": env.get("ALIAS"), "tag": tag, "job": job_dir, "jobid": jobid, "inst": inst,
               "index": int(inst.rsplit("__", 1)[-1]) if inst.rsplit("__", 1)[-1].isdigit() else None,
               "finished": r.get("finished"), "rc": r.get("rc"), "status": status, "reason": r.get("reason"), "wall_s": r.get("wall_s"),
               "failure_type": None if ok else (v.get("category") or r.get("reason") or status), "action": v.get("action"),
               "action_index": v.get("action_index"), "wall_clock_exceeded": bool(meta.get("wall_clock_exceeded")),
               "meta": bool(meta), "sidecar": bool(sidecar),
               "ctx": {"snapshot": snap, "spec": spec, "keys": keys_fp},
               "place": place, "pick": pick,
               "sampled_with": ((sidecar.get("stances") or {}).get("placement") or {}).get("alias_memory"),
               "sampled_with_pick": _pick_provenance(sidecar),
               "memory_pinned": env.get(ENV_PATH) or None,
               "mode": env.get(ENV_MODE) or None}  # i7-memory2: the store this demo belongs to (None: a job from before the modes)
        row["outcome"] = "success" if ok else "failure"
        row["stage"] = None if ok else stage_of(meta)
        row["excluded"] = exclusion_of(row, rules)
        po, ko = attribute(row)
        row["place_obs"], row["pick_obs"] = po, ko
        out.append(row)
    return out


def _pick_provenance(sidecar: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The pick hook's provenance on the written instance: the ``alias_memory_pick`` extra of its pick / manipulation stance."""
    st = sidecar.get("stances") or {}
    for name in ("pick", "manipulation"):
        p = (st.get(name) or {}).get("alias_memory_pick")
        if p is not None:
            return p
    return None


def record_job(job_dir: str, out: Optional[str] = None) -> str:
    """Write the job's shard (``<job dir>/alias_memory_shard.jsonl``; atomic replace)."""
    rows = demo_rows(job_dir)
    path = out or os.path.join(job_dir, SHARD_NAME)
    tmp = path + f".tmp{os.getpid()}"
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    os.replace(tmp, path)
    return path


def job_finished(job_dir: str, stale_s: float = 6 * 3600) -> bool:
    """A job whose shard the aggregator may back-fill: ISGF_END in job.txt, or nothing written for ``stale_s`` (a killed job)."""
    try:
        if "ISGF_END" in open(os.path.join(job_dir, "job.txt"), errors="replace").read():
            return True
    except OSError:
        pass
    try:
        newest = max(os.path.getmtime(os.path.join(job_dir, f)) for f in os.listdir(job_dir))
    except (OSError, ValueError):
        return False
    return time.time() - newest > stale_s


# ---------------------------------------------------------------------------------------------------------------- posterior

_GRID = np.linspace(0.0005, 0.9995, 1000)


def _beta_pdf(a: float, b: float) -> np.ndarray:
    lg = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
    p = np.exp(lg + (a - 1) * np.log(_GRID) + (b - 1) * np.log1p(-_GRID))
    return p / p.sum()


def p_less(a1: float, b1: float, a2: float, b2: float) -> float:
    """P(X < Y) for X ~ Beta(a1, b1), Y ~ Beta(a2, b2) (grid integration, 1e-3)."""
    px, py = _beta_pdf(a1, b1), _beta_pdf(a2, b2)
    sy = 1.0 - np.cumsum(py)  # P(Y > grid_i)
    return float(np.clip((px * sy).sum(), 0.0, 1.0))


def beta_quantile(a: float, b: float, p: float) -> float:
    c = np.cumsum(_beta_pdf(a, b))
    return float(_GRID[min(int(np.searchsorted(c, p)), len(_GRID) - 1)])


@dataclasses.dataclass
class Bin:
    level: int
    key: Tuple[Any, ...]
    s: float = 0.0       # decayed passes
    f: float = 0.0       # decayed failures
    n_raw: int = 0       # attributed demos (raw count, the min_n gate)
    s_raw: int = 0
    failures: Counter = dataclasses.field(default_factory=Counter)

    @property
    def mean(self) -> float:
        return (1.0 + self.s) / (2.0 + self.s + self.f)


def level_key(cfg: Dict[str, Any], fields: Sequence[str]) -> Tuple[Any, ...]:
    return tuple(cfg.get(k) for k in fields)


@dataclasses.dataclass
class Decision:
    """A bin's verdict: ``down`` and the posterior facts (for the report and the instance's provenance)."""
    down: bool
    level: int
    key: Tuple[Any, ...]
    n_raw: int
    s: float
    f: float
    rest_s: float
    rest_f: float
    p_worse: float
    mean: float
    ub95: float

    def facts(self) -> Dict[str, Any]:
        return {"down": self.down, "level": self.level, "bin": list(self.key), "n": self.n_raw, "s": round(self.s, 3), "f": round(self.f, 3),
                "rest_s": round(self.rest_s, 3), "rest_f": round(self.rest_f, 3), "p_worse": round(self.p_worse, 4), "mean": round(self.mean, 4),
                "ub95": round(self.ub95, 4)}


class AliasMemory:
    """One alias's observations of one memory version, weighted for a sampling context (snapshot, customs version, keys)."""

    def __init__(self, alias: str, rows: Sequence[Dict[str, Any]], thresholds: Thresholds, version: str = "-", snap_order: Sequence[str] = ()):
        self.alias, self.rows, self.th, self.version = alias, list(rows), thresholds, version
        self.snap_order = list(snap_order) or _snap_order(self.rows)
        self.w: List[float] = [1.0] * len(self.rows)
        self.context: Optional[Dict[str, str]] = None
        self.bins: Dict[str, Dict[Tuple[int, Tuple[Any, ...]], Bin]] = {}
        self.totals: Dict[str, Tuple[float, float]] = {}

    # -- forgetting
    def weights(self, context: Optional[Dict[str, str]]) -> List[float]:
        """Evidence weights for a sampling context:

        * a SEEDED demo (``row["seed"]``, validation stores only): ``seed_weight``, flat;
        * a demo under another alias customs version / another entry / map key fingerprint: ``customs_change`` / ``keys_change`` (the
          smaller when both changed), flat (0 = reset, the production default; i7-memory2: a carried-over demo is discounted once, not
          also by its age - compounded with the snapshot decay no weight below ~0.9 kept inside_large_01's 0 / 3 bin in the replay);
        * a demo of the context's own customs and keys: ``snapshot_decay ** k`` for evidence k snapshots older than the context's (the
          alias's snapshot order, the context's own snapshot last when it is new), never below ``weight_floor``."""
        self.context = dict(context) if context else None
        order = list(self.snap_order)
        if context and context.get("snapshot") and context["snapshot"] not in order:
            order.append(context["snapshot"])
        pos = {s: i for i, s in enumerate(order)}
        cur = pos.get(context["snapshot"], len(order) - 1) if context and context.get("snapshot") else len(order) - 1
        out = []
        for r in self.rows:
            c = r.get("ctx") or {}
            if r.get("seed"):
                out.append(max(0.0, float(self.th.seed_weight)))
                continue
            other = []
            if context and context.get("spec") not in (None, "-") and c.get("spec") != context.get("spec"):
                other.append(float(self.th.customs_change))
            if context and context.get("keys") not in (None, "-") and c.get("keys") != context.get("keys"):
                other.append(float(self.th.keys_change))
            if other:
                out.append(max(0.0, min(other)))
                continue
            age = max(0, cur - pos.get(c.get("snapshot"), cur))
            out.append(max(float(self.th.snapshot_decay) ** age, float(self.th.weight_floor)))
        self.w = out
        self.bins, self.totals = {}, {}
        return out

    # -- bins
    def _build(self, kind: str) -> Dict[Tuple[int, Tuple[Any, ...]], Bin]:
        if kind in self.bins:
            return self.bins[kind]
        levels = PLACE_LEVELS if kind == "place" else PICK_LEVELS
        bins: Dict[Tuple[int, Tuple[Any, ...]], Bin] = {}
        S = F = 0.0
        for r, w in zip(self.rows, self.w):
            obs, cfg = r.get(f"{kind}_obs"), r.get(kind)
            if obs is None or cfg is None or w <= 0.0:
                continue
            S += w * obs
            F += w * (1 - obs)
            for li, fields in enumerate(levels):
                k = level_key(cfg, fields)
                b = bins.setdefault((li, k), Bin(li, k))
                b.s += w * obs
                b.f += w * (1 - obs)
                b.n_raw += 1
                b.s_raw += int(obs)
                if not obs:
                    b.failures[str(r.get("failure_type"))] += 1
        self.bins[kind], self.totals[kind] = bins, (S, F)
        return bins

    def decide(self, kind: str, cfg: Dict[str, Any], max_level: Optional[int] = None) -> Optional[Decision]:
        """The decision for a configuration: the finest level (up to ``max_level`` levels) whose bin has ``min_n`` attributed demos;
        None when no level has (no memory for it: the prior order stands)."""
        bins = self._build(kind)
        levels = PLACE_LEVELS if kind == "place" else PICK_LEVELS
        top = len(levels) if max_level is None else max_level
        S, F = self.totals.get(kind, (0.0, 0.0))
        for li in reversed(range(top)):
            b = bins.get((li, level_key(cfg, levels[li])))
            if b is None or b.n_raw < self.th.min_n:
                continue
            rs, rf = max(S - b.s, 0.0), max(F - b.f, 0.0)
            pw = p_less(1.0 + b.s, 1.0 + b.f, 1.0 + rs, 1.0 + rf)
            mean = b.mean
            down = pw >= self.th.p_worse and mean <= self.th.max_mean
            return Decision(down, li, b.key, b.n_raw, b.s, b.f, rs, rf, pw, mean, beta_quantile(1.0 + b.s, 1.0 + b.f, 0.95))
        return None

    def down_bins(self, kind: str, max_level: Optional[int] = None) -> List[Decision]:
        """Every down-weighted bin of the alias (at the level that decides for it; a finer bin under a down-weighted coarser one is
        reported once, at the finest deciding level)."""
        bins = self._build(kind)
        levels = PLACE_LEVELS if kind == "place" else PICK_LEVELS
        top = len(levels) if max_level is None else max_level
        seen, out = set(), []
        for (li, k), b in sorted(bins.items(), key=lambda kv: (-kv[0][0], str(kv[0][1]))):
            if li >= top or b.n_raw < self.th.min_n:
                continue
            # the decision a configuration of this bin gets (the finest level with data decides)
            d = self.decide(kind, dict(zip(levels[li], k)), max_level=li + 1)
            if d is None or not d.down or (d.level, d.key) in seen:
                continue
            # a coarser bin holding exactly the evidence of a finer down-weighted one is the same finding
            if any(o.level > d.level and o.key[:len(d.key)] == d.key and o.n_raw == d.n_raw for o in out):
                continue
            seen.add((d.level, d.key))
            out.append(d)
        return out


def _snap_order(rows: Sequence[Dict[str, Any]]) -> List[str]:
    first: Dict[str, str] = {}
    for r in rows:
        s, t = (r.get("ctx") or {}).get("snapshot"), str(r.get("finished") or "")
        if s and (s not in first or t < first[s]):
            first[s] = t
    return [s for s, _ in sorted(first.items(), key=lambda kv: kv[1])]


# ---------------------------------------------------------------------------------------------------------------- sampling hook

_SPLIT: Dict[str, Optional[str]] = {"split": None}
_LOADED: Dict[Tuple[str, ...], Optional[AliasMemory]] = {}
_MISMATCH: set = set()  # (path, alias, mode) whose pinned version belongs to the other mode's store


def enable_for_split(split: Optional[str]) -> None:
    """Declare the generating split. The memory is consulted only for ``train`` (multiply_isg --split train); never for eval."""
    _SPLIT["split"] = split


def active_path() -> Optional[str]:
    p = os.environ.get(ENV_PATH, "").strip()
    if not p or p.lower() in ("0", "off", "none", "false") or _SPLIT["split"] != "train":
        return None
    return p


def _count(sampler, key: str, n: int = 1) -> None:
    st = getattr(getattr(sampler, "kit", None), "stats", None)
    if isinstance(st, dict):
        st[f"alias_memory.{key}"] = st.get(f"alias_memory.{key}", 0) + int(n)


def load_version(path: str, alias: str) -> Optional[AliasMemory]:
    meta = _json(os.path.join(path, "meta.json")) or {}
    data = _json(os.path.join(path, "aliases", f"{alias}.json"))
    if data is None:
        return None
    return AliasMemory(alias, data.get("rows") or [], Thresholds.from_dict(meta.get("thresholds")), version=str(meta.get("version") or os.path.basename(path)),
                       snap_order=data.get("snap_order") or ())


def sampler_context(sampler) -> Dict[str, str]:
    """The sampling job's context: its trunk snapshot (ISGE_CODE, else the code root of this module), its alias customs version (the
    spec hash the TRO records), the entry / map key fingerprint of its models (the libraries / maps the environment points to)."""
    code = os.environ.get("ISGE_CODE") or os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    spec = getattr(getattr(sampler, "ctx", None), "spec_sha256", None)
    if not spec:
        gp = getattr(sampler, "gp", None)
        spec = gp.spec_hash() if gp is not None and callable(getattr(gp, "spec_hash", None)) else "-"
    entry = getattr(getattr(sampler, "gp", None), "entry", None)
    keys = keys_fingerprint(os.environ.get("ISPATIALGYM_GRASP_LIBRARY_V2") or None, os.environ.get("ISPATIALGYM_REACH_MAPS") or None,
                            None, model_prefixes(entry))
    return {"snapshot": snapshot_name(code), "spec": str(spec)[:16], "keys": keys}


def job_mode() -> str:
    """The job's memory mode (job.env ``ISPATIALGYM_ALIAS_MEMORY_MODE``; unset = production)."""
    m = (os.environ.get(ENV_MODE) or "production").strip().lower()
    return m if m in MODES else "production"


def version_mode(path: str) -> str:
    """The mode of a memory version (its meta.json thresholds; a version from before the modes = production)."""
    meta = _json(os.path.join(path, "meta.json")) or {}
    return str(meta.get("mode") or (meta.get("thresholds") or {}).get("mode") or "production")


def memory_for(sampler) -> Optional[AliasMemory]:
    """The alias memory the sampler consults (cached per process): None when off (unset, eval generation, no file for the alias, or a
    version of the other mode's store - counted ``alias_memory.mode_mismatch``: a validation run never reads the production memory, nor
    the reverse)."""
    path = active_path()
    if path is None:
        if os.environ.get(ENV_PATH) and _SPLIT["split"] != "train":
            _count(sampler, "blocked_not_train")
        return None
    alias = os.environ.get(ENV_ALIAS) or getattr(getattr(sampler, "gp", None), "variant", None) or "-"
    k = (path, alias, job_mode())
    if k not in _LOADED:
        mem = None
        if version_mode(path) == job_mode():
            mem = load_version(path, alias)
            if mem is not None:
                mem.weights(sampler_context(sampler))
        else:
            _MISMATCH.add(k)
        _LOADED[k] = mem
    mem = _LOADED[k]
    if k in _MISMATCH:
        _count(sampler, "mode_mismatch")
        return None
    _count(sampler, "loaded" if mem is not None else "no_alias_file")
    _count(sampler, f"mode.{job_mode()}")
    return mem


def _sub_rng(rng) -> np.random.Generator:
    """A generator derived from the sampler's without advancing it (its state hashed): the memory's draws never shift the sampler's
    own random stream."""
    try:
        st = json.dumps(rng.bit_generator.state, sort_keys=True, default=str)
    except Exception:
        st = repr(rng)
    return np.random.default_rng(int(hashlib.sha1(st.encode()).hexdigest()[:15], 16))


def reorder(sampler, kind: str, order: Sequence[int], cfgs: Sequence[Optional[Dict[str, Any]]], rng, max_level: Optional[int] = None,
            mem: Optional[AliasMemory] = None) -> Tuple[List[int], Dict[int, Dict[str, Any]]]:
    """Compose the sampler's PRIOR ORDER with the memory: ``order`` (indices into ``cfgs``, the prior's order) is kept, except that
    the candidates of a down-weighted bin are deferred to the end (their relative order kept) unless the bin keeps its place this
    coupling, with probability ``max(floor, theta_bin / theta_rest)`` (one Thompson draw per bin and call). Returns the new order
    and per candidate index its provenance (``alias_memory`` extras: version, level, bin, weight, deferred)."""
    mem = mem if mem is not None else memory_for(sampler)
    if mem is None or not order:
        return list(order), {}
    _count(sampler, f"{kind}.consulted")
    dec: Dict[Tuple[Any, ...], Optional[Decision]] = {}
    per: Dict[int, Optional[Decision]] = {}
    for k in order:
        c = cfgs[k]
        if c is None:
            per[k] = None
            continue
        ck = tuple(sorted((a, b) for a, b in c.items() if not isinstance(b, (list, dict))))
        if ck not in dec:
            dec[ck] = mem.decide(kind, c, max_level=max_level)
        per[k] = dec[ck]
    downs = {(d.level, d.key): d for d in per.values() if d is not None and d.down}
    prov: Dict[int, Dict[str, Any]] = {}
    if not downs:
        for k, d in per.items():
            if d is not None:
                prov[int(k)] = {"version": mem.version, "level": d.level, "bin": list(d.key), "weight": 1.0, "deferred": False, "down": False}
        _count(sampler, f"{kind}.no_down")
        return list(order), prov
    g = _sub_rng(rng)
    keep: Dict[Tuple[int, Tuple[Any, ...]], Tuple[bool, float]] = {}
    for bk, d in sorted(downs.items(), key=lambda kv: str(kv[0])):
        th_b = float(g.beta(1.0 + d.s, 1.0 + d.f))
        th_r = float(g.beta(1.0 + d.rest_s, 1.0 + d.rest_f))
        w = float(min(1.0, max(mem.th.floor, th_b / max(th_r, 1e-9))))
        keep[bk] = (bool(g.random() < w), w)
    front, back = [], []
    order = [int(k) for k in order]
    for k in order:
        d = per.get(k)
        if d is not None and d.down:
            kept, w = keep[(d.level, d.key)]
            prov[k] = {"version": mem.version, "level": d.level, "bin": list(d.key), "weight": round(w, 4), "deferred": not kept, "down": True}
            (front if kept else back).append(k)
        else:
            if d is not None:
                prov[k] = {"version": mem.version, "level": d.level, "bin": list(d.key), "weight": 1.0, "deferred": False, "down": False}
            front.append(k)
    n_kept = sum(1 for k in front if (prov.get(k) or {}).get("down"))
    _count(sampler, f"{kind}.down_bins", len(downs))
    _count(sampler, f"{kind}.deferred", len(back))
    _count(sampler, f"{kind}.kept_down", n_kept)
    # i7-memory2: per hook, the candidates in flagged bins, those moved to the end, those a Thompson draw / the floor kept in place
    _count(sampler, f"{kind}.flagged", len(back) + n_kept)
    _count(sampler, f"{kind}.reordered", len(back))
    _count(sampler, f"{kind}.explored", n_kept)
    return front + back, prov


def pick_cfg(arm: Optional[str], stance) -> Dict[str, Any]:
    """The pick key of a pick (manipulation) stance candidate at sampling time - the hook's levels (arm x trunk x stand-off) of the key
    :func:`config_keys` reads from the sidecar's pick / manipulation stance (the grasp family is drawn later and only recorded)."""
    return {"arm": arm or "-", "stance": "manipulation", "stand_off": q(getattr(stance, "stand_off", None), 0.05),
            "translate": q(getattr(stance, "translate", None), 0.01), "tilt": q(getattr(stance, "tilt", None), 0.01)}


def reorder_pick(sampler, order: Sequence[int], stances: Sequence[Any], arm_of, rng) -> List[int]:
    """i7-memory2 (owner 2.3): the pick hook - ``order`` (indices into ``stances``, the sampler's prior order of its pick / manipulation
    stance candidates) through :func:`reorder` with :func:`pick_cfg` keys (``arm_of(stance)`` the working arm); the provenance goes into
    each candidate's ``extras["alias_memory_pick"]``. Memory off / nothing flagged: the order as given, no random number drawn."""
    order = [int(k) for k in order]
    new, prov = reorder(sampler, "pick", order, [pick_cfg(arm_of(st), st) for st in stances], rng, max_level=PICK_HOOK_LEVELS)
    for k, p in prov.items():
        ex = getattr(stances[int(k)], "extras", None)
        if isinstance(ex, dict):
            ex["alias_memory_pick"] = p
    return new


def place_cfg(arm: Optional[str], stance) -> Dict[str, Any]:
    """The place key of a stance-C candidate at sampling time (the same fields :func:`config_keys` reads from the sidecar)."""
    return {"arm": arm or "-", "stance": "placement", "stand_off": q(getattr(stance, "stand_off", None), 0.05),
            "translate": q(getattr(stance, "translate", None), 0.01), "tilt": q(getattr(stance, "tilt", None), 0.01)}


# ---------------------------------------------------------------------------------------------------------------- aggregate

def job_dirs(root: str) -> List[str]:
    return sorted(glob.glob(os.path.join(root, "out", "*", "*", "g*")))


def collect_rows(root: str, backfill: bool = True, write_backfill: bool = False, rules=None) -> List[Dict[str, Any]]:
    """Every job's shard under ``root``/out; a finished job without a shard is back-filled (computed; written into the job dir
    only with ``write_backfill``)."""
    rules = load_exclusions() if rules is None else rules
    rows: List[Dict[str, Any]] = []
    for d in job_dirs(root):
        shard = os.path.join(d, SHARD_NAME)
        if os.path.exists(shard):
            for line in open(shard):
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
            continue
        if not backfill or not os.path.exists(os.path.join(d, "results.jsonl")) or not job_finished(d):
            continue
        rs = demo_rows(d, rules)
        for r in rs:
            r["backfilled"] = True
        rows += rs
        if write_backfill:
            try:
                record_job(d)
            except OSError:
                pass
    return rows


def store_root(root: str, mode: str = "production") -> str:
    """The memory store of a mode under a root (``<root>/alias_memory`` / ``<root>/alias_memory_validation``)."""
    if mode not in MODES:
        raise ValueError(f"alias memory mode {mode!r}: one of {MODES}")
    return os.path.join(root, STORE_DIRS[mode])


def store_thresholds(mem_root: str, mode: str = "production") -> Thresholds:
    """The thresholds the store's versions are built with: ``<store>/thresholds.json`` (the owner's file: it wins), written with the
    mode's defaults (:meth:`Thresholds.for_mode`) when it does not exist yet. A file of the other mode is refused (separate stores)."""
    path = os.path.join(mem_root, "thresholds.json")
    d = _json(path)
    if d is None:
        th = Thresholds.for_mode(mode)
        os.makedirs(mem_root, exist_ok=True)
        tmp = path + f".tmp{os.getpid()}"
        with open(tmp, "w") as f:
            json.dump(dataclasses.asdict(th), f, indent=1)
        os.replace(tmp, path)
        return th
    th = Thresholds.from_dict(d)
    if th.mode != mode:
        raise SystemExit(f"alias memory: {path} is a {th.mode} store's thresholds - not a {mode} store (the stores are separate)")
    return th


def resolve_version(path: str) -> str:
    """A version dir (its meta.json), or a memory root's LATEST version."""
    if os.path.exists(os.path.join(path, "meta.json")):
        return os.path.abspath(path)
    v = latest(path)
    if v is None:
        raise SystemExit(f"alias memory: {path} is neither a version (meta.json) nor a memory root with a LATEST version")
    return os.path.abspath(v)


def seed_rows(seed_version: str) -> List[Dict[str, Any]]:
    """Every demo of a seed version, marked ``seed`` (weighted ``seed_weight``)."""
    tag = "seed:" + os.path.join(os.path.basename(os.path.dirname(os.path.abspath(seed_version))), os.path.basename(os.path.abspath(seed_version)))
    out = []
    for f in sorted(glob.glob(os.path.join(seed_version, "aliases", "*.json"))):
        for r in (_json(f) or {}).get("rows") or []:
            out.append({**r, "seed": tag})
    return out


def store_rows(rows: Sequence[Dict[str, Any]], mode: str, seed_version: Optional[str] = None) -> List[Dict[str, Any]]:
    """The demos a mode's store holds (i7-memory2, owner 2.2 / 2.4). Production: only demos of production-mode jobs (it starts empty).
    Validation: the seed's demos and those of jobs from before the modes (``mode`` unset; both marked ``seed``, the reduced weight), and
    the validation-mode jobs' demos. A demo of one mode never enters the other mode's store."""
    if mode == "production":
        return [r for r in rows if r.get("mode") == "production"]
    out: List[Dict[str, Any]] = []
    seen = set()
    if seed_version:
        for r in seed_rows(seed_version):
            out.append(r)
            seen.add((r.get("job"), r.get("inst")))
    for r in rows:
        k = (r.get("job"), r.get("inst"))
        if k in seen:
            continue
        if r.get("mode") == "validation":
            out.append(r)
        elif r.get("mode") is None:
            out.append({**r, "seed": "legacy"})
        seen.add(k)
    return out


def next_version(mem_root: str) -> str:
    os.makedirs(mem_root, exist_ok=True)
    n = 1 + max([int(m.group(1)) for m in (re.match(r"v(\d{4})$", x) for x in os.listdir(mem_root)) if m] or [0])
    while True:
        p = os.path.join(mem_root, f"v{n:04d}")
        try:
            os.mkdir(p)
            return p
        except FileExistsError:
            n += 1


def write_version(mem_root: str, rows: Sequence[Dict[str, Any]], thresholds: Thresholds, *, batch: Optional[str] = None,
                  sources: Sequence[str] = (), gpu_s: Optional[Dict[str, float]] = None, seed: Optional[str] = None) -> str:
    """A new immutable version: per alias its rows (sorted by finish time) and snapshot order, meta.json, the guardrail report."""
    path = next_version(mem_root)
    os.makedirs(os.path.join(path, "aliases"))
    by_alias: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_alias[r["alias"]].append(r)
    for a, rs in by_alias.items():
        rs.sort(key=lambda r: str(r.get("finished") or ""))
        with open(os.path.join(path, "aliases", f"{a}.json"), "w") as f:
            json.dump({"schema": SCHEMA, "alias": a, "snap_order": _snap_order(rs), "rows": rs}, f, default=str)
    meta = {"schema": SCHEMA, "version": os.path.basename(path), "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "batch": batch,
            "thresholds": dataclasses.asdict(thresholds), "n_rows": len(rows), "n_aliases": len(by_alias), "sources": list(sources),
            "exclusions_file": EXCLUSIONS_FILE, "mode": thresholds.mode, "seed": seed,
            "n_seeded": sum(1 for r in rows if r.get("seed"))}
    with open(os.path.join(path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    rep = guardrails(rows, thresholds, batch=batch, gpu_s=gpu_s)
    rep["version"] = meta["version"]
    with open(os.path.join(path, "report.json"), "w") as f:
        json.dump(rep, f, indent=1, default=str)
    with open(os.path.join(path, "report.md"), "w") as f:
        f.write(report_md(rep))
    tmp = os.path.join(mem_root, f"LATEST.tmp{os.getpid()}")
    with open(tmp, "w") as f:
        f.write(os.path.basename(path) + "\n")
    os.replace(tmp, os.path.join(mem_root, "LATEST"))
    return path


def latest(mem_root: str) -> Optional[str]:
    try:
        v = open(os.path.join(mem_root, "LATEST")).read().strip()
    except OSError:
        return None
    p = os.path.join(mem_root, v)
    return p if os.path.isdir(p) else None


def sacct_gpu_s(jobids: Iterable[str]) -> Dict[str, float]:
    ids = sorted({str(j) for j in jobids if j})
    out: Dict[str, float] = {}
    for i in range(0, len(ids), 200):
        try:
            res = subprocess.run(["sacct", "-j", ",".join(ids[i:i + 200]), "-X", "-n", "-o", "JobID,ElapsedRaw"], capture_output=True, text=True,
                                 timeout=120).stdout
        except (OSError, subprocess.SubprocessError):
            return out
        for line in res.splitlines():
            p = line.split()
            if len(p) == 2 and p[1].isdigit():
                out[p[0]] = float(p[1])
    return out


def batch_pending(root: str, batch: str, ledger: Optional[str] = None) -> List[str]:
    """The job ids of the batch (the ledger's rows with this tag) still queued or running: the aggregation waits for the last one."""
    ledger = ledger or os.path.join(root, "manifests", "allocations.tsv")
    ids = set()
    try:
        lines = open(ledger).read().splitlines()
        cols = lines[0].split("\t")
        for line in lines[1:]:
            r = dict(zip(cols, line.split("\t")))
            if r.get("tag") == batch and r.get("jobid"):
                ids.add(r["jobid"].strip())
    except (OSError, IndexError):
        return []
    try:
        res = subprocess.run(["squeue", "-h", "-u", os.environ.get("USER", ""), "-o", "%i"], capture_output=True, text=True, timeout=120).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return sorted(ids & {x.strip() for x in res.splitlines()})


def aggregate(root: str, mem_root: Optional[str] = None, batch: Optional[str] = None, thresholds: Optional[Thresholds] = None,
              write_backfill: bool = False, gpu: bool = True, mode: str = "production", seed: Optional[str] = None,
              rows: Optional[Sequence[Dict[str, Any]]] = None) -> str:
    """A new version of the mode's store (default ``<root>/alias_memory`` / ``<root>/alias_memory_validation``) from the root's demos
    (:func:`store_rows`) with the store's thresholds (:func:`store_thresholds`). Validation: the seed version is pinned in the store's
    ``seed.json`` at its first aggregation (``seed`` or :data:`VALIDATION_SEED`, a root's LATEST resolved) and kept after."""
    mem_root = mem_root or store_root(root, mode)
    th = thresholds or store_thresholds(mem_root, mode)
    if th.mode != mode:
        raise SystemExit(f"alias memory: thresholds of mode {th.mode} for a {mode} store")
    seed_v = None
    if mode == "validation":
        sj = os.path.join(mem_root, "seed.json")
        pinned = (_json(sj) or {}).get("seed")
        if pinned:
            if seed and resolve_version(seed) != pinned:
                raise SystemExit(f"alias memory: {mem_root} is seeded from {pinned}, not {seed}")
            seed_v = pinned
        else:
            seed_v = resolve_version(seed or VALIDATION_SEED)
            os.makedirs(mem_root, exist_ok=True)
            with open(sj, "w") as f:
                json.dump({"seed": seed_v, "weight": th.seed_weight, "pinned": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}, f, indent=1)
    all_rows = list(rows) if rows is not None else collect_rows(root, write_backfill=write_backfill)
    mine = store_rows(all_rows, mode, seed_v)
    gpu_s = sacct_gpu_s(r.get("jobid") for r in mine) if gpu else None
    return write_version(mem_root, mine, th, batch=batch, sources=[root] + ([seed_v] if seed_v else []), gpu_s=gpu_s, seed=seed_v)


# ---------------------------------------------------------------------------------------------------------------- guardrails

def latest_context(rows: Sequence[Dict[str, Any]]) -> Optional[Dict[str, str]]:
    rs = [r for r in rows if r.get("ctx")]
    if not rs:
        return None
    return dict(max(rs, key=lambda r: str(r.get("finished") or ""))["ctx"])


def guardrails(rows: Sequence[Dict[str, Any]], th: Thresholds, batch: Optional[str] = None, gpu_s: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
    """The per-batch report per alias (decisions in the alias's LATEST context): the place / pick bins down-weighted, the share of the
    observed configuration space they hold (bins and demos), the yield and GPU-h per success of the batch vs before, alerts, and the
    cross-alias shared-fix candidates. Alerts only - nothing acts on them."""
    by_alias: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_alias[r["alias"]].append(r)
    out: Dict[str, Any] = {"batch": batch, "thresholds": dataclasses.asdict(th), "aliases": {}, "alerts": [], "shared_fix_candidates": []}
    shared: Dict[Tuple[str, Tuple[Any, ...]], List[str]] = defaultdict(list)
    for a, rs in sorted(by_alias.items()):
        rs = sorted(rs, key=lambda r: str(r.get("finished") or ""))
        mem = AliasMemory(a, rs, th)
        ctx = latest_context(rs)
        mem.weights(ctx)
        ent: Dict[str, Any] = {"context": ctx, "n_demos": len(rs), "excluded": dict(Counter(r["excluded"] for r in rs if r.get("excluded")))}
        for kind in ("place", "pick"):
            downs = mem.down_bins(kind, max_level=PLACE_HOOK_LEVELS if kind == "place" else None)
            bins = mem._build(kind)
            lv = PLACE_HOOK_LEVELS - 1 if kind == "place" else 1
            space = [b for (li, _), b in bins.items() if li == lv]
            n_obs = sum(b.n_raw for b in space)
            levels = PLACE_LEVELS if kind == "place" else PICK_LEVELS
            down_space = [b for b in space if (mem.decide(kind, dict(zip(levels[lv], b.key)), max_level=lv + 1) or Decision(False, 0, (), 0, 0, 0, 0, 0, 0, 0, 0)).down]
            ent[kind] = {"down": [dict(d.facts(), failures=dict(bins[(d.level, d.key)].failures), fields=list(levels[d.level])) for d in downs],
                         "bins_observed": len(space), "bins_down": len(down_space),
                         "space_share_bins": round(len(down_space) / len(space), 3) if space else 0.0,
                         "space_share_demos": round(sum(b.n_raw for b in down_space) / n_obs, 3) if n_obs else 0.0}
            for d in downs:
                cross = tuple(x for x in d.key[:3])  # arm, translate, tilt: the configuration across aliases
                shared[(kind, cross)].append(a)
            if ent[kind]["space_share_demos"] > th.alert_space_share:
                out["alerts"].append({"alias": a, "kind": "space_lost", "what": kind, "share": ent[kind]["space_share_demos"]})
        # yield / GPU-h: this batch vs before (counted rows only)
        cnt = [r for r in rs if not r.get("excluded")]
        cur = [r for r in cnt if batch is not None and r.get("tag") == batch]
        before = [r for r in cnt if batch is None or r.get("tag") != batch]

        def y(xs):
            return (sum(1 for r in xs if r["outcome"] == "success"), len(xs))

        def g(xs):
            if not gpu_s:
                return None
            jobs = {r.get("jobid") for r in xs if r.get("jobid")}
            sec = sum(gpu_s.get(j, 0.0) for j in jobs)
            ok = sum(1 for r in xs if r["outcome"] == "success")
            return round(sec / 3600.0 / ok, 3) if ok else None
        ent["yield_before"], ent["yield_batch"] = y(before), y(cur)
        ent["gpuh_per_success_before"], ent["gpuh_per_success_batch"] = g(before), g(cur)
        sb, nb = ent["yield_before"]
        sc, nc = ent["yield_batch"]
        if nc >= th.alert_min_demos and sc == 0:
            out["alerts"].append({"alias": a, "kind": "all_failed", "batch": batch, "n": nc,
                                  "failures": dict(Counter(f"{r.get('stage')}:{r.get('failure_type')}" for r in cur)),
                                  "note": "everything failed - a bug, not a bad configuration (nothing is down-weighted for it)"})
        elif nc >= th.alert_min_demos and nb >= th.alert_min_demos and sc / nc < sb / nb - th.alert_yield_drop:
            out["alerts"].append({"alias": a, "kind": "yield_drop", "batch": batch, "before": [sb, nb], "batch_yield": [sc, nc]})
        ent["sampled_with_memory"] = sum(1 for r in cur if r.get("sampled_with"))
        out["aliases"][a] = ent
    for (kind, cross), aliases in sorted(shared.items(), key=lambda kv: str(kv[0])):
        if len(set(aliases)) >= th.shared_fix_aliases:
            out["shared_fix_candidates"].append({"kind": kind, "config": list(cross), "aliases": sorted(set(aliases))})
    return out


def report_md(rep: Dict[str, Any]) -> str:
    L = [f"# alias memory {rep.get('version', '')} ({rep['thresholds'].get('mode', 'production')}) - batch {rep.get('batch')}", "",
         f"thresholds: {json.dumps(rep['thresholds'])}", ""]
    L.append("## alerts (owner; nothing acts on them)")
    L += [f"- {json.dumps(a)}" for a in rep["alerts"]] or ["- none"]
    L.append("")
    L.append("## shared-fix candidates (the same configuration down-weighted across aliases)")
    L += [f"- {c['kind']} {c['config']}: {', '.join(c['aliases'])}" for c in rep["shared_fix_candidates"]] or ["- none"]
    L.append("")
    L.append("## per alias")
    for a, e in rep["aliases"].items():
        downs = e["place"]["down"] + e["pick"]["down"]
        L.append(f"- {a}: demos {e['n_demos']} yield before {e['yield_before']} batch {e['yield_batch']} GPU-h/success "
                 f"{e['gpuh_per_success_before']} -> {e['gpuh_per_success_batch']}; place down {e['place']['bins_down']}/{e['place']['bins_observed']} "
                 f"({e['place']['space_share_demos']:.0%} of demos), pick down {e['pick']['bins_down']}/{e['pick']['bins_observed']}")
        for kind in ("place", "pick"):
            for d in e[kind]["down"]:
                L.append(f"  - {kind} {dict(zip(d['fields'], d['bin']))}: n {d['n']} s {d['s']} f {d['f']} vs rest {d['rest_s']}/{d['rest_f']} "
                         f"P(worse) {d['p_worse']} mean {d['mean']} failures {d['failures']}")
        _ = downs
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------------------------------------------------------- replay

def replay(root: str, out_dir: str, thresholds: Optional[Thresholds] = None, rows: Optional[List[Dict[str, Any]]] = None,
           seed: Optional[str] = None) -> Dict[str, Any]:
    """Offline replay: the tags of ``root``/out in finish order are the batches; after each one a version is built; for every later
    demo, what the memory of the version before it decided for its place / pick bin at the hooks' levels (the predictive check:
    down-weighted bins should fail later, healthy ones should not be caught; ``hist`` = the bin's raw record at that moment). Writes
    rows.jsonl, replay.json, final_report.md. With ``seed`` (a version): what a store seeded from it decides for every alias in its
    latest context at a NEW snapshot (``seeded``: the bins a validation run would defer on its first attempt)."""
    th = thresholds or Thresholds()
    rows = rows if rows is not None else collect_rows(root)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "rows.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=str) + "\n")
    tag_t: Dict[str, str] = {}
    for r in rows:
        t = str(r.get("finished") or "")
        if t and (r["tag"] not in tag_t or t > tag_t[r["tag"]]):
            tag_t[r["tag"]] = t
    tags = [t for t, _ in sorted(tag_t.items(), key=lambda kv: kv[1])]
    seen: List[Dict[str, Any]] = []
    pred = []
    for tag in tags:
        cur = [r for r in rows if r["tag"] == tag]
        by_alias = defaultdict(list)
        for r in seen:
            by_alias[r["alias"]].append(r)
        for r in cur:
            if r.get("excluded"):
                continue
            hist = by_alias.get(r["alias"])
            if not hist:
                continue
            mem = AliasMemory(r["alias"], hist, th)
            mem.weights(r.get("ctx"))
            for kind in ("place", "pick"):
                cfg, obs = r.get(kind), r.get(f"{kind}_obs")
                if cfg is None or obs is None:
                    continue
                d = mem.decide(kind, cfg, max_level=PLACE_HOOK_LEVELS if kind == "place" else PICK_HOOK_LEVELS)
                if d is not None and d.down:
                    b = mem._build(kind)[(d.level, d.key)]
                    pred.append({"tag": tag, "alias": r["alias"], "inst": r["inst"], "kind": kind, "bin": list(d.key), "level": d.level, "obs": obs,
                                 "failure_type": r.get("failure_type"), "stage": r.get("stage"), "facts": d.facts(),
                                 "hist": {"n": b.n_raw, "ok": b.s_raw}})
        seen += cur
    final = guardrails(rows, th, batch=tags[-1] if tags else None)
    summ = {k: {"later_flagged": sum(1 for p in pred if p["kind"] == k), "failed": sum(1 for p in pred if p["kind"] == k and not p["obs"]),
                "findings": sorted({json.dumps([p["alias"], p["bin"]]) for p in pred if p["kind"] == k})} for k in ("place", "pick")}
    res = {"tags": tags, "n_rows": len(rows), "thresholds": dataclasses.asdict(th), "summary": summ, "predictions": pred, "final": final,
           "by_context": by_context(rows, th)}
    if seed:
        res["seeded"] = seeded_findings(seed, th)
    with open(os.path.join(out_dir, "replay.json"), "w") as f:
        json.dump(res, f, indent=1, default=str)
    with open(os.path.join(out_dir, "final_report.md"), "w") as f:
        f.write(report_md(final))
    return res


def seeded_findings(seed: str, th: Thresholds, snapshot: str = "trunk_NEW") -> List[Dict[str, Any]]:
    """Per alias of a seed version: the place / pick bins (hook levels) a store seeded from it down-weights in the alias's latest
    context at a new snapshot - what a validation run's first attempt defers."""
    v = resolve_version(seed)
    rows = seed_rows(v)
    by_alias: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_alias[r["alias"]].append(r)
    out = []
    for a, rs in sorted(by_alias.items()):
        ctx = latest_context(rs)
        if not ctx:
            continue
        mem = AliasMemory(a, rs, th)
        mem.weights(dict(ctx, snapshot=snapshot))
        for kind in ("place", "pick"):
            for d in mem.down_bins(kind, max_level=PLACE_HOOK_LEVELS if kind == "place" else PICK_HOOK_LEVELS):
                b = mem._build(kind)[(d.level, d.key)]
                out.append({"alias": a, "kind": kind, **d.facts(), "fields": list((PLACE_LEVELS if kind == "place" else PICK_LEVELS)[d.level]),
                            "hist": {"n": b.n_raw, "ok": b.s_raw}, "failures": dict(b.failures)})
    return out


def by_context(rows: Sequence[Dict[str, Any]], th: Thresholds) -> Dict[str, List[Dict[str, Any]]]:
    """Per alias and per context (customs version x entry / map keys) it ran in: what a job submitted right after that context's last
    demo, in that context, would have down-weighted (place at the hook's levels; pick at every level, recorded only)."""
    by_alias: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_alias[r["alias"]].append(r)
    out: Dict[str, List[Dict[str, Any]]] = {}
    for a, rs in sorted(by_alias.items()):
        rs = sorted(rs, key=lambda r: str(r.get("finished") or ""))
        ctxs: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for r in rs:
            c = r.get("ctx") or {}
            ctxs[(c.get("spec"), c.get("keys"))] = {"ctx": dict(c), "until": str(r.get("finished") or ""), "tags": sorted(set(
                x["tag"] for x in rs if (x.get("ctx") or {}).get("spec") == c.get("spec") and (x.get("ctx") or {}).get("keys") == c.get("keys")))}
        lst = []
        for (spec, keys), e in ctxs.items():
            hist = [r for r in rs if str(r.get("finished") or "") <= e["until"]]
            mem = AliasMemory(a, hist, th)
            mem.weights(e["ctx"])
            n_place = sum(1 for r, w in zip(hist, mem.w) if w > 0 and r.get("place_obs") is not None)
            n_pick = sum(1 for r, w in zip(hist, mem.w) if w > 0 and r.get("pick_obs") is not None)
            lst.append({"context": e["ctx"], "tags": e["tags"], "n_place_obs": n_place, "n_pick_obs": n_pick,
                        "place_down": [dict(d.facts(), fields=list(PLACE_LEVELS[d.level]), failures=dict(mem._build("place")[(d.level, d.key)].failures))
                                       for d in mem.down_bins("place", max_level=PLACE_HOOK_LEVELS)],
                        "pick_down": [dict(d.facts(), fields=list(PICK_LEVELS[d.level]), failures=dict(mem._build("pick")[(d.level, d.key)].failures))
                                      for d in mem.down_bins("pick")]})
        out[a] = lst
    return out


# ---------------------------------------------------------------------------------------------------------------- CLI

def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="alias_memory", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("record", help="write a job's shard")
    s.add_argument("job_dir")
    s = sub.add_parser("aggregate", help="merge the shards of a root into a new version")
    s.add_argument("--root", required=True)
    s.add_argument("--memory", default=None, help="the memory root (default <root>/alias_memory)")
    s.add_argument("--batch", default=None, help="the batch's tag (the report's 'batch' column)")
    s.add_argument("--write-backfill", action="store_true")
    s.add_argument("--no-gpu", action="store_true")
    s.add_argument("--if-batch-done", action="store_true", help="exit without a version while a job of the batch (ledger tag) is queued / running")
    s.add_argument("--mode", default="production", choices=MODES, help="i7-memory2: the store (production: <root>/alias_memory, production-mode "
                                                                        "jobs only; validation: <root>/alias_memory_validation, seeded)")
    s.add_argument("--seed", default=None, help="validation: the seed version / memory root at the store's first aggregation (default "
                                                + VALIDATION_SEED + ")")
    s = sub.add_parser("replay", help="offline replay over a root (batches = tags)")
    s.add_argument("--root", required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--thresholds", default=None, help="JSON of Thresholds fields")
    s.add_argument("--rows", default=None, help="reuse a rows.jsonl of an earlier replay (no job dirs read)")
    s.add_argument("--mode", default="production", choices=MODES, help="i7-memory2: the mode's thresholds (Thresholds.for_mode), --thresholds on top")
    s.add_argument("--seed", default=None, help="also report what a store seeded from this version defers at a new snapshot")
    s = sub.add_parser("show", help="a version's decisions for an alias")
    s.add_argument("version")
    s.add_argument("alias")
    a = ap.parse_args(argv)
    if a.cmd == "record":
        print(record_job(a.job_dir))
    elif a.cmd == "aggregate":
        if a.if_batch_done and a.batch:
            pend = batch_pending(a.root, a.batch)
            if pend:
                print(f"ALIAS_MEMORY_BATCH_PENDING {a.batch}: {len(pend)} job(s) still queued / running ({','.join(pend[:10])}); the last one aggregates")
                return 0
        p = aggregate(a.root, a.memory, a.batch, write_backfill=a.write_backfill, gpu=not a.no_gpu, mode=a.mode, seed=a.seed)
        print(f"ALIAS_MEMORY_VERSION {p}")
    elif a.cmd == "replay":
        th = Thresholds.from_dict({**dataclasses.asdict(Thresholds.for_mode(a.mode)), **(json.loads(a.thresholds) if a.thresholds else {})})
        rows = [json.loads(x) for x in open(a.rows)] if a.rows else None
        r = replay(a.root, a.out, th, rows=rows, seed=a.seed)
        print(f"replay ({th.mode}): {r['n_rows']} rows, {len(r['tags'])} batches, {len(r['predictions'])} later demos in down-weighted bins "
              f"(place {r['summary']['place']['later_flagged']} / {r['summary']['place']['failed']} failed, pick {r['summary']['pick']['later_flagged']} / "
              f"{r['summary']['pick']['failed']} failed){'; seeded findings ' + str(len(r['seeded'])) if 'seeded' in r else ''} -> {a.out}")
    elif a.cmd == "show":
        mem = load_version(a.version, a.alias)
        if mem is None:
            print("no memory for", a.alias)
            return 1
        mem.weights(latest_context(mem.rows))
        for kind in ("place", "pick"):
            for d in mem.down_bins(kind):
                print(kind, json.dumps(d.facts()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
