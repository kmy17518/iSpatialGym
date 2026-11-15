"""Replay-time lighting perturbation for the batched kinematic replayer (owner 2026-11-15).

Manipulation collection is physics-only (no rendering), so the lighting drawn at collection never affected a trajectory or an
outcome. Lighting therefore becomes a replay-time perturbation like the scene removals: the recorded collection lighting is NOT
applied or verified (campaign restoration runs with an empty lighting request, i.e. every fixture at its default); instead each
replayed episode draws a fresh value, deterministic in (seed, demo, replay index), and applies it to the whole replay scene - the
crop and, in the merged scene, the neighbour rooms' fixtures.

Design for the global skybox (one per simulator): **fixtures per env, skybox per batch**. A slot's fixture lights belong to its own
scene, so each env gets its own intensity factor and colour tint; the skybox gets one draw per batch, keyed by (seed, batch key
= the sorted demo names of the run), applied once before the first frame and never changed while the batch runs. Every episode
record carries both; ``check_batch_skybox`` refuses a run whose episodes would need two skyboxes (the replay-time analogue of the F4
guard; ``--lighting-pert-from`` with records of different batches fails there). Indoors the fixtures dominate the light, so envs
differ visibly; batches differ additionally in sky/window light.

The draw keeps ``campaign_augmentation.lighting_draw``'s form (with probability ``p`` a Gaussian factor around 1, clipped) with the
range as parameters, because the collection-time range (sigma 0.12, [0.7, 1.3]) changes rendered brightness by only ~1.5 %.
Camera relocalization is exempt (its goal image is rendered at collection): its recorded lighting is restored as before.
No simulator import at module load.
"""

from __future__ import annotations

import hashlib
import json
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

SCHEMA = "isg-replay-lighting-v1"

#: owner 2026-11-15: intensity x0.5-1.5 from a normal centred at 1. sigma 0.17 puts the clip bounds at 2.9 sigma (0.4 % of draws
#: clipped), 68 % of draws within +-17 %, 95 % within +-34 %: large changes stay possible but rare (long tail)
DEFAULTS = {"p": 0.5, "sigma": 0.17, "lo": 0.5, "hi": 1.5, "color_sigma": 0.04, "color_lo": 0.88, "color_hi": 1.12,
            "sky_sigma": 0.17, "sky_lo": 0.5, "sky_hi": 1.5, "sky_color_sigma": 0.03, "sky_color_lo": 0.92, "sky_color_hi": 1.08}


def _key(*parts) -> int:
    return int(hashlib.sha256("|".join([SCHEMA] + [str(p) for p in parts]).encode()).hexdigest()[:16], 16)


def _clip(x, lo, hi):
    return min(float(hi), max(float(lo), float(x)))


def draw_fixtures(seed: int, demo_name: str, replay_index: int, cfg: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    """One env's fixture draw: {} (defaults) with probability 1 - p, else an intensity factor and per-channel colour factors."""
    c = {**DEFAULTS, **(cfg or {})}
    rng = random.Random(_key(seed, "fixtures", demo_name, replay_index))
    if rng.random() >= float(c["p"]):
        return {"perturbed": False, "intensity_factor": 1.0, "color_factors": [1.0, 1.0, 1.0]}
    return {"perturbed": True, "intensity_factor": _clip(rng.gauss(1.0, c["sigma"]), c["lo"], c["hi"]),
            "color_factors": [_clip(rng.gauss(1.0, c["color_sigma"]), c["color_lo"], c["color_hi"]) for _ in range(3)]}


def batch_key(demo_names: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(sorted(demo_names)).encode()).hexdigest()[:16]


def draw_skybox(seed: int, key: str, cfg: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    c = {**DEFAULTS, **(cfg or {})}
    rng = random.Random(_key(seed, "skybox", key))
    if rng.random() >= float(c["p"]):
        return {"perturbed": False, "intensity_factor": 1.0, "color_factors": [1.0, 1.0, 1.0], "batch_key": key}
    return {"perturbed": True, "intensity_factor": _clip(rng.gauss(1.0, c["sky_sigma"]), c["sky_lo"], c["sky_hi"]),
            "color_factors": [_clip(rng.gauss(1.0, c["sky_color_sigma"]), c["sky_color_lo"], c["sky_color_hi"]) for _ in range(3)],
            "batch_key": key}


def check_batch_skybox(records: Sequence[Mapping[str, Any]]) -> None:
    sky = {json.dumps(r["skybox"], sort_keys=True) for r in records if r.get("skybox") is not None}
    if len(sky) > 1:
        raise ValueError("replay lighting: one batch needs one skybox draw (the skybox is global); records mix " + str(len(sky)))


def load_from_records(path: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    """``--lighting-pert-from``: (demo name, replay index) -> the lighting record of earlier REPLAY_SCENE_PERT lines."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("REPLAY_SCENE_PERT "):
                line = line[len("REPLAY_SCENE_PERT "):]
            rec = json.loads(line)
            if rec.get("tag", "REPLAY_SCENE_PERT") != "REPLAY_SCENE_PERT" or not rec.get("lighting"):
                continue
            out[(rec["demo_name"], int(rec["replay_index"]))] = rec["lighting"]
    return out


# ----------------------------------------------------------------------------------------------------------------------------
# simulator side
# ----------------------------------------------------------------------------------------------------------------------------

def _walk(prim):
    yield prim
    for child in prim.GetChildren():
        yield from _walk(child)


def fixture_attrs(scene) -> Dict[str, Tuple[Any, Any]]:
    """{obj:subpath: (intensity attr, colour attr or None)} of every light under the scene's objects (as campaign_augmentation._lights)."""
    out = {}
    for obj in scene.objects:
        prefix = str(obj.prim.GetPath())
        for prim in _walk(obj.prim):
            if "Light" not in prim.GetTypeName():
                continue
            attr = prim.GetAttribute("inputs:intensity")
            if attr and attr.Get() is not None:
                col = prim.GetAttribute("inputs:color")
                out[obj.name + ":" + str(prim.GetPath())[len(prefix):]] = (attr, col if col and col.Get() is not None else None)
    return out


def fixture_defaults(scene) -> Dict[str, Dict[str, Any]]:
    """The scene's fixture defaults, captured once (call before anything changes a light; the campaign light defaults agree)."""
    d = getattr(scene, "_isg_replay_light_defaults", None)
    if d is None:
        d = {}
        for name, (ia, ca) in fixture_attrs(scene).items():
            d[name] = {"intensity": float(ia.Get()), "color": None if ca is None else [float(x) for x in ca.Get()]}
        scene._isg_replay_light_defaults = d
    return d


def sky_defaults(sim) -> Dict[str, Any]:
    sky = sim.skybox
    d = getattr(sky, "_isg_replay_defaults", None)
    if d is None:
        # campaign restoration may have captured the untouched skybox first; prefer that capture
        base = getattr(sky, "_isg_campaign_defaults", None)
        d = dict(base) if base else {"intensity": float(sky.intensity), "color": [float(x) for x in sky.color]}
        sky._isg_replay_defaults = d
    return d


def apply_fixtures(scene, sim, draw: Mapping[str, Any]) -> Dict[str, Any]:
    """Every fixture of ``scene`` = its default x the draw (intensity factor, colour factors); verified; returns a summary."""
    import omnigibson.lazy as lazy

    defaults = fixture_defaults(scene)
    attrs = fixture_attrs(scene)
    if set(attrs) != set(defaults):
        raise ValueError("replay lighting: fixture set changed within a loaded scene")
    f = float(draw["intensity_factor"])
    cf = [float(x) for x in draw["color_factors"]]
    with sim.editing_usd():
        for name, (ia, ca) in attrs.items():
            ia.Set(defaults[name]["intensity"] * f)
            if ca is not None and defaults[name]["color"] is not None:
                ca.Set(lazy.pxr.Gf.Vec3f(*[min(1.0, a * b) for a, b in zip(defaults[name]["color"], cf)]))
    for name, (ia, _) in attrs.items():
        want = defaults[name]["intensity"] * f
        if abs(float(ia.Get()) - want) > 1e-5 * max(1.0, abs(want)):
            raise ValueError(f"replay lighting not applied: {name}")
    vals = [defaults[n]["intensity"] * f for n in attrs]
    return {"fixtures": len(attrs), "intensity_range": [min(vals, default=None), max(vals, default=None)]}


def apply_skybox(sim, draw: Mapping[str, Any]) -> Dict[str, Any]:
    d = sky_defaults(sim)
    sky = sim.skybox
    sky.intensity = d["intensity"] * float(draw["intensity_factor"])
    sky.color = tuple(a * float(b) for a, b in zip(d["color"], draw["color_factors"]))
    return {"intensity": float(sky.intensity), "color": [float(x) for x in sky.color]}


def ignored_recorded(meta: Optional[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    """The collection-time lighting request a replay draw replaces (for the record)."""
    if not meta:
        return None
    return dict(((meta.get("campaign") or {}).get("perturbations") or {}).get("lighting") or {})


def without_lighting(meta: Mapping[str, Any]) -> Dict[str, Any]:
    """A campaign TRO block whose restoration leaves lighting at the defaults (the recorded lighting is not applied or verified)."""
    out = json.loads(json.dumps(meta))
    camp = out["campaign"]
    camp["perturbations"]["lighting"] = {}
    aug = (camp.get("runtime") or {}).get("augmentation")
    if isinstance(aug, dict):
        aug.pop("lighting", None)
    return out


# ----------------------------------------------------------------------------------------------------------------------------
# replayed-instance TROs (owner 2026-11-15): the replay-time perturbations live in the instance record like the collection-time ones
# ----------------------------------------------------------------------------------------------------------------------------

REPLAY_BLOCK = "replay_perturbations"
REPLAY_BLOCK_SCHEMA = "isg-replay-perturbations-v1"


def replay_block(rec: Mapping[str, Any]) -> Dict[str, Any]:
    """The ``campaign.replay_perturbations`` block of one replayed episode, from its REPLAY_SCENE_PERT record."""
    light = rec.get("lighting") or None
    return {"schema": REPLAY_BLOCK_SCHEMA, "replay_index": int(rec["replay_index"]), "demo_name": rec["demo_name"],
            "scene": {"schema": rec.get("schema"), "seed": rec.get("seed"), "scope": rec.get("scope"), "scene_mode": rec.get("scene_mode"),
                      "removed_objects": list(rec.get("removed") or []), "perturbed": bool(rec.get("perturbed"))},
            "lighting": None if light is None else {"schema": light.get("schema"), "seed": light.get("seed"), "cfg": light.get("cfg"),
                                                    "fixtures_draw": light["fixtures_draw"], "skybox": light["skybox"],
                                                    "recorded_ignored": light.get("recorded_ignored")}}


def write_replayed_tro(src_tro: str, out_dir: str, rec: Mapping[str, Any]) -> str:
    """A copy of the instance's TRO with ``ispatialgym.campaign.replay_perturbations`` (the collection-time ``perturbations`` untouched):
    a later load of this instance (``--replay-tro-dir``) reproduces the replayed scene - removals, fixture and skybox lighting."""
    import os

    with open(src_tro) as f:
        tro = json.load(f)
    camp = tro["ispatialgym"].setdefault("campaign", {})
    block = replay_block(rec)
    from scripts.campaign_20261110.contracts import file_digest

    block["source_tro"] = {"path": src_tro, "sha256": file_digest(src_tro)}
    camp[REPLAY_BLOCK] = block
    os.makedirs(out_dir, exist_ok=True)
    name = os.path.basename(src_tro)
    if int(rec["replay_index"]):
        name = name.replace("-tro_state.json", f"-r{int(rec['replay_index'])}-tro_state.json")
    path = os.path.join(out_dir, name)
    with open(path, "w") as f:
        json.dump(tro, f, indent=1)
    return path


def load_replayed_tros(directory: str) -> Tuple[Dict[Tuple[str, int], List[str]], Dict[Tuple[str, int], Dict[str, Any]]]:
    """``--replay-tro-dir``: (demo name, replay index) -> removed objects, and -> lighting record, from replayed-instance TROs."""
    import glob
    import os

    removed, lighting = {}, {}
    for p in sorted(glob.glob(os.path.join(directory, "*-tro_state.json"))):
        with open(p) as f:
            block = (json.load(f)["ispatialgym"].get("campaign") or {}).get(REPLAY_BLOCK)
        if not block:
            continue
        if block.get("schema") != REPLAY_BLOCK_SCHEMA:
            raise ValueError(f"{p}: unknown replay perturbation schema")
        k = (block["demo_name"], int(block["replay_index"]))
        removed[k] = list(block["scene"]["removed_objects"])
        if block.get("lighting"):
            lighting[k] = {"fixtures_draw": block["lighting"]["fixtures_draw"], "skybox": block["lighting"]["skybox"]}
    return removed, lighting


def sweep_stats(rgb) -> Dict[str, float]:
    """Clipped-black / clipped-white pixel fractions and mean luminance (Rec. 709, 0-255) of an rgb uint8 image."""
    import numpy as np

    a = np.asarray(rgb)[..., :3].astype(np.float64)
    lum = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]
    return {"black_frac": float((a.max(axis=2) <= 2).mean()), "white_frac": float((a.min(axis=2) >= 253).mean()),
            "mean_luma": float(lum.mean())}
