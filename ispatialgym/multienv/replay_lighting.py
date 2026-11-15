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

DEFAULTS = {"p": 0.5, "sigma": 0.3, "lo": 0.4, "hi": 1.8, "color_sigma": 0.06, "color_lo": 0.82, "color_hi": 1.18,
            "sky_sigma": 0.3, "sky_lo": 0.4, "sky_hi": 1.8, "sky_color_sigma": 0.04, "sky_color_lo": 0.9, "sky_color_hi": 1.1}


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
