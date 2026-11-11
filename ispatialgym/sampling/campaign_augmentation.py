"""Campaign-only scene parking and reproducible lighting adapters.

Parking is performed by the established scene-variant loader before any sampling
or visibility checks. Lights are changed before template sampling and restored
from exact recorded values on demo/replay load. No simulator import at module
load: contract checks also run in CPU manifest preparation.
"""

from __future__ import annotations

import math
import random

from scripts.campaign_20261110.contracts import canonical_json, require
from scripts.campaign_20261110.training_scene_policy import STRUCTURAL


LIGHT_SCHEMA = "isg-lighting-v1"


def lighting_draw(seed):
    """Candidate bounds; rendered validation is still required before release."""
    rng = random.Random(seed)
    if rng.random() >= 0.5:
        return {}
    return {"schema": LIGHT_SCHEMA,
            "intensity_factor": min(1.3, max(0.7, rng.gauss(1.0, 0.12))),
            "skybox_color_factors": [min(1.08, max(0.92, rng.gauss(1.0, 0.03))) for _ in range(3)]}


def validate_lighting(request):
    if not request:
        return
    require(set(request) == {"schema", "intensity_factor", "skybox_color_factors"}
            and request["schema"] == LIGHT_SCHEMA, "lighting runtime adapter requires an isg-lighting-v1 request")
    factor = request["intensity_factor"]
    colors = request["skybox_color_factors"]
    require(isinstance(factor, (float, int)) and math.isfinite(factor) and 0.7 <= factor <= 1.3,
            "lighting intensity factor is outside validated candidate bounds")
    require(isinstance(colors, (list, tuple)) and len(colors) == 3
            and all(isinstance(x, (int, float)) and math.isfinite(x) and 0.92 <= x <= 1.08 for x in colors),
            "invalid skybox color factors")


def removal_candidates(stable, *, rooms, protected, seed):
    rng = random.Random(seed)
    if rng.random() >= 0.5:
        return []
    protected, rooms = set(protected), set(rooms)
    candidates = []
    for name, info in sorted(stable["objects_info"]["init_info"].items()):
        args = info["args"]
        if name in protected or args.get("category") in STRUCTURAL or args.get("type", "").endswith("Robot"):
            continue
        if rooms and not rooms.intersection(args.get("in_rooms") or []):
            continue
        if rng.random() < 0.5:
            candidates.append(name)
    return candidates


def checked_removals(stable, names, protected):
    init = stable["objects_info"]["init_info"]
    protected = set(protected)
    for name in names:
        require(name in init, f"unknown campaign scene removal: {name}")
        require(name not in protected, f"cannot park task-relevant object: {name}")
        require(init[name]["args"].get("category") not in STRUCTURAL, f"cannot park structural object: {name}")
        require(not init[name]['args'].get('type', '').endswith('Robot'), f"cannot park a robot: {name}")
    return sorted(set(names))


def verify_parking(scene, names, moved, relevant=()):
    require(not set(names).intersection(relevant), "cannot park a task-relevant object")
    observed = {}
    for name in names:
        require(name in moved and "moved_to" in moved[name], f"campaign parking record absent: {name}")
        obj = scene.object_registry("name", name)
        require(obj is not None, f"campaign parked object absent from loaded scene: {name}")
        require(obj.category not in STRUCTURAL, f"cannot park structural object: {name}")
        pos, _ = obj.get_position_orientation(frame="scene")
        actual = [float(x) for x in pos]
        expected = moved[name]["moved_to"]
        require(len(actual) == len(expected) == 3 and all(abs(a - b) < 0.002 for a, b in zip(actual, expected)),
                f"campaign parked object moved or was not restored: {name}")
        require(obj.fixed_base, f"campaign parked object is not fixed: {name}")
        observed[name] = actual
    return observed


def _walk(prim):
    yield prim
    for child in prim.GetChildren():
        yield from _walk(child)


def _lights(scene):
    result = {}
    for obj in scene.objects:
        prefix = str(obj.prim.GetPath())
        for prim in _walk(obj.prim):
            if "Light" not in prim.GetTypeName():
                continue
            attr = prim.GetAttribute("inputs:intensity")
            if attr and attr.Get() is not None:
                key = obj.name + ":" + str(prim.GetPath())[len(prefix):]
                result[key] = attr
    return result


def _near(actual, expected):
    return math.isfinite(float(actual)) and abs(float(actual) - float(expected)) <= 1e-5 * max(1.0, abs(float(expected)))


def lighting_difference(actual, expected):
    """Describe restoration failures without hiding fixture-set or value drift."""
    actual_fixtures, wanted_fixtures = actual.get("fixtures", {}), expected.get("fixtures", {})
    shared = sorted(set(actual_fixtures) & set(wanted_fixtures))
    return {"missing_fixtures": sorted(set(wanted_fixtures)-set(actual_fixtures)),
            "extra_fixtures": sorted(set(actual_fixtures)-set(wanted_fixtures)),
            "changed_fixtures": {name: {"loaded": actual_fixtures[name], "recorded": wanted_fixtures[name]}
                                 for name in shared if actual_fixtures[name] != wanted_fixtures[name]},
            "skybox": {"loaded": actual.get("skybox"), "recorded": expected.get("skybox")},
            "request_matches": actual.get("request") == expected.get("request")}


def apply_lighting(scene, sim, request, expected=None):
    """Apply from captured defaults, never multiply the previous episode's value."""
    validate_lighting(request)
    if not request and not getattr(scene, "_isg_campaign_light_defaults", None) and not getattr(sim.skybox, "_isg_campaign_defaults", None):
        return {}
    attrs = _lights(scene)
    defaults = getattr(scene, "_isg_campaign_light_defaults", None)
    if defaults is None:
        defaults = {name: float(attr.Get()) for name, attr in attrs.items()}
        scene._isg_campaign_light_defaults = defaults
    require(set(defaults) == set(attrs), "lighting fixture set changed within a loaded scene")
    sky = sim.skybox
    require(sky is not None, "campaign lighting needs a loaded skybox")
    sky_default = getattr(sky, "_isg_campaign_defaults", None)
    if sky_default is None:
        sky_default = {"intensity": float(sky.intensity), "color": list(sky.color)}
        sky._isg_campaign_defaults = sky_default
    factor = float(request.get("intensity_factor", 1.0))
    color_factors = request.get("skybox_color_factors", [1.0] * 3)
    record = {"request": dict(request),
              "fixtures": {name: {"default": value, "applied": value * factor} for name, value in defaults.items()},
              "skybox": {"default": sky_default, "applied": {
                  "intensity": sky_default["intensity"] * factor,
                  "color": [a * b for a, b in zip(sky_default["color"], color_factors)]}}}
    if expected is not None:
        require(canonical_json(record) == canonical_json(expected),
                "recorded lighting differs from loaded fixture defaults: " + canonical_json(lighting_difference(record, expected)))
    with sim.editing_usd():
        for name, attr in attrs.items():
            attr.Set(record["fixtures"][name]["applied"])
    # LightObject setters enter editing_usd themselves; nesting that context is
    # rejected by the real simulator even though direct USD attributes need it.
    sky.intensity = record["skybox"]["applied"]["intensity"]
    sky.color = tuple(record["skybox"]["applied"]["color"])
    verify_lighting(scene, sim, record)
    return record if request else {}


def verify_lighting(scene, sim, record):
    if not record:
        return
    attrs = _lights(scene)
    require(set(attrs) == set(record["fixtures"]), "recorded lighting fixture set differs from loaded scene")
    for name, data in record["fixtures"].items():
        require(_near(attrs[name].Get(), data["applied"]), f"fixture intensity was not restored: {name}")
    sky = record["skybox"]["applied"]
    require(_near(sim.skybox.intensity, sky["intensity"]), "skybox intensity was not restored")
    require(all(_near(a, b) for a, b in zip(sim.skybox.color, sky["color"])), "skybox color was not restored")


def require_shared_skybox(metas):
    """The simulator has one skybox: vector batches must share its settings."""
    requests = [canonical_json((meta or {}).get("perturbations", {}).get("lighting", {})) for meta in metas]
    require(len(set(requests)) <= 1, "vector batch mixes lighting settings for the shared skybox; group by lighting configuration")
