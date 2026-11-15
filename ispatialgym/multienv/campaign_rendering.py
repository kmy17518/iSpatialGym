"""Manifest-bound restoration for kinematic campaign observation rendering."""
import json
import os
from pathlib import Path

from scripts.campaign_20261110.contracts import file_digest, require, validate_campaign_block


def selected_records(manifest, demo_dirs, *, sequential_augmentation=False, slots=1, replay_lighting=False):
    """Select accepted episodes only, with one exact rendering configuration."""
    records = {}
    for line in Path(manifest).read_text().splitlines():
        row = json.loads(line)
        directory = str(Path(row["artifacts"]["demo"]["path"]).resolve().parent)
        require(directory not in records, "duplicate demo in accepted manifest")
        records[directory] = row
    wanted = [str(Path(path).resolve()) for path in demo_dirs]
    probe_repeat = (os.environ.get('ISPATIALGYM_CAMPAIGN_RESTORATION_PROBE') == '1'
                    and ((len(wanted) == 3 and wanted[0] == wanted[2] and wanted[0] != wanted[1])
                         or (sequential_augmentation and slots == 1 and len(wanted) == 2 and wanted[0] == wanted[1])))
    require(wanted and (len(set(wanted)) == len(wanted) or probe_repeat), "empty or duplicated campaign render input")
    require(all(path in records for path in wanted), "render input is not in accepted manifest")
    selected = {path: records[path] for path in wanted}
    if sequential_augmentation:
        from ispatialgym.oracle.campaign_batch import record_scope
        require(slots == 1, 'sequential lighting requires exactly one renderer slot')
        # Independent A-only/B-only references retain the full audited pair
        # contract, while DemoSource consumes only the requested episode list.
        record_scope(list(records.values()))
        return records
    else:
        require(len({row["render_compatibility_sha256"] for row in selected.values()}) == 1,
                "campaign rendering batch mixes scene, scale, light or validation configuration")
        if not replay_lighting:
            # Defense in depth: the skybox is global to the simulator, so a batch must share one lighting record even
            # when an exporter computed equal compatibility hashes (render_compatibility_sha256 is the only guard today).
            from ispatialgym.sampling.campaign_augmentation import require_shared_skybox
            require_shared_skybox([{"perturbations": row.get("perturbations") or {}} for row in selected.values()])
        # replay-time lighting (owner 2026-11-15): the recorded collection lighting is not rendered, so it cannot conflict; the
        # skybox guard is the replay's own one-draw-per-batch check (replay_lighting.check_batch_skybox / check_group_skybox)
    return selected


def checked_metadata(record, demo_dir):
    demo_dir = Path(demo_dir).resolve()
    require(Path(record["artifacts"]["demo"]["path"]).resolve() == demo_dir / "demo.hdf5", "manifest demo path mismatch")
    require({'tro', 'demo', 'report', 'sidecar'} <= set(record['artifacts']), 'incomplete render inputs')
    for kind, item in record["artifacts"].items():
        require(file_digest(item["path"]) == item["sha256"], f"render input {kind} changed after export")
    with open(record["artifacts"]["tro"]["path"]) as stream:
        meta = json.load(stream)["ispatialgym"]
    require(meta["instance_id"] == record["instance_id"] == demo_dir.name, "render instance identity mismatch")
    validate_campaign_block(meta["campaign"], object_bindings=meta["object_bindings"])
    require(meta["campaign"]["validation_key"] == record["validation_key"], "render validation key mismatch")
    require(meta["campaign"]["perturbations"] == record["perturbations"], "render perturbations differ from accepted manifest")
    return meta


class CropSceneView:
    """A merged (full) replay scene seen as the cropped world a demo was collected in: ``objects`` are the crop's objects, every
    other attribute is the scene's. Campaign restoration (parking, lighting with its exact recorded fixture set, camera context)
    runs against this view exactly as in the recorded crop. Attributes set on the view (lighting defaults) stay on the view."""

    def __init__(self, scene, crop_names):
        object.__setattr__(self, "_scene", scene)
        object.__setattr__(self, "_crop", frozenset(crop_names))

    @property
    def objects(self):
        return [o for o in self._scene.objects if o.name in self._crop]

    def object_registry(self, *args, **kwargs):
        return self._scene.object_registry(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._scene, name)


def _full_light_defaults(scene):
    """Every fixture light's intensity of the merged scene, captured once before any campaign lighting touches it."""
    from ispatialgym.sampling.campaign_augmentation import _lights

    defaults = getattr(scene, "_isg_merged_light_defaults", None)
    if defaults is None:
        defaults = {name: float(attr.Get()) for name, attr in _lights(scene).items()}
        scene._isg_merged_light_defaults = defaults
    return defaults


def crop_view(scene, crop_names):
    """The slot's cached :class:`CropSceneView` for this crop, its lighting defaults seeded from the merged scene's untouched ones."""
    from ispatialgym.sampling.campaign_augmentation import _lights

    full = _full_light_defaults(scene)
    views = getattr(scene, "_isg_crop_views", None)
    if views is None:
        views = scene._isg_crop_views = {}
    key = frozenset(crop_names)
    if key not in views:
        view = CropSceneView(scene, key)
        object.__setattr__(view, "_isg_campaign_light_defaults", {n: full[n] for n in _lights(view)})
        views[key] = view
    return views[key]


def apply_outside_lighting(scene, sim, crop_names, request):
    """Merged scene: the fixture lights of objects outside the crop (never part of collection) get the demo's recorded intensity
    factor from their own untouched defaults, so the whole merged scene is lit by one draw (an unlit demo: defaults)."""
    from ispatialgym.sampling.campaign_augmentation import _lights, _near

    full = _full_light_defaults(scene)
    factor = float((request or {}).get("intensity_factor", 1.0))
    crop = frozenset(crop_names)
    attrs = {n: a for n, a in _lights(scene).items() if n.split(":", 1)[0] not in crop}
    require(set(attrs) <= set(full), "merged scene lighting fixture set changed")
    with sim.editing_usd():
        for name, attr in attrs.items():
            attr.Set(full[name] * factor)
    for name, attr in attrs.items():
        require(_near(attr.Get(), full[name] * factor), f"outside-crop fixture intensity not applied: {name}")
    return {"outside_fixtures": len(attrs), "factor": factor,
            "applied_range": [min((full[n] * factor for n in attrs), default=None), max((full[n] * factor for n in attrs), default=None)]}


def restore_campaign_scene(scene, sim, meta, crop_names=None, ignore_lighting=False):
    """Call after loading the first recorded state, before rendering any frame. ``crop_names`` (merged replay of a cropped demo:
    the recorded crop's object names): restoration runs against the crop view and the outside-crop fixtures get the same light."""
    from ispatialgym.sampling.campaign_runtime import verify_loaded

    merged = crop_names is not None
    target = crop_view(scene, crop_names) if merged else scene
    if ignore_lighting:
        # replay-time lighting (owner 2026-11-15): the collection lighting is neither applied nor verified; fixtures stay at their
        # defaults here and the replay's own draw is applied by the caller
        from ispatialgym.multienv.replay_lighting import without_lighting

        meta = without_lighting(meta)
    physical = {}
    for instance in meta["campaign"]["perturbations"]["scales"]:
        binding = meta["object_bindings"][instance]
        name = binding.get("scene_object_name") if isinstance(binding, dict) else binding
        require(bool(name), f"missing campaign render binding: {instance}")
        obj = scene.object_registry("name", name)
        require(obj is not None, f"campaign render object missing: {name}")
        if isinstance(binding, dict):
            for field in ("category", "model"):
                if binding.get(field):
                    require(getattr(obj, field) == binding[field], f"campaign render {field} differs: {instance}")
        physical[instance] = obj
    result = verify_loaded(meta["campaign"], physical, scene=target, sim=sim)
    if merged and not ignore_lighting:
        result = dict(result)
        result["merged_lighting"] = apply_outside_lighting(scene, sim, crop_names,
                                                           meta["campaign"]["perturbations"].get("lighting") or {})
    if ignore_lighting:
        result = dict(result)
        result["recorded_lighting"] = "ignored: replay-time lighting"
    return result
