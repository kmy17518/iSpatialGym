"""Prevent training augmentation leaks and replay state drift."""

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest

from ispatialgym.sampling import campaign_augmentation as aug
from scripts.campaign_20261110.contracts import ContractError


class Attr:
    def __init__(self, value):
        self.value = value

    def Get(self):
        return self.value

    def Set(self, value):
        self.value = value


class Prim:
    def __init__(self, path, kind, intensity=None, children=()):
        self.path, self.kind, self.children = path, kind, children
        self.attr = Attr(intensity)

    def GetPath(self):
        return self.path

    def GetTypeName(self):
        return self.kind

    def GetChildren(self):
        return self.children

    def GetAttribute(self, key):
        assert key == "inputs:intensity"
        return self.attr


def light_world(prefix="/World/scene_0"):
    light = Prim(prefix + "/lamp/link/light", "SphereLight", 1200)
    obj = SimpleNamespace(name="lamp", prim=Prim(prefix + "/lamp", "Xform", children=[light]))
    scene = SimpleNamespace(objects=[obj])
    sim = SimpleNamespace(skybox=SimpleNamespace(intensity=2500, color=(1.07, 0.85, 0.61)), editing_usd=nullcontext)
    return scene, sim, light


def test_lighting_replay_is_absolute_and_default_episode_resets_it():
    request = {"schema": aug.LIGHT_SCHEMA, "intensity_factor": 0.8, "skybox_color_factors": [0.97, 1, 1.03]}
    scene, sim, light = light_world()
    record = aug.apply_lighting(scene, sim, request)
    assert light.attr.Get() == 960
    assert aug.apply_lighting(scene, sim, request, record) == record
    assert light.attr.Get() == 960  # Never multiply the previous perturbation.
    scene2, sim2, light2 = light_world("/World/scene_3")
    assert aug.apply_lighting(scene2, sim2, request, record) == record
    assert light2.attr.Get() == 960
    assert aug.apply_lighting(scene, sim, {}) == {}
    assert light.attr.Get() == 1200 and sim.skybox.intensity == 2500
    assert sim.skybox.color == (1.07, 0.85, 0.61)


def test_lighting_rejects_different_defaults_and_out_of_bounds_request():
    request = {"schema": aug.LIGHT_SCHEMA, "intensity_factor": 1.2, "skybox_color_factors": [1, 1, 1]}
    scene, sim, _ = light_world()
    record = aug.apply_lighting(scene, sim, request)
    fresh, fresh_sim, changed = light_world()
    changed.attr.Set(1500)
    with pytest.raises(ContractError, match="defaults"):
        aug.apply_lighting(fresh, fresh_sim, request, record)
    with pytest.raises(ContractError, match="bounds"):
        aug.validate_lighting({**request, "intensity_factor": 1.4})


def test_skybox_properties_open_their_own_usd_edit_context():
    scene, sim, _ = light_world()
    active = False

    @contextmanager
    def editing():
        nonlocal active
        assert not active, "Cannot nest editing_usd() contexts"
        active = True
        try:
            yield
        finally:
            active = False

    class Sky:
        _intensity, _color = 2500, (1.07, .85, .61)

        @property
        def intensity(self):
            return self._intensity

        @intensity.setter
        def intensity(self, value):
            with editing():
                self._intensity = value

        @property
        def color(self):
            return self._color

        @color.setter
        def color(self, value):
            with editing():
                self._color = value

    sim.editing_usd, sim.skybox = editing, Sky()
    aug.apply_lighting(scene, sim, {"schema": aug.LIGHT_SCHEMA, "intensity_factor": .8, "skybox_color_factors": [1, 1, 1]})
    assert sim.skybox.intensity == 2000


def test_scene_removals_protect_bound_and_structural_objects():
    stable = {"objects_info": {"init_info": {name: {"args": {"category": cat, "in_rooms": ["room"]}}
               for name, cat in [("table", "coffee_table"), ("wall", "walls"),
                                 ("door", "door"), ("window", "fixed_window"), ("clutter", "chair")]}}}
    assert aug.checked_removals(stable, ["clutter"], ["table"]) == ["clutter"]
    for name in ["table", "wall", "door", "window", "unknown"]:
        with pytest.raises(ContractError):
            aug.checked_removals(stable, [name], ["table"])
    for seed in range(30):
        assert set(aug.removal_candidates(stable, rooms=["room"], protected=["table"], seed=seed)) <= {"clutter"}


def test_parking_checks_scene_coordinates_fixed_base_and_exact_pose():
    obj = SimpleNamespace(category="chair", fixed_base=True,
                          get_position_orientation=lambda frame: ((-100, -101, -102), (0, 0, 0, 1)))
    scene = SimpleNamespace(object_registry=lambda key, name: obj)
    moved = {"clutter": {"moved_to": [-100, -101, -102]}}
    assert aug.verify_parking(scene, ["clutter"], moved)["clutter"] == [-100, -101, -102]
    obj.fixed_base = False
    with pytest.raises(ContractError, match="fixed"):
        aug.verify_parking(scene, ["clutter"], moved)
    obj.fixed_base = True
    with pytest.raises(ContractError, match="task-relevant"):
        aug.verify_parking(scene, ["clutter"], moved, ["clutter"])


def test_vector_batch_cannot_silently_use_another_groups_skybox():
    meta = {"perturbations": {"lighting": {"schema": aug.LIGHT_SCHEMA, "intensity_factor": 0.9,
                                            "skybox_color_factors": [1, 1, 1]}}}
    aug.require_shared_skybox([meta, meta])
    with pytest.raises(ContractError, match="shared skybox"):
        aug.require_shared_skybox([meta, None])
