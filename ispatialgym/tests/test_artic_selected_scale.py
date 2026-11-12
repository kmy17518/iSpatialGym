"""Selected articulation geometry must be born at the sampler's exact scale."""

from types import SimpleNamespace

import numpy as np
import pytest

from ispatialgym.grasping import artic_motions as AM
from ispatialgym.grasping import object_geometry as OG
from ispatialgym.grasping import schema as S
from ispatialgym.grasping import validate_artic as VA


def entry(scale):
    return {"roles": {"target": {"category": "blender", "representative": "cwkvib", "joint": "j_lid", "scale": scale}}}


@pytest.mark.parametrize("spec,expected", [
    (None, None), (.9316, [.9316] * 3), ([1.1, .9, 1.2], [1.1, .9, 1.2]),
    ({"blender/cwkvib": .9316, "cwkvib": .7, "default": 1.2}, [.9316] * 3),
    ({"cwkvib": .8115, "default": 1.2}, [.8115] * 3),
    ({"default": 1.0554}, [1.0554] * 3), ({"unrelated": .8}, None),
])
def test_scale_uses_native_role_resolution(spec, expected):
    assert AM.target_scale(entry(spec)) == expected


@pytest.mark.parametrize("spec", [[1, 2], 0, -1, float("nan"), float("inf")])
def test_invalid_explicit_scale_fails_before_world_load(spec):
    with pytest.raises(ValueError):
        AM.target_scale(entry(spec))


@pytest.mark.parametrize("scale", [None, .9316, [1.1, .9, 1.2]])
def test_setup_keys_load_geometry_and_motion_to_the_selected_scale(monkeypatch, tmp_path, scale):
    selected = entry(scale)
    obj = SimpleNamespace(scale=np.array(S.canonical_scale(scale)), joints={"j_lid": SimpleNamespace(lower_limit=0, upper_limit=.87)})
    calls = {}
    def load(job, directory):
        calls.update(job=job, directory=directory)
        return obj
    mw = VA.MotionWorld.__new__(VA.MotionWorld)
    mw.w = SimpleNamespace(load=load, unload=lambda o: calls.update(unloaded=o), settle=lambda *a: ({}, "state"))
    mw.profile = lambda *a: {}
    mw.refresh_probe = lambda *a: {}
    mw.rest = lambda *a: ({}, "state", [])
    monkeypatch.setattr(AM, "load_alias", lambda *a: selected)
    monkeypatch.setattr(OG, "load", lambda d: calls.setdefault("geometry", d))
    monkeypatch.setattr(VA, "local_hull", lambda o: [])
    monkeypatch.setattr(VA, "load_annotation", lambda k: {})
    def build(*args, **kwargs):
        calls["motion_scale"] = kwargs["scale"]
        return {"face_yaw_offset": 0}
    monkeypatch.setattr(AM, "build", build)
    result = mw.setup("selected", str(tmp_path))
    expected_dir = S.model_dir("blender", "cwkvib", scale, str(tmp_path), "articulation")
    assert result["dir"] == calls["geometry"] == calls["directory"] == expected_dir
    assert calls["job"]["id"] == S.entry_id("blender", "cwkvib", scale, "articulation")
    assert calls["job"]["scale"] == AM.target_scale(selected)
    assert calls["motion_scale"] == list(S.canonical_scale(scale))
    assert "unloaded" not in calls
    if scale is not None:
        obj.scale = np.ones(3)
        with pytest.raises(ValueError, match="requested"):
            mw.setup("selected", str(tmp_path))
        assert calls["unloaded"] is obj
