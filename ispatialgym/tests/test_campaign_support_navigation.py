"""Inserted table footprints persist in template navigation maps and fail closed."""
from types import SimpleNamespace

import pytest

from ispatialgym.sampling import scene_variants as SV


def fixture():
    obj=SimpleNamespace(category="pedestal_table",model="tdpuit",scale=[1.,1.,1.],aabb=([1.,2.,0.],[2.,3.,.7]))
    scene=SimpleNamespace(object_registry=lambda kind,name: obj if name=="table" else None)
    metadata={"name":"table","model_key":"pedestal_table/tdpuit","default_scale":[1.,1.,1.],
              "blocked_footprints":{"free":[],"block":[[[.98,1.98],[2.02,1.98],[2.02,3.02],[.98,3.02]]]}}
    return scene,metadata


def test_historical_template_map_is_unchanged(monkeypatch):
    monkeypatch.setattr(SV,"block_trav_map",lambda *_:pytest.fail("historical map was changed"))
    assert SV.patch_trav_map_from_meta(None,{})==0


def test_added_support_is_blocked_after_freed_chair_cells(monkeypatch):
    scene,metadata=fixture();calls=[]
    monkeypatch.setattr(SV,"patch_trav_map",lambda *args:calls.append("free") or 7)
    monkeypatch.setattr(SV,"block_trav_map",lambda *args:calls.append("block") or 25)
    assert SV.patch_trav_map_from_meta(scene,{"scene_variant":{"freed_footprints":{"free":[1]},"campaign_added_support":metadata}})==7
    assert calls==["free","block"]


@pytest.mark.parametrize("change",["missing_object","wrong_model","wrong_scale","undersized_footprint","oversized_footprint","crossed_footprint","nonfinite"])
def test_invalid_support_metadata_cannot_silently_change_map(monkeypatch,change):
    scene,metadata=fixture()
    if change=="missing_object":metadata["name"]="absent"
    elif change=="wrong_model":metadata["model_key"]="pedestal_table/other"
    elif change=="wrong_scale":metadata["default_scale"]=[1.,1.,1.2]
    elif change=="undersized_footprint":metadata["blocked_footprints"]["block"][0]=[[1.1,2.1],[1.9,2.1],[1.9,2.9],[1.1,2.9]]
    elif change=="oversized_footprint":metadata["blocked_footprints"]["block"][0]=[[0.,1.],[3.,1.],[3.,4.],[0.,4.]]
    elif change=="crossed_footprint":metadata["blocked_footprints"]["block"][0]=[[.98,1.98],[2.02,3.02],[2.02,1.98],[.98,3.02]]
    else:metadata["blocked_footprints"]["block"][0][0][0]=float("nan")
    monkeypatch.setattr(SV,"block_trav_map",lambda *_:pytest.fail("invalid metadata reached rasterizer"))
    with pytest.raises(ValueError):SV.block_campaign_support(scene,metadata)
