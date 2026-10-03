"""ispatialgym.oracle.shard_format: the effective format and the resume check."""

import json
import os

import pytest

from ispatialgym.oracle import shard_format as SF


def _shard(tmp_path, name, depth):
    d = tmp_path / name
    (d / "meta").mkdir(parents=True)
    feats = {"observation.rgb.zed_link_camera_0": {}, "action": {}}
    if depth:
        feats["observation.depth_linear.zed_link_camera_0"] = {}
    (d / "meta" / "info.json").write_text(json.dumps({"features": feats}))
    return str(d)


def test_effective():
    assert SF.effective({}) == {"name": "nodepth_nvenc", "depth": False, "rgb_encoder": "hevc_nvenc,qp=33,preset=p7"}
    assert SF.effective({"ISPATIALGYM_SHARD_FORMAT": "legacy"}) == {"name": "legacy", "depth": True, "rgb_encoder": "x265"}
    assert SF.effective({"ISPATIALGYM_SHARD_FORMAT": "legacy", "ISPATIALGYM_SHARD_NO_DEPTH": "1"})["name"] == "custom"
    assert SF.effective({"ISPATIALGYM_SHARD_RGB_ENCODER": "x265"}) == {"name": "custom", "depth": False, "rgb_encoder": "x265"}
    with pytest.raises(ValueError):
        SF.effective({"ISPATIALGYM_SHARD_FORMAT": "nope"})


def test_resume_same_and_different(tmp_path):
    new, legacy = SF.effective({}), SF.effective({"ISPATIALGYM_SHARD_FORMAT": "legacy"})
    s = _shard(tmp_path, "shard_a", depth=False)
    SF.write_marker(s, new)
    SF.check_resume(s, new)
    with pytest.raises(RuntimeError):
        SF.check_resume(s, legacy)
    with pytest.raises(RuntimeError):
        SF.check_resume(s, SF.effective({"ISPATIALGYM_SHARD_RGB_ENCODER": "x265"}))


def test_resume_without_marker(tmp_path):
    new, legacy = SF.effective({}), SF.effective({"ISPATIALGYM_SHARD_FORMAT": "legacy"})
    prod = _shard(tmp_path, "shard_prod", depth=True)
    with pytest.raises(RuntimeError):
        SF.check_resume(prod, new)
    SF.check_resume(prod, legacy)
    assert SF.read_marker(prod) == {"name": "legacy", "depth": True, "rgb_encoder": "x265"}
    nodepth = _shard(tmp_path, "shard_nodepth", depth=False)
    for fmt in (new, legacy):
        with pytest.raises(RuntimeError):
            SF.check_resume(nodepth, fmt)
    assert not os.path.exists(SF.marker_path(nodepth))
