"""A campaign cannot silently replay at the default scale."""

import json
from dataclasses import asdict
from types import SimpleNamespace

import pytest

from ispatialgym.sampling import campaign_runtime as runtime
from scripts.campaign_20261110.contracts import (
    ContractError, ObjectRef, Perturbations, ScaleState, Template, ValidationKey,
    attach_tro_metadata, digest,
)


def fixture(scale=(0.9, 0.9, 2.25), split="train"):
    obj = ObjectRef("target", "target", "book", "abc", digest("asset"), (1, 1, 2.5),
                    ready=True, readiness_source="test-fixture")
    template = Template("t1", "object_scaling", "book", split, "house_single_floor", "room0", (obj,), digest("bddl"))
    p = Perturbations({"target": ScaleState(obj.default_scale, scale)})
    key = ValidationKey(template.fingerprint, digest("code"), digest("physics"), digest("data"), p.scale_fingerprint)
    return template, p, key


def test_restored_scale_is_verified_and_mismatch_rejected():
    template, p, key = fixture()
    tro = attach_tro_metadata({"ispatialgym": {}}, campaign_id="fresh", split="train", template=template,
                              created_at=1, seed=1, perturbations=p, validation_key=key)
    meta = tro["ispatialgym"]["campaign"]
    assert runtime.verify_loaded(meta, {"target": SimpleNamespace(scale=[0.9, 0.9, 2.25])})["scale_verified"]
    with pytest.raises(ContractError, match="scale mismatch"):
        runtime.verify_loaded(meta, {"target": SimpleNamespace(scale=[1, 1, 2.5])})
    with pytest.raises(ContractError, match="absent"):
        runtime.verify_loaded(meta, {})


def test_eval_cannot_smuggle_changed_scale():
    template, p, key = fixture()
    tro = attach_tro_metadata({"ispatialgym": {}}, campaign_id="fresh", split="train", template=template,
                              created_at=1, seed=1, perturbations=p, validation_key=key)
    meta = tro["ispatialgym"]["campaign"]
    meta["split"] = "eval_standard"
    with pytest.raises(ContractError, match="no perturbations"):
        runtime.verify_loaded(meta, {"target": SimpleNamespace(scale=[0.9, 0.9, 2.25])})


def test_failed_draw_records_unavailable_proposal_without_stale_pose(tmp_path, monkeypatch):
    template, p, key = fixture()
    config = {"campaign_id": "fresh", "template": asdict(template), "perturbations": asdict(p),
              "validation_key": asdict(key), "split": "train"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setenv("ISPATIALGYM_CAMPAIGN_CONFIG", str(path))
    sampler = SimpleNamespace(ctx=SimpleNamespace(out_dir=str(tmp_path)))
    error = SimpleNamespace(reason="no_stance", details={"tries": 2})
    runtime.record_failure(sampler, None, index=3, attempt=4, seed=5, phase="sample", error=error)
    record = json.loads((tmp_path / "campaign_sampling_failures.jsonl").read_text())
    assert record["proposal_available"] is False
    assert record["start_poses"] is record["goal_poses"] is None
    assert record["failure_kind"] == "sampling"
    assert record["diagnostics"] == {"tries": 2}


def test_unapplied_scene_or_light_metadata_is_rejected():
    template, p, key = fixture()
    tro = attach_tro_metadata({"ispatialgym": {}}, campaign_id="fresh", split="train", template=template,
                              created_at=1, seed=1, perturbations=p, validation_key=key)
    meta = tro["ispatialgym"]["campaign"]
    meta["perturbations"]["lighting"] = {"fixture": 0.8}
    with pytest.raises(ContractError, match="runtime adapter"):
        runtime.verify_loaded(meta, {"target": SimpleNamespace(scale=[0.9, 0.9, 2.25])})
