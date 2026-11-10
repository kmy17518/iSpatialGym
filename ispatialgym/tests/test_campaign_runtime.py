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


def test_campaign_metadata_passes_real_tro_parser_and_stays_strict():
    from ispatialgym.eval.targets import ISGTroMeta, SCHEMA_VERSION, TROValidationError

    template, p, key = fixture()
    base = {"ispatialgym": {"schema_version": SCHEMA_VERSION, "activity": "object_scaling",
                            "activity_definition_id": 0, "instance_id": "fresh-1",
                            "object_bindings": {"target": {"scene_object_name": "target_book_abc_0"}}}}
    tro = attach_tro_metadata(base, campaign_id="fresh", split="train", template=template,
                              created_at=1, seed=1, perturbations=p, validation_key=key)
    parsed = ISGTroMeta.from_dict(tro["ispatialgym"])
    assert parsed.raw["campaign"]["campaign_id"] == "fresh"
    tro["ispatialgym"]["campaign"]["validation_key"]["scale_sha256"] = digest("wrong")
    with pytest.raises(TROValidationError, match="scale"):
        ISGTroMeta.from_dict(tro["ispatialgym"])


def test_eval_cannot_smuggle_changed_scale():
    template, p, key = fixture()
    tro = attach_tro_metadata({"ispatialgym": {}}, campaign_id="fresh", split="train", template=template,
                              created_at=1, seed=1, perturbations=p, validation_key=key)
    meta = tro["ispatialgym"]["campaign"]
    meta["split"] = "eval_standard"
    with pytest.raises(ContractError, match="no perturbations"):
        runtime.verify_loaded(meta, {"target": SimpleNamespace(scale=[0.9, 0.9, 2.25])})


def test_failed_draw_records_unavailable_proposal_without_stale_pose(tmp_path, monkeypatch):
    import errno
    import fcntl

    def unsupported_lock(*args):
        raise OSError(errno.EIO, "NFS does not implement flock")

    monkeypatch.setattr(fcntl, "flock", unsupported_lock)
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
    runtime.record_failure(sampler, None, index=3, attempt=5, seed=6, phase="sample", error=error)
    rows = [json.loads(line) for line in (tmp_path / "campaign_sampling_failures.jsonl").read_text().splitlines()]
    assert [row["attempt"] for row in rows] == [4, 5]
    assert not (tmp_path / "campaign_sampling_failures.lockdir").exists()


def test_unapplied_scene_or_light_metadata_is_rejected():
    template, p, key = fixture()
    tro = attach_tro_metadata({"ispatialgym": {}}, campaign_id="fresh", split="train", template=template,
                              created_at=1, seed=1, perturbations=p, validation_key=key)
    meta = tro["ispatialgym"]["campaign"]
    meta["perturbations"]["lighting"] = {"fixture": 0.8}
    with pytest.raises(ContractError, match="runtime adapter"):
        runtime.verify_loaded(meta, {"target": SimpleNamespace(scale=[0.9, 0.9, 2.25])})


def test_native_attempt_journal_preserves_interrupted_draw_and_early_group_identity(tmp_path, monkeypatch):
    template, p, key = fixture()
    config = {"campaign_id": "fresh", "template": asdict(template), "perturbations": asdict(p),
              "validation_key": asdict(key), "split": "train"}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    plan = {"campaign_id": "fresh", "split": "train", "task": template.task,
            "contract_template_sha256": key.template_sha256, "group_id": "g1"}
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    monkeypatch.setenv("ISPATIALGYM_CAMPAIGN_CONFIG", str(path))
    monkeypatch.setenv("ISPATIALGYM_CAMPAIGN_GOAL_PLAN", str(plan_path))
    monkeypatch.delenv("ISPATIALGYM_CAMPAIGN_GOAL_QUEUE", raising=False)
    sampler = SimpleNamespace(ctx=SimpleNamespace(out_dir=str(tmp_path)))
    journal = tmp_path / "campaign_attempts.jsonl"
    runtime.attempt_finished(sampler, outcome="rejected", phase="sample", reason="campaign_group_timeout")
    assert not journal.exists()  # Pure pre-draw transitions have no native attempt.
    runtime.attempt_started(sampler, index=5, attempt=0, seed=7)
    started = json.loads(journal.read_text())
    assert started["goal_group_id"] == "g1" and started["validation_key"] == asdict(key)
    assert started["config_sha256"] == runtime.file_digest(path)
    # A stopped process at this point leaves a durable started row, not an invented failure/success.
    assert started["event"] == "started" and "outcome" not in started
    error = SimpleNamespace(reason="no_stance", details={})
    runtime.record_failure(sampler, None, index=5, attempt=0, seed=7, phase="sample", error=error)
    failure = json.loads((tmp_path / "campaign_sampling_failures.jsonl").read_text())
    assert failure["goal_group_id"] == "g1" and failure["native_attempt_id"] == started["attempt_id"]
    runtime.attempt_finished(sampler, outcome="rejected", phase="sample", reason="no_stance")
    rows = [json.loads(line) for line in journal.read_text().splitlines()]
    assert [r["event"] for r in rows] == ["started", "finished"]
    assert rows[1]["attempt_id"] == started["attempt_id"] and rows[1]["outcome"] == "rejected"
    runtime.attempt_started(sampler, index=5, attempt=1, seed=8)
    assert json.loads(journal.read_text().splitlines()[-1])["attempt_id"] != started["attempt_id"]


def test_unactivated_queue_cannot_claim_a_group_attempt(tmp_path, monkeypatch):
    template, p, key = fixture()
    config = {"campaign_id": "fresh", "template": asdict(template), "perturbations": asdict(p),
              "validation_key": asdict(key), "split": "train"}
    path = tmp_path / "config.json"; path.write_text(json.dumps(config))
    monkeypatch.setenv("ISPATIALGYM_CAMPAIGN_CONFIG", str(path))
    monkeypatch.setenv("ISPATIALGYM_CAMPAIGN_GOAL_QUEUE", str(tmp_path / "authored_queue.json"))
    sampler = SimpleNamespace(ctx=SimpleNamespace(out_dir=str(tmp_path)))
    with pytest.raises(ContractError, match="activated queue"):
        runtime.attempt_started(sampler, index=5, attempt=0, seed=7)
    assert not (tmp_path / "campaign_attempts.jsonl").exists()


def test_full_training_buckets_remain_eligible_when_another_is_underfilled():
    class CaptureRNG:
        def choice(self, count, p):
            assert count == 3
            assert all(probability > 0 for probability in p)
            assert p[2] > p[0]
            return 0

    buckets = [{"max_m": 3, "n": 3}, {"max_m": 5, "n": 4}, {"max_m": 10, "n": 3}]
    assert runtime.choose_soft_bucket(CaptureRNG(), buckets, {3.0: 10, 5.0: 4, 10.0: 0}) == 0
