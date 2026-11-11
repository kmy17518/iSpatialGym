import json
import pytest

from scripts.campaign_20261110.contracts import ContractError
from scripts.campaign_20261110.controller_held import source_environment


def source(tmp_path, *, split="eval_model", model="isgm10", flag="1", closure=True):
    provider = tmp_path / "assets/custom_mental_v1/objects/classroom_mock_up/isgm10"
    (provider / "usd").mkdir(parents=True)
    (provider / "usd/isgm10.usd").write_text("native provider bytes")
    config = tmp_path / "source_config.json"
    config.write_text(json.dumps({"split": split, "template": {
        "task": "mental_rotation", "class_id": "mental_rotation-classroom_mock_up",
        "objects": [{"role": "target", "category": "classroom_mock_up", "model": model}]}}))
    job = tmp_path / "source_job"; job.mkdir()
    values = {"ISPATIALGYM_CAMPAIGN_CONFIG": str(config), "START_PER_ATTEMPT": "0",
              "ISPATIALGYM_HELD_BOUND_FLOOR_START": flag}
    if closure: values["ISGF_TRANSPORT_ASSET_CLOSURE"] = str(provider)
    (job / "job.env").write_text("".join(f"{key}={value}\n" for key, value in values.items()))
    return job, provider


def test_verified_classroom_source_retains_floor_and_provider_for_grouped_refill(tmp_path):
    job, provider = source(tmp_path)
    values, template_customs = source_environment(job, root=tmp_path)
    assert template_customs
    assert "ISPATIALGYM_HELD_BOUND_FLOOR_START=1" in values
    assert "ISGF_TRANSPORT_ASSET_CLOSURE=" + str(provider) in values


@pytest.mark.parametrize("changes,reason", [
    ({"split": "train"}, "outside exact source scope"),
    ({"model": "another"}, "outside exact source scope"),
    ({"flag": "0"}, "bound-floor flag differs"),
    ({"closure": False}, "provider closure differs"),
])
def test_classroom_transport_flags_fail_closed_for_changed_scope_or_missing_source(tmp_path, changes, reason):
    job, _ = source(tmp_path, **changes)
    with pytest.raises(ContractError, match=reason): source_environment(job, root=tmp_path)


def test_missing_provider_never_releases_a_source_that_would_repeat_staging_failure(tmp_path):
    job, provider = source(tmp_path)
    (provider / "usd/isgm10.usd").unlink()
    with pytest.raises(ContractError, match="provider closure differs"):
        source_environment(job, root=tmp_path)
