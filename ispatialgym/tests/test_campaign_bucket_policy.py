"""Bucket relaxation cannot waive meaningful camera detours or finite paths."""
from ispatialgym.sampling.v2.tasks.camera_relocalization import detour_failure
import pytest
from ispatialgym.sampling.campaign_bucket_policy import (
    POLICY, PREPARATION_ACTIVATION, PREPARATION_ENV, PREPARATION_SELECTOR, active,
)


def test_cross_bucket_path_retains_detour_requirement():
    assert detour_failure(3.2, 2.9, 3., .2) == "too_long"
    assert detour_failure(3.2, 2.9, 3., .2, relaxed=True) is None
    assert detour_failure(3.1, 3., 3., .2, relaxed=True) == "short_detour"
    for value in (None, float("inf"), float("nan")):
        assert detour_failure(value, 2.9, 3., .2, relaxed=True) == "disconnected"


def test_policy_is_opt_in_for_ordinary_camera_runs(monkeypatch):
    monkeypatch.delenv("ISPATIALGYM_CAMPAIGN_CONFIG", raising=False)
    monkeypatch.delenv(PREPARATION_ENV, raising=False)
    assert active() is False
    assert POLICY["bucket_ranges_enforced"] is False
    assert POLICY["bucket_quotas_enforced"] is False


def test_preparation_without_collection_contract_uses_owner_policy(monkeypatch):
    monkeypatch.delenv("ISPATIALGYM_CAMPAIGN_CONFIG", raising=False)
    monkeypatch.setenv(PREPARATION_ENV, PREPARATION_SELECTOR)
    assert active() is True
    assert PREPARATION_ACTIVATION["policy"] == POLICY
    assert PREPARATION_ACTIVATION["acceptance_authorization"] is False
    assert detour_failure(3.2, 2.9, 3., .2, relaxed=active()) is None
    assert detour_failure(3.1, 3., 3., .2, relaxed=active()) == "short_detour"


@pytest.mark.parametrize("selector", ["", "relaxed", "campaign_20261110:descriptive:v1"])
def test_unknown_preparation_policy_fails_closed(monkeypatch, selector):
    monkeypatch.delenv("ISPATIALGYM_CAMPAIGN_CONFIG", raising=False)
    monkeypatch.setenv(PREPARATION_ENV, selector)
    with pytest.raises(ValueError, match="Unknown"):
        active()


def test_preparation_policy_cannot_override_another_campaign(monkeypatch):
    from ispatialgym.sampling import campaign_runtime
    monkeypatch.setenv(PREPARATION_ENV, PREPARATION_SELECTOR)
    monkeypatch.setattr(campaign_runtime, "active", lambda: True)
    monkeypatch.setattr(campaign_runtime, "configuration", lambda: ({"campaign_id": "other"}, None))
    with pytest.raises(ValueError, match="conflicts"):
        active()


def test_existing_campaign_configuration_still_activates_policy(monkeypatch):
    from ispatialgym.sampling import campaign_runtime
    monkeypatch.delenv(PREPARATION_ENV, raising=False)
    monkeypatch.setattr(campaign_runtime, "active", lambda: True)
    monkeypatch.setattr(campaign_runtime, "configuration", lambda: ({"campaign_id": POLICY["campaign_id"]}, None))
    assert active() is True
