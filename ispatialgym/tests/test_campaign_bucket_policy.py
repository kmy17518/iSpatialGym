"""Bucket relaxation cannot waive meaningful camera detours or finite paths."""
from ispatialgym.sampling.v2.tasks.camera_relocalization import detour_failure
from ispatialgym.sampling.campaign_bucket_policy import POLICY, active


def test_cross_bucket_path_retains_detour_requirement():
    assert detour_failure(3.2, 2.9, 3., .2) == "too_long"
    assert detour_failure(3.2, 2.9, 3., .2, relaxed=True) is None
    assert detour_failure(3.1, 3., 3., .2, relaxed=True) == "short_detour"
    for value in (None, float("inf"), float("nan")):
        assert detour_failure(value, 2.9, 3., .2, relaxed=True) == "disconnected"


def test_policy_is_opt_in_for_ordinary_camera_runs(monkeypatch):
    monkeypatch.delenv("ISPATIALGYM_CAMPAIGN_CONFIG", raising=False)
    assert active() is False
    assert POLICY["bucket_ranges_enforced"] is False
    assert POLICY["bucket_quotas_enforced"] is False
