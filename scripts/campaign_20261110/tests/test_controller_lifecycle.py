from copy import deepcopy

import pytest

from scripts.campaign_20261110 import controller_lifecycle as lifecycle
from scripts.campaign_20261110.contracts import ContractError, write_json_atomic


def fixture(tmp_path):
    cfg = {"root": str(tmp_path), "campaign_id": "campaign", "stop_at": 10000}
    policy = {"schema": lifecycle.SCHEMA, "version": 1, "campaign_id": "campaign",
              "mode": "owner_controlled", "target_deadline_at": 10000,
              "budget_and_quota_limits_unchanged": True,
              "owner_instruction": {"attachment_sha256": lifecycle.REQUEST_SHA256, "quote": lifecycle.OWNER_QUOTE}}
    path = tmp_path / "manifests/controller_operating_policy.json"
    return cfg, policy, path


def test_explicit_owner_policy_retains_bounded_work_after_reporting_deadline(tmp_path):
    cfg, policy, path = fixture(tmp_path)
    assert lifecycle.continue_running(cfg, now=9000)
    assert not lifecycle.continue_running(cfg, now=10001)
    assert lifecycle.bounded_group_count(cfg, 6, 900, now=5000) == 1
    write_json_atomic(path, policy)
    assert lifecycle.continue_running(cfg, now=10001)
    assert lifecycle.bounded_group_count(cfg, 3, 900, now=10001) == 3
    assert lifecycle.bounded_group_count(cfg, 0, 900, now=10001) == 0
    assert lifecycle.sleep_seconds(cfg, now=10001) == 60
    state = {"config_sha256": "original", "seeds": {"seed": {"queues": ["immutable"]}}}
    lifecycle.record_policy(cfg, state)
    original = deepcopy(state)
    lifecycle.record_policy(cfg, state)
    assert state == original and state["config_sha256"] == "original"
    path.unlink()
    with pytest.raises(ContractError, match="disappeared"):
        lifecycle.record_policy(cfg, state)


@pytest.mark.parametrize("field,value", [("campaign_id", "other"), ("target_deadline_at", 20000),
                                          ("mode", "unbounded_budget"), ("budget_and_quota_limits_unchanged", False),
                                          ("owner_instruction", {"quote": "assumed permission"})])
def test_unrelated_or_weaker_authorization_cannot_remove_cutoff(tmp_path, field, value):
    cfg, policy, path = fixture(tmp_path)
    policy[field] = value
    write_json_atomic(path, policy)
    with pytest.raises(ContractError):
        lifecycle.continue_running(cfg, now=10001)
