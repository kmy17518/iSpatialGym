from copy import deepcopy
from types import SimpleNamespace
import pytest

from ispatialgym.sampling import campaign_class_recipes as R
from ispatialgym.sampling.campaign_tabletop_starts import prioritize_written_stances


def inputs(class_id):
    p = R.PROFILES[class_id]
    config = {"split": "eval_standard", "template": {"class_id": class_id, "task": p["task"]},
              "validation_key": {k: p["applicability"][k] for k in
                                 ("template_sha256", "scale_sha256", "data_sha256", "physics_sha256")}}
    hint = {"base_pose2d": [8., 0., 0.], "focus": [8., 0., 1.], "translate": .2, "tilt": -.3}
    plan = SimpleNamespace(contract_template_sha256=p["template_sha256"], sha256="a" * 64,
                           anchor_manipulation_stance=hint)
    candidates = [SimpleNamespace(base_pose2d=[float(i), 0., 0.], focus=[float(i), 0., 1.],
                                  translate=.2, tilt=-.3) for i in range(10)]
    return p["route_id"], config, plan, candidates


@pytest.mark.parametrize("class_id", R.PROFILES)
def test_exact_historical_order_and_complete_trace(class_id):
    route, config, plan, candidates = inputs(class_id)
    order = list(range(10)); before = list(order); trace = []
    fn = R.attachment03_order_stances if class_id.startswith("configuration") else R.under_small03_order_stances
    result = fn(route, config, plan, order, candidates, trace=trace)
    assert result == prioritize_written_stances(order, candidates, [plan.anchor_manipulation_stance])
    assert result[:6] == [0, 1, 2, 3, 8, 7]
    assert sorted(result) == order == before
    assert trace[0]["action"] == "source_stance_ranked"
    assert trace[0]["route_id"] == route and trace[0]["phase"] == "sampling"
    assert trace[0]["native_checks_unchanged"] and len(trace[0]["hint_sha256"]) == 64


def test_current_written_under_stance_precedes_historical_hint():
    route, config, plan, candidates = inputs("inter_object_relation_satisfaction-under_small_scale_03")
    written = [dict(plan.anchor_manipulation_stance, base_pose2d=[5., 0., 0.], focus=[5., 0., 1.])]
    trace = []
    out = R.under_small03_order_stances(route, config, plan, list(range(10)), candidates, written, trace=trace)
    assert out[:6] == [0, 1, 2, 3, 5, 4]
    assert trace[0]["hint_source"] == "current_group_written_member"


@pytest.mark.parametrize("field", ["template_sha256", "scale_sha256", "data_sha256", "physics_sha256"])
def test_mismatched_variant_or_physics_fails_closed(field):
    route, config, plan, _ = inputs("configuration_matching-attachment_03")
    config["validation_key"][field] = "b" * 64
    with pytest.raises(ValueError): R.profile_for_config(config)
    with pytest.raises(ValueError): R.validate_recipe_inputs(route, config, plan, phase="oracle", trace=[])


def test_unselected_class_keeps_existing_dispatch_and_profiles_are_copied():
    assert R.profile_for_config({"template": {"class_id": "configuration_matching-attachment_04"}}) is None
    _, config, _, _ = inputs("configuration_matching-attachment_03")
    profile = R.profile_for_config(config); profile["applicability"]["splits"] = ["corrupt"]
    assert R.profile_for_config(config)["applicability"]["splits"] == ["eval_standard", "train"]


@pytest.mark.parametrize("phase", ["oracle", "replay"])
def test_oracle_and_replay_validate_without_planning_or_state_changes(phase):
    route, config, plan, _ = inputs("configuration_matching-attachment_03")
    original = deepcopy(config); trace = []
    event = R.validate_recipe_inputs(route, config, plan, phase=phase, trace=trace)
    assert event["action"] == ("oracle_native_unchanged" if phase == "oracle" else "replay_recorded_route_verified") and config == original
    assert event is trace[0]


def test_missing_hint_cannot_silently_fallback():
    route, config, plan, candidates = inputs("configuration_matching-attachment_03")
    plan.anchor_manipulation_stance = None
    with pytest.raises(ValueError): R.attachment03_order_stances(route, config, plan, range(10), candidates, trace=[])


def test_trace_sink_and_plan_contract_required():
    route, config, plan, _ = inputs("configuration_matching-attachment_03")
    with pytest.raises(ValueError): R.validate_recipe_inputs(route, config, plan, phase="sampling", trace=None)
    plan.contract_template_sha256 = "f" * 64
    with pytest.raises(ValueError): R.validate_recipe_inputs(route, config, plan, phase="replay", trace=[])


def test_raw_recorded_plan_and_runtime_revision_are_supported():
    route, config, plan, _ = inputs("configuration_matching-attachment_03")
    config["validation_key"]["code_sha256"] = "f" * 64
    raw = dict(contract_template_sha256=plan.contract_template_sha256,
               anchor_manipulation_stance=plan.anchor_manipulation_stance)
    trace = []
    R.validate_recipe_inputs(route, config, raw, phase="replay", trace=trace)
    assert len(trace[0]["plan_sha256"]) == 64


def test_canonical_training_inherits_exact_physical_profile():
    route, config, plan, _ = inputs("configuration_matching-attachment_03")
    config["split"] = "train"; config["template"]["split"] = "eval_standard"
    assert R.profile_for_config(config)["route_id"] == route
    assert R.validate_recipe_inputs(route, config, plan, phase="sampling", trace=[])["route_id"] == route
