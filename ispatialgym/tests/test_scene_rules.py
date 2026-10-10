"""The environment rules per scene on the merge of lab-vec-eval into v3.0-reconciled: the attach rule at a gripper release
(``predicates.attach_on_release`` over the per-scene attachment registry) and the assisted-grasp joint rule the lab
evaluator re-applies through ``execute._patch_ag_joint_type``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ispatialgym.sampling import predicates as PRED


def _scene(names, idx=None):
    reg = {n: SimpleNamespace(name=n) for n in names}
    scene = SimpleNamespace(object_registry=lambda key, name: reg.get(name))
    if idx is not None:
        scene.idx = idx
    for o in reg.values():
        o.scene = scene
    return reg


@pytest.fixture
def attach_calls(monkeypatch):
    """``attach_if_aligned`` replaced by a recorder of (child, parent, the registered pair's child_link)."""
    calls = []

    def fake_attach(child, parent, equivalences=None, in_place=True, released=False):
        calls.append((child, parent, (PRED.attachment_context(child, parent) or {}).get("child_link"), released))
        return {"ok": True, "attached": True, "joint": "in_place", "fix": "T"}

    monkeypatch.setattr(PRED, "attach_if_aligned", fake_attach)
    yield calls
    PRED.set_attachment_context([])
    for idx in (0, 1):
        PRED.set_attachment_context([], scene_idx=idx)


def test_release_joins_each_clone_part_to_its_own_scenes_partner(attach_calls):
    s0, s1 = _scene(("cap_0", "bottle_0"), idx=0), _scene(("cap_0", "bottle_0"), idx=1)
    PRED.set_attachment_context([{"child": "cap_0", "parent": "bottle_0", "child_link": "s0_link"}], scene_idx=0)
    PRED.set_attachment_context([{"child": "cap_0", "parent": "bottle_0", "child_link": "s1_link"}], scene_idx=1)
    assert PRED.attach_on_release(None, "right", s1["cap_0"])["parent"] == "bottle_0"
    assert PRED.attach_on_release(None, "left", s0["cap_0"])["parent"] == "bottle_0"
    assert attach_calls == [(s1["cap_0"], s1["bottle_0"], "s1_link", True), (s0["cap_0"], s0["bottle_0"], "s0_link", True)]
    assert PRED.attach_on_release(None, "right", s1["bottle_0"]) is None and len(attach_calls) == 2


def test_release_uses_only_the_partners_its_scene_registered(attach_calls):
    names = ("cap_0", "bottle_0", "bottle_1")
    s0, s1 = _scene(names, idx=0), _scene(names, idx=1)
    PRED.set_attachment_context([{"child": "cap_0", "parent": "bottle_0"}], scene_idx=0)
    PRED.set_attachment_context([{"child": "cap_0", "parent": "bottle_1"}], scene_idx=1)
    assert PRED.attachment_partners(s1["cap_0"]) == ["bottle_1"] and PRED.attachment_partners(s0["cap_0"]) == ["bottle_0"]
    PRED.attach_on_release(None, "right", s1["cap_0"])
    assert [(c, p) for c, p, _, _ in attach_calls] == [(s1["cap_0"], s1["bottle_1"])]


def test_single_env_registration_keeps_its_order_and_fallback(attach_calls):
    # a single-env task registers under scene None: the pairs in registration order, the first partner present wins
    PRED.set_attachment_context([{"child": "lid_0", "parent": "teapot_0"}, {"child": "lid_0", "parent": "kettle_0"}])
    only_kettle, both = _scene(("lid_0", "kettle_0")), _scene(("lid_0", "teapot_0", "kettle_0"), idx=0)
    assert PRED.attachment_partners(only_kettle["lid_0"]) == ["teapot_0", "kettle_0"]
    assert PRED.attach_on_release(None, "right", only_kettle["lid_0"])["parent"] == "kettle_0"
    assert PRED.attach_on_release(None, "right", both["lid_0"])["parent"] == "teapot_0"
    # a clone without its own registration falls back to it, as attachment_context does; its own pairs come first
    clone = _scene(("lid_0", "teapot_0", "kettle_0", "pot_0"), idx=1)
    PRED.set_attachment_context([{"child": "lid_0", "parent": "pot_0"}], scene_idx=1)
    assert PRED.attachment_partners(clone["lid_0"]) == ["pot_0", "teapot_0", "kettle_0"]
    PRED.set_attachment_context([], scene_idx=1)
    assert PRED.attachment_partners(clone["lid_0"]) == ["teapot_0", "kettle_0"]
    assert PRED.attach_on_release(None, "right", SimpleNamespace(name="lid_0")) is None  # no scene: nothing to join
    assert [p.name for _, p, _, _ in attach_calls] == ["kettle_0", "teapot_0"]


def test_ag_joint_shim_only_reapplies_the_tasks_rule():
    from ispatialgym.oracle import execute as EX
    from ispatialgym.tasks.ispatialgym_task import AG_MOVING_LINK_JOINT_TYPE

    robot = SimpleNamespace()
    EX._patch_ag_joint_type(robot, None)
    assert vars(robot) == {}
    EX._patch_ag_joint_type(robot, "FixedJoint")
    assert vars(robot) == {"ag_moving_link_joint_type": AG_MOVING_LINK_JOINT_TYPE} and AG_MOVING_LINK_JOINT_TYPE == "FixedJoint"
    with pytest.raises(ValueError):
        EX._patch_ag_joint_type(robot, "SphericalJoint")
