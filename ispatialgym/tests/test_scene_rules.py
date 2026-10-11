"""The environment rules per scene on the merge of lab-vec-eval into v3.0-reconciled: the attach rule at a gripper release
(``predicates.attach_on_release`` and the ``AttachedTo`` overrides over the per-scene attachment registry) and the
assisted-grasp joint rule the lab evaluator re-applies through ``execute._patch_ag_joint_type``."""

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

    def fake_attach(child, parent, equivalences=None, in_place=True, released=False, trigger="per_step"):
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


def test_attach_overrides_read_the_per_scene_registry(monkeypatch):
    """v3.0's ``AttachedTo`` overrides (only the attach rule joins a registered pair) find a pair as ``attachment_context``
    does: registered for the object's own scene, else for a single env (scene None)."""
    import omnigibson.object_states as OS

    calls = []

    class FakeAT:
        def __init__(self, obj, parent=None):
            self.obj, self.parent = obj, parent

        def _update(self):
            calls.append(("stock_update", self.obj.scene.idx))

        def _attach(self, other, child_link, parent_link, joint_type=None, can_joint_break=True):
            calls.append(("stock_attach", self.obj.scene.idx))

        def _detach(self):
            calls.append(("stock_detach", self.obj.scene.idx))
            self.parent = None

    monkeypatch.setattr(OS, "AttachedTo", FakeAT)
    monkeypatch.setattr(PRED, "_OVERRIDES", {})
    for name in ("_FILTERED", "_PENDING_FILTER", "_PENDING_UNFILTER"):
        monkeypatch.setattr(PRED, name, set())
    monkeypatch.setattr(PRED, "attach_in_place", lambda c, p, cl, pl, can_joint_break=True: calls.append(("in_place", c.scene.idx)) or True)
    scenes = [_scene(("cap_0", "bottle_0"), idx=i) for i in (0, 1)]
    for s in scenes:
        s["cap_0"].links, s["bottle_0"].links = {"M": object()}, {"F": object()}

    def exercise():
        calls.clear()
        for s in scenes:
            cap, bottle = s["cap_0"], s["bottle_0"]
            FakeAT(cap)._update()
            FakeAT(bottle)._update()
            FakeAT(cap)._attach(bottle, cap.links["M"], bottle.links["F"])
            FakeAT(cap, parent=bottle)._detach()
        return list(calls)

    try:
        PRED.set_attachment_context([{"child": "cap_0", "parent": "bottle_0"}], scene_idx=1)
        assert PRED._OVERRIDES
        assert exercise() == [("stock_update", 0), ("stock_update", 0), ("stock_attach", 0), ("stock_detach", 0), ("in_place", 1), ("stock_detach", 1)]
        PRED.set_attachment_context([], scene_idx=1)
        PRED.set_attachment_context([{"child": "cap_0", "parent": "bottle_0"}])
        assert exercise() == [("in_place", 0), ("stock_detach", 0), ("in_place", 1), ("stock_detach", 1)]
    finally:
        PRED.set_attachment_context([], scene_idx=1)
        PRED.set_attachment_context([])


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


def test_vec_task_gives_every_scenes_robot_both_rules(monkeypatch):
    """``VecISpatialGymTask.initialize_activity`` re-implements the parent's: per scene, the robots of every scene get the
    release hook (once) and the assisted-grasp joint rule, as ``ISpatialGymTask.initialize_activity`` gives a single env."""
    from ispatialgym.tasks import vec_task as VT
    from ispatialgym.tasks.ispatialgym_task import AG_MOVING_LINK_JOINT_TYPE, apply_environment_rules

    robots = [[SimpleNamespace()], [SimpleNamespace(release_callbacks=[print])], [SimpleNamespace()]]
    env = SimpleNamespace(robots=robots, scenes=[SimpleNamespace(idx=i) for i in range(3)])
    task = SimpleNamespace(_object_bindings_source=lambda env: {}, _scopes=[{}, {}, {}], compiled_task=None,
                           get_agent=lambda env, i: env.robots[i][0], _tro=None)
    monkeypatch.setattr(VT, "physical_instances", lambda compiled_task: [])
    for i in range(3):
        assert VT.VecISpatialGymTask.initialize_activity(task, env, i) == (True, None)
    for (robot,) in robots:
        assert robot.release_callbacks.count(PRED.attach_on_release) == 1
        assert robot.ag_moving_link_joint_type == AG_MOVING_LINK_JOINT_TYPE
    assert robots[1][0].release_callbacks == [print, PRED.attach_on_release]
    assert apply_environment_rules(robots) == 3 and robots[0][0].release_callbacks == [PRED.attach_on_release]
