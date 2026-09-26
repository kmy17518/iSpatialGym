"""oracle.protection (owner 2026-09-26, overnight plan item 2): protected objects during transits."""

from __future__ import annotations

from types import SimpleNamespace

import torch as th

from ispatialgym.oracle import protection as PROT


def _obj(name, category, model, box=None, pos=(0.0, 0.0, 0.0)):
    o = SimpleNamespace(name=name, category=category, model=model, prim_path=f"/World/scene_0/{name}")
    c, half = (th.tensor(pos, dtype=th.float32), th.tensor(box or (0.05, 0.05, 0.05), dtype=th.float32))
    o.get_base_aligned_bbox = lambda visual=False: (c, th.tensor([0.0, 0.0, 0.0, 1.0]), 2 * half, th.zeros(3))
    o.get_position_orientation = lambda: (c.clone(), th.tensor([0.0, 0.0, 0.0, 1.0]))
    return o


def test_positive_goal_instances_skip_negated_and_quantified_atoms():
    """Disembedding: the distractors appear only in negated atoms and are not protected; nested and / or are followed."""
    goal = ["(ontop ?bar_soap.n.01_1 ?tray.n.01_2)", "(not (ontop ?bar_soap.n.01_2 ?tray.n.01_2))", "(not (grasped ?agent.n.01_1 ?bar_soap.n.01_1))",
            "(pose_settled ?bar_soap.n.01_1)", ["and", ["under", "?saucer.n.02_1", "?teacup.n.02_1"]], "(forall (?x - toy.n.01) (ontop ?x ?box.n.01_1))"]
    assert PROT.positive_goal_instances(goal) == ["bar_soap.n.01_1", "tray.n.01_2", "saucer.n.02_1", "teacup.n.02_1"]


def test_task_objects_are_protected_furniture_is_not():
    """Teacup -> the saucer and the cup; the cap's goal names the table (ontop bottle table): furniture, not protected; the
    customs override excludes / includes by bddl name."""
    physical = {"teacup.n.02_1": _obj("target_teacup_wopjex_0", "teacup", "wopjex"), "saucer.n.02_1": _obj("partner_saucer_mgbeah_0", "saucer", "mgbeah"),
                "breakfast_table.n.01_1": _obj("breakfast_table_rhjoby_0", "breakfast_table", "rhjoby"), "door.n.01_1": _obj("door_crtrpa_0", "door", "crtrpa"),
                "tray.n.01_1": _obj("board_tray_isgboard_0", "tray", "isgboard")}
    goal = ["(under ?saucer.n.02_1 ?teacup.n.02_1)", "(ontop ?saucer.n.02_1 ?breakfast_table.n.01_1)", "(not (open ?door.n.01_1))", "(layout_reached ?door.n.01_1)"]
    got = PROT.goal_protected(goal, physical)
    assert [b for b, _ in got] == ["saucer.n.02_1", "teacup.n.02_1"]
    assert PROT.is_task_object(physical["tray.n.01_1"]) and not PROT.is_task_object(physical["door.n.01_1"])
    got = PROT.goal_protected(goal, physical, {"exclude": ["teacup.n.02_1"], "include": ["tray.n.01_1"]})
    assert [b for b, _ in got] == ["saucer.n.02_1", "tray.n.01_1"]


def test_sphere_box_clearance_and_the_endpoint_relative_threshold():
    """A 15 cm x 1.5 cm saucer box: a transit whose middle skims 5 mm above it fires (threshold 2 cm); one whose endpoints are
    already closer (a pre-contact 12 mm off) may approach as close as its endpoints are, no closer."""
    saucer = _obj("partner_saucer_mgbeah_0", "saucer", "mgbeah", box=(0.075, 0.075, 0.0075), pos=(0.0, 0.0, 0.8235))
    box = PROT.object_box(saucer)
    # one sphere of radius 1 cm travelling over the saucer at heights 0.95 -> 0.84 -> 0.95 (surface at 0.831)
    zs = [0.95, 0.90, 0.846, 0.90, 0.95]
    sp = th.tensor([[[0.0, 0.0, z, 0.01]] for z in zs])
    val, idx = PROT.sphere_box_clearance(sp, box)
    assert abs(float(val[2]) - (0.846 - 0.831 - 0.01)) < 1e-4 and int(idx[2]) == 0
    v = PROT.transit_violations(val)
    assert v is not None and v["index"] == 2 and v["clearance_m"] < 0.02 and v["threshold_m"] == 0.02
    # endpoints 12 mm off: the threshold drops to 10 mm; a middle at 11 mm clears, at 5 mm fires
    base = th.tensor([0.012, 0.03, 0.011, 0.03, 0.012])
    assert PROT.transit_violations(base) is None
    base[2] = 0.005
    v = PROT.transit_violations(base)
    assert v is not None and abs(v["threshold_m"] - 0.010) < 1e-6
    # disabled spheres (r <= 0) never count
    sp2 = th.tensor([[[0.0, 0.0, 0.832, 0.0], [0.0, 0.0, 1.2, 0.01]]])
    val2, _ = PROT.sphere_box_clearance(sp2, box)
    assert float(val2[0]) > 0.3


def test_check_trajectory_uses_only_the_moving_arm_and_its_carried_object():
    """The static arm resting on the protected object (a brace) sets no threshold; the moving arm's fingers and the object
    attached to its hand do."""
    part = _obj("partner_box_of_tissues_ntbrtz_0", "box_of_tissues", "ntbrtz", box=(0.05, 0.05, 0.05), pos=(0.0, 0.0, 0.0))
    links = ["left_gripper_finger_link1", "right_gripper_finger_link1", "attached_object_right_eef_link", "torso_link4"]
    T = 5
    sp = th.zeros(T, 4, 4)
    sp[:, 0] = th.tensor([0.06, 0.0, 0.0, 0.01])  # the left (static) fingers touch the box all along
    sp[:, 3] = th.tensor([0.0, 0.0, 0.2, 0.05])   # the torso
    for t, z in enumerate([0.30, 0.20, 0.15, 0.20, 0.30]):
        sp[t, 1] = th.tensor([0.0, 0.0, z, 0.01])
        sp[t, 2] = th.tensor([0.2, 0.0, z, 0.02])
    assert PROT.check_trajectory(sp, links, ["right"], [("box", part)]) == []
    sp[2, 1] = th.tensor([0.0, 0.0, 0.065, 0.01])  # the right fingers dip to 5 mm above the box mid-transit
    viol = PROT.check_trajectory(sp, links, ["right"], [("box", part)])
    assert len(viol) == 1 and viol[0]["link"] == "right_gripper_finger_link1" and viol[0]["object"] == "partner_box_of_tissues_ntbrtz_0"
    assert PROT.moving_sphere_mask(links, ["right"]).tolist() == [False, True, True, False]


def test_qa_transit_span_flags_new_contacts_and_movement_only():
    """The run-time check: a contact that exists when the span opens (a brace) is exempt, a new contact of the acting fingers -
    exempt from the ordinary collision monitor during a manipulation action - is a verdict, and so is a protected object
    moving more than 1 cm."""
    from ispatialgym.oracle.qa import Monitors, QAConfig

    robot = SimpleNamespace(prim_path="/World/scene_0/robot_r1")
    saucer = _obj("partner_saucer_mgbeah_0", "saucer", "mgbeah")
    registry = {saucer.prim_path: saucer}
    mon = Monitors.__new__(Monitors)
    Monitors.__init__.__globals__  # noqa: B018 - keep the import side effects
    mon.cfg, mon.robot = QAConfig(), robot
    mon.scene = SimpleNamespace(idx=0, object_registry=lambda key, path, default=None: registry.get(path, default))
    mon.stats = SimpleNamespace(record=lambda v: None, event=lambda e: None)
    for k, v in {"collision_events": [], "_prim_cache": {}, "_contact_failed": False, "action": SimpleNamespace(name="grasp"), "action_index": 1, "attempt": 0,
                 "manipulation": True, "intended": set(), "exempt_links": {"right_gripper_finger_link1"}, "held": [], "arms": ("right",), "action_steps": 0,
                 "episode_steps": 0, "epilogue": False, "forbidden": set(), "watch": [], "watch_hits": [], "allowed": [], "transit": None, "protect_events": [],
                 "_pairs_step": None, "_pairs": set()}.items():
        setattr(mon, k, v)
    from collections import Counter

    mon._contact_streak, mon.collisions_by_object = Counter(), Counter()
    pairs = {"now": {(f"{robot.prim_path}/left_gripper_finger_link1", f"{saucer.prim_path}/base_link")}}
    mon._contact_api = SimpleNamespace(get_contact_pairs=lambda idx, query, other, current_only=False: set(pairs["now"]))
    mon.begin_transit([("saucer.n.02_1", saucer)], {"task": "teacup", "action": "grasp", "index": 1, "leg": "pre_grasp"})
    assert mon.transit["exempt"] == {("left_gripper_finger_link1", "partner_saucer_mgbeah_0")}
    mon.episode_steps = 1
    assert mon._check_protected() is None  # only the start contact
    pairs["now"] = pairs["now"] | {(f"{robot.prim_path}/right_gripper_finger_link1", f"{saucer.prim_path}/base_link")}
    mon.episode_steps = 2
    v = mon._check_protected()
    assert v is not None and v.category == "protected_contact" and v.details["link"] == "right_gripper_finger_link1"
    # movement: 12 mm
    pairs["now"] = set()
    mon.begin_transit([("saucer.n.02_1", saucer)], {"leg": "pre_grasp"})
    moved = th.tensor([0.012, 0.0, 0.0])
    saucer.get_position_orientation = lambda: (moved.clone(), th.tensor([0.0, 0.0, 0.0, 1.0]))
    mon.episode_steps = 3
    v = mon._check_protected()
    assert v is not None and v.category == "protected_moved" and abs(v.details["moved_m"] - 0.012) < 1e-6
    mon.end_transit()
    assert mon._check_protected() is None
