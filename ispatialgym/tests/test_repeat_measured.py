"""train-fast REPEAT_MEASURED (owner 2026-11-01): the train-repeat rule compares what the task measures (TRO pose_targets /
articulation_targets: each measured object's start state and goal target, at the task's own tolerances), not the robot base alone."""

import copy
import math

import pytest

from ispatialgym.sampling import instance_separation as SEP


def _yaw_q(yaw):
    return [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]


def tro(obj_xy=(1.0, 2.0), obj_yaw=0.0, goal_xy=(1.2, 2.0), goal_yaw=0.0, base=(0.0, 0.0), joint=None, distractor_xy=(5.0, 5.0)):
    t = {
        "robot_poses": {"robot": [{"position": [base[0], base[1], 0.0], "orientation": _yaw_q(0.0)}]},
        "cup.n.01_1": {"root_link": {"pos": [obj_xy[0], obj_xy[1], 0.8], "ori": _yaw_q(obj_yaw)}},
        "plate.n.04_1": {"root_link": {"pos": [distractor_xy[0], distractor_xy[1], 0.8], "ori": _yaw_q(0.0)}},
        "ispatialgym": {
            "activity": "inter_object_relation_satisfaction", "task_variant": "nextto-cup", "scene": {"model": "house", "task_rooms": ["dining"]},
            "pose_targets": {"cup.n.01_1": {"frame": "world", "position": [goal_xy[0], goal_xy[1], 0.8], "orientation": _yaw_q(goal_yaw),
                                            "position_metric": "full", "orientation_metric": "full", "position_tolerance": 0.03,
                                            "orientation_tolerance": 0.2, "orientation_equivalences": [_yaw_q(math.pi)]}},
            "articulation_targets": {},
        },
    }
    if joint is not None:
        t["ispatialgym"]["articulation_targets"] = {"cup.n.01_1": {"joints": {"j_lid": {"position": joint, "tolerance": 0.15}}}}
        t["cup.n.01_1"]["joint_pos"] = [0.0]
    return t


def rec(**kw):
    # the sidecar's goal base, as the tabletop instances carry it (the 2026-10-11 rule compares it)
    return SEP.record_from_dicts(tro(**kw), {"goal": {"base_pose2d": [0.3, 0.0, 0.0]}}, {}, {})


def test_measured_entries_exclude_distractors():
    r = rec(joint=1.5)
    assert [m.name for m in r.measured_goal] == ["goal:cup.n.01_1", "goal:cup.n.01_1:j_lid"]
    assert [m.name for m in r.measured_start] == ["start:cup.n.01_1", "start:cup.n.01_1:j_lid"]
    assert not any("plate" in m.name for m in r.measured_start + r.measured_goal)


def test_within_tolerance_is_a_repeat_and_beyond_is_not():
    e = rec()
    assert SEP.find_repeat(rec(obj_xy=(1.02, 2.0), goal_xy=(1.21, 2.0)), [e]) is not None
    assert SEP.find_repeat(rec(goal_xy=(1.25, 2.0)), [e]) is None  # goal 5 cm away > 3 cm
    assert SEP.find_repeat(rec(obj_xy=(1.1, 2.0)), [e]) is None  # start 10 cm away
    assert SEP.find_repeat(rec(goal_yaw=0.3), [e]) is None  # 0.3 rad > 0.2


def test_symmetry_equivalence_counts_as_the_same_orientation():
    assert SEP.find_repeat(rec(goal_yaw=math.pi), [rec()]) is not None


def test_distractor_and_robot_base_alone_do_not_separate_or_join():
    e = rec()
    assert SEP.find_repeat(rec(distractor_xy=(9.0, 9.0)), [e]) is not None  # a distractor elsewhere: still the same instance
    assert SEP.find_repeat(rec(base=(1.0, 0.0)), [e]) is None  # the robot starts 1 m away: another instance


def test_the_old_rule_rejected_what_the_measured_rule_keeps(monkeypatch):
    e, r = rec(goal_xy=(1.2, 2.0)), rec(goal_xy=(1.6, 2.4), obj_xy=(0.6, 2.3))  # objects elsewhere, the robot at the same edge
    assert SEP.find_repeat(r, [e]) is None
    monkeypatch.setenv("ISPATIALGYM_FAST_REPEAT_MEASURED", "0")
    assert SEP.find_repeat(r, [e]) is not None  # 2026-10-11 rule: start base + goal base (none here) -> a repeat


def test_joint_targets_compare_at_their_tolerance():
    e = rec(joint=1.5)
    assert SEP.find_repeat(rec(joint=1.6), [e]) is not None
    assert SEP.find_repeat(rec(joint=1.8), [e]) is None


def test_depth_metric_compares_depth_only():
    a = SEP.Measured("goal:x", "pose", "agent::cam", [0.0, 0.0, -0.40], None, "depth", 0.02, None)
    assert a.within(SEP.Measured("goal:x", "pose", "agent::cam", [0.3, 0.1, -0.41], None, "depth", 0.02, None)) is not None
    assert a.within(SEP.Measured("goal:x", "pose", "agent::cam", [0.0, 0.0, -0.45], None, "depth", 0.02, None)) is None
    assert a.within(SEP.Measured("goal:x", "pose", "world", [0.0, 0.0, -0.40], None, "depth", 0.02, None)) is None  # another frame


def test_index_round_trip_keeps_the_measured_entries():
    r = rec(joint=1.5)
    back = SEP.InstanceRecord.from_json(copy.deepcopy(r.to_json()))
    assert back.measured_goal == r.measured_goal and back.measured_start == r.measured_start
    assert SEP.find_repeat(back, [r]) is not None


def test_an_old_index_record_falls_back_to_the_old_rule():
    old = rec()
    old.measured_start, old.measured_goal = [], []
    assert SEP.find_repeat(rec(goal_xy=(1.6, 2.4), obj_xy=(0.6, 2.3)), [old]) is not None  # legacy comparison (base only)


def test_early_record_without_the_object_state_is_incomplete():
    t = tro()
    del t["cup.n.01_1"]
    r = SEP.record_from_dicts(t, {}, {}, {})
    assert SEP.record_complete(r) == "missing start:cup.n.01_1"
    assert SEP.record_complete(rec()) is None


def _group_tro(starts, slots, mode="exchangeable", base=(0.0, 0.0)):
    t = {"robot_poses": {"robot": [{"position": [base[0], base[1], 0.0], "orientation": _yaw_q(0.0)}]},
         "ispatialgym": {"activity": "configuration_matching", "task_variant": "block_design", "scene": {"model": "house"}, "pose_targets": {},
                         "groups": {"g.n.01_1": {"members": [f"die.n.01_{i + 1}" for i in range(len(starts))], "assignment": {"mode": mode},
                                                 "orientation_equivalences": [], "pose_tolerances": {"position": 0.01, "orientation": 0.15},
                                                 "target_slots": {f"slot_{k}": {"position": [x, y, 0.8], "orientation": _yaw_q(0.0)}
                                                                  for k, (x, y) in enumerate(slots)}}}}}
    for i, (x, y) in enumerate(starts):
        t[f"die.n.01_{i + 1}"] = {"root_link": {"pos": [x, y, 0.8], "ori": _yaw_q(0.0)}}
    return SEP.record_from_dicts(t, {"goal": {"base_pose2d": [0.3, 0.0, 0.0]}}, {}, {})


def test_exchangeable_group_compares_members_and_slots_as_sets():
    e = _group_tro([(1.0, 1.0), (1.2, 1.0)], [(2.0, 2.0), (2.1, 2.0)])
    swapped = _group_tro([(1.2, 1.0), (1.0, 1.0)], [(2.1, 2.0), (2.0, 2.0)])  # the same layout, members and slots relabelled
    assert SEP.find_repeat(swapped, [e]) is not None
    assert SEP.find_repeat(_group_tro([(1.0, 1.0), (1.2, 1.0)], [(2.0, 2.0), (2.1, 2.05)]), [e]) is None  # one slot 5 cm off > 1 cm
    assert SEP.record_complete(e) is None


def test_fixed_group_compares_slots_by_name():
    e = _group_tro([(1.0, 1.0), (1.2, 1.0)], [(2.0, 2.0), (2.1, 2.0)], mode="fixed")
    swapped = _group_tro([(1.0, 1.0), (1.2, 1.0)], [(2.1, 2.0), (2.0, 2.0)], mode="fixed")  # male / female slots exchanged: another goal
    assert SEP.find_repeat(swapped, [e]) is None
    assert SEP.find_repeat(_group_tro([(1.0, 1.0), (1.2, 1.0)], [(2.0, 2.0), (2.1, 2.0)], mode="fixed"), [e]) is not None
