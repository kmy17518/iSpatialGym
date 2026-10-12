"""Simulator-free tests of the eval / train split (owner 2026-10-11): per-alias split seeds, the held-object scene / room draw,
the eval separation check, the train-repeat rule and their CLI plumbing (``sample_isg``, ``multiply_isg``, ``SamplerV2``)."""

from __future__ import annotations

import json
import math
import os
import random
from types import SimpleNamespace

import numpy as np
import pytest

from ispatialgym import ASSETS_ROOT
from ispatialgym.sampling import instance_separation as SEP
from ispatialgym.sampling import split_seeds as SS
from ispatialgym.sampling.v2.problem import GroundedProblem, GroundingError, TemplateChoice, iter_template_choices


def _held_entry(scene="house_single_floor", room="living_room_0"):
    return {"iddl": "object_scaling/object_scaling.iddl", "definition_id": 0,
            "roles": {"target": {"category": "apple", "sampleable": True, "representative": "a"}, "floor": {"sampleable": False}},
            "scenes": {scene: {"rooms": [room], "bindings": {"floor": {"room": room}}, "whitelist": {"target": {"apple": ["a"]}},
                               "blacklist": {"target": {"apple": {"b": "heldout"}}}, "_note": "kept"}},
            "generation": {"templates_max": 20, "instances_per_template": 1}, "constraints": []}


def _camera_entry():
    e = _held_entry()
    e["iddl"] = "camera_relocalization/standard.iddl"
    e["roles"] = {"floor": {"sampleable": False}}
    e["scenes"]["house_single_floor"]["whitelist"] = {}
    return e


# ----------------------------------------------------------------------------------------------------------------
# seeds
# ----------------------------------------------------------------------------------------------------------------


def test_split_seeds_are_stable_disjoint_across_splits_and_unique_per_alias():
    # pinned: a changed derivation would silently change every eval set
    assert SS.split_seed("eval", "camera_relocalization-standard") == 684377364
    assert SS.split_seed("train", "object_scaling-apple") == 639875273
    for alias in ("object_scaling-apple", "mental_rotation-mug", "camera_relocalization-standard"):
        for k in range(5):
            e, t = SS.split_seed("eval", alias, k), SS.split_seed("train", alias, k)
            assert e % 2 == 0 and t % 2 == 1 and 0 <= e < 2 ** 30 and 0 <= t < 2 ** 30
        assert SS.split_seed("eval", alias, 0) != SS.split_seed("eval", alias, 1)
    with pytest.raises(ValueError):
        SS.split_seed("test", "x")
    with pytest.raises(ValueError):
        SS.split_seed("eval", "x", -1)
    # every alias of the shipped customs list x both splits x the checked re-sample indices: pairwise distinct
    with open(os.path.join(ASSETS_ROOT, "task_specs", "isg_customs.json")) as f:
        aliases = [a for a in json.load(f) if not a.startswith("_")]
    seen = SS.check_unique(aliases)
    assert len(seen) == len(set(aliases)) * 2 * SS.CHECKED_INDICES
    # derived attempt seeds fit in int64
    assert (max(seen) * 7919 + 7918) * 1000003 + 990 * 1009 + 1008 < 2 ** 63
    with pytest.raises(ValueError):
        SS.check_unique(["a"], indices=[0, 0])


def test_distinct_seeds_give_disjoint_attempt_seed_streams():
    """The samplers' attempt seed ``(seed * 7919 + t) * 1000003 + index * 1009 + attempt`` decodes back to (seed, t, index, attempt)
    inside the documented bounds, so two seeds never share an attempt seed."""
    rng = random.Random(0)
    for _ in range(2000):
        s, t, i, a = rng.randrange(2 ** 30), rng.randrange(7919), rng.randrange(991), rng.randrange(1009)
        f = (s * 7919 + t) * 1000003 + i * 1009 + a
        base, rest = divmod(f, 1000003)
        assert divmod(base, 7919) == (s, t) and divmod(rest, 1009) == (i, a)


# ----------------------------------------------------------------------------------------------------------------
# scene / room draw
# ----------------------------------------------------------------------------------------------------------------


def test_eligible_rooms_outdoor_only_exclusion_and_threshold():
    """owner 2026-10-11 ~13:50: every room but an outdoor one with at least 1 m^2 of robot-reachable floor (the samplers' 6-cell
    erosion); no room-type whitelist."""
    assert SS.NAV_ERODE_PX == 6 and SS.MIN_NAV_FREE_M2 == 1.0 and not hasattr(SS, "ROOM_TYPES")
    rooms = SS.eligible_rooms()
    assert len(rooms) == 45 and round(sum(SS.ROOM_FLOOR_M2[s][r][2] for s, r in rooms), 2) == 611.73
    assert {sc: len(SS.eligible_rooms(sc)) for sc in SS.SCENE_CANDIDATES} == {
        "house_single_floor": 15, "house_double_floor_lower": 5, "house_double_floor_upper": 5, "Rs_int": 1, "restaurant_diner": 5,
        "hotel_suite_large": 2, "office_cubicles_right": 12}
    assert SS.eligible_rooms("Rs_int") == [("Rs_int", "living_room_0")]
    for scene, room in rooms:
        assert not SS.is_outdoor(room) and SS.ROOM_FLOOR_M2[scene][room][2] >= SS.MIN_NAV_FREE_M2
    # out: the closets and Rs_int's small rooms (< 1 m^2), both gardens (outdoor, however large)
    out = {(sc, r) for sc, table in SS.ROOM_FLOOR_M2.items() for r in table} - set(rooms)
    assert out == ({("house_single_floor", f"closet_{i}") for i in range(5)} | {("Rs_int", r) for r in ("bathroom_0", "bedroom_0", "entryway_0", "kitchen_0")}
                   | {("house_single_floor", "garden_0"), ("house_double_floor_lower", "garden_0")})
    assert SS.ROOM_FLOOR_M2["house_single_floor"]["garden_0"][2] > 1000
    assert ("office_cubicles_right", "corridor_0") in rooms and ("house_double_floor_lower", "garage_0") in rooms and ("restaurant_diner", "bathroom_0") in rooms
    assert set(SS.eligible_rooms(min_nav_free_m2=0.5)) > set(rooms)
    assert SS.room_type("living_room_0") == "living_room" and SS.room_type("private_office_12") == "private_office" and SS.room_type("lobby") == "lobby"
    assert SS.scene_draw_applies("object_scaling") and SS.scene_draw_applies("mental_rotation")
    assert not SS.scene_draw_applies("camera_relocalization-standard") and not SS.scene_draw_applies("disembedding")
    assert SS.scene_draw_task("object_scaling-apple") == "object_scaling" and SS.scene_draw_task("mental_rotation") == "mental_rotation"
    assert SS.scene_draw_task("disembedding-bar_soap") is None and SS.scene_draw_task("object_scaling_x-apple") is None


def _held_aliases():
    with open(os.path.join(ASSETS_ROOT, "task_specs", "isg_customs.json")) as f:
        return [a for a in json.load(f) if SS.scene_draw_task(a)]


def test_eval_rooms_are_the_frozen_per_task_assignment():
    """owner 2026-10-11 ~13:50: per task 10 distinct rooms covering all 7 scenes, the 3 repeats in the scenes with more rooms; the
    tasks share no room but Rs_int's only one; deterministic; frozen (EVAL_ROOMS = the recomputed assignment); a re-sample keeps
    the room."""
    aliases = _held_aliases()
    assert sorted(aliases) == sorted(SS.EVAL_ROOMS)  # a new held-object alias needs its room added to the table
    got, orders = SS.eval_assignment(aliases, moves=SS.EVAL_ROOM_MOVES)
    assert got == SS.EVAL_ROOMS
    assert SS.eval_assignment(list(reversed(aliases)) + ["camera_relocalization-standard"], moves=SS.EVAL_ROOM_MOVES)[0] == got
    eligible = set(SS.eligible_rooms())
    base, _ = SS.eval_assignment(aliases)  # before any room move (EVAL_ROOM_MOVES may cost a task a scene)
    for table in (base, SS.EVAL_ROOMS):
        per_task = {}
        for task in SS.SCENE_DRAW_SUBTASKS:
            rooms = [table[a] for a in aliases if SS.scene_draw_task(a) == task]
            assert len(rooms) == 10 and len(set(rooms)) == 10 and set(rooms) <= eligible
            assert sorted(orders[task]) == sorted(eligible)
            per_task[task] = set(rooms)
        shared = per_task["object_scaling"] & per_task["mental_rotation"]
        assert shared == {("Rs_int", "living_room_0")} if table is base else shared <= {("Rs_int", "living_room_0")}
    for task in SS.SCENE_DRAW_SUBTASKS:
        rooms = [base[a] for a in aliases if SS.scene_draw_task(a) == task]
        counts = {sc: sum(s == sc for s, _ in rooms) for sc in SS.SCENE_CANDIDATES}
        assert min(counts.values()) == 1 and max(counts.values()) == 2
        repeats = {sc for sc, n in counts.items() if n == 2}
        assert len(repeats) == 3 and {"house_single_floor", "office_cubicles_right"} < repeats
        assert [len(SS.eligible_rooms(sc)) for sc in repeats - {"house_single_floor", "office_cubicles_right"}] == [5]
    assert all(SS.EVAL_ROOMS[a] == base[a] for a in aliases if a not in SS.EVAL_ROOM_MOVES)
    for a in aliases:
        assert all(SS.draw_scene_room("eval", a, k) == SS.EVAL_ROOMS[a] for k in range(4))
    with pytest.raises(KeyError):
        SS.draw_scene_room("eval", "object_scaling-not_in_the_table")
    # pinned: a changed derivation would silently move eval instances
    assert base["object_scaling-apple"] == ("house_double_floor_lower", "living_room_0")
    assert base["mental_rotation-mug"] == ("Rs_int", "living_room_0")


def test_eval_room_move_takes_the_tasks_next_unused_room():
    aliases = list(SS.EVAL_ROOMS)
    base, orders = SS.eval_assignment(aliases)
    a = "object_scaling-apple"
    held = set(base.values())
    first = next(x for x in orders["object_scaling"] if x not in held)
    moved, _ = SS.eval_assignment(aliases, moves=[a])
    assert moved[a] == first != base[a]
    assert {k: v for k, v in moved.items() if k != a} == {k: v for k, v in base.items() if k != a}  # nobody else moves
    twice, _ = SS.eval_assignment(aliases, moves=[a, a])  # never back to a room it left
    assert twice[a] == next(x for x in orders["object_scaling"] if x not in held | {first})
    with pytest.raises(KeyError):
        SS.eval_assignment(aliases, moves=["object_scaling-unknown"])


def test_train_draw_weights_rooms_by_capped_reachable_floor():
    """owner 2026-10-11 ~13:50: one room among all eligible rooms, weighted by its reachable floor capped at 3x the median."""
    rooms, weights, cap = SS.train_weights()
    assert rooms == SS.eligible_rooms() and cap == 2169  # 3 x the median 7.23 m^2, in 0.01 m^2
    assert {r for r, w in zip(rooms, weights) if w == cap} == {
        ("office_cubicles_right", "corridor_0"), ("office_cubicles_right", "conference_hall_0"), ("house_single_floor", "corridor_0"),
        ("house_single_floor", "living_room_0"), ("restaurant_diner", "dining_room_0"), ("house_double_floor_lower", "garage_0")}
    total = sum(weights)
    assert {sc: round(100 * sum(w for (s, _), w in zip(rooms, weights) if s == sc) / total, 1) for sc in SS.SCENE_CANDIDATES} == {
        "house_single_floor": 28.7, "house_double_floor_lower": 13.9, "house_double_floor_upper": 7.7, "Rs_int": 1.3, "restaurant_diner": 14.4,
        "hotel_suite_large": 4.7, "office_cubicles_right": 29.2}
    # pinned: a changed derivation would silently change every train set
    assert SS.draw_scene_room("train", "object_scaling-apple") == ("hotel_suite_large", "bedroom_0")
    assert SS.draw_scene_room("train", "mental_rotation-mug", 3) == ("office_cubicles_right", "conference_hall_0")
    # frequencies over many (alias, g) follow the weights (chi-square, 44 dof); only eligible rooms
    counts: dict = {}
    n = 0
    for i in range(200):
        for g in range(100):
            r = SS.draw_scene_room("train", f"object_scaling-x{i}", g)
            counts[r] = counts.get(r, 0) + 1
            n += 1
    assert set(counts) == set(rooms)
    chi2 = sum((counts[r] - n * w / total) ** 2 / (n * w / total) for r, w in zip(rooms, weights))
    assert chi2 < 80
    assert all(abs(counts[r] / n - cap / total) < 0.01 for r, w in zip(rooms, weights) if w == cap)


def test_frozen_room_table_matches_the_layout_maps():
    pytest.importorskip("cv2")
    try:
        measured = SS.measured_table(erode_px=6)
    except (FileNotFoundError, OSError) as e:
        pytest.skip(f"layout maps unavailable: {e}")
    assert measured == SS.ROOM_FLOOR_M2


# ----------------------------------------------------------------------------------------------------------------
# synthetic instances
# ----------------------------------------------------------------------------------------------------------------


def _q_yaw(yaw):
    return [0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]


def _tro(task, alias, base, scene="house_single_floor", room="living_room_0", cam=None, obj=None, iid=None):
    x, y, yaw = base
    isg = {"activity": task, "task_variant": alias, "instance_id": iid or f"{alias}__{scene}__0000", "scene": {"model": scene, "task_rooms": [room]},
           "pose_targets": {}}
    tro = {"robot_poses": {"robot": [{"position": [x, y, 0.005], "orientation": _q_yaw(yaw)}]}, "ispatialgym": isg}
    if cam is not None:
        isg["pose_targets"][SEP.AGENT] = {"frame": "world", "position": list(cam[0]), "orientation": list(cam[1])}
    if obj is not None:
        tro["mug.n.04_1"] = {"root_link": {"pos": list(obj[0]), "ori": list(obj[1]), "lin_vel": [0, 0, 0]}}
        isg["grasps"] = [{"object": "mug.n.04_1", "object_pose_in_eef_frame": {"position": [0.0, 0.0, 0.1], "orientation": [0.0, 0.0, 0.0, 1.0]}}]
    return tro


def _camera(alias, start, goal, room="living_room_0", scene="house_single_floor", cam=None):
    sc = {"alias": alias, "stances": {"init": {"base_pose2d": list(start)}, "goal": {"base_pose2d": list(goal)}}, "goal": {"base_pose2d": list(goal)}}
    return _tro(SEP.CAMERA, alias, start, scene, room, cam=cam), sc


def _held(alias, base, obj0, eef_goal, task="mental_rotation", room="living_room_0", scene="house_single_floor"):
    sc = {"alias": alias, "stances": {"init": {"base_pose2d": list(base)}}, "goal": {"base_pose2d": list(base),
                                                                                      "eef": {"position": list(eef_goal[0]), "orientation": list(eef_goal[1])}}}
    return _tro(task, alias, base, scene, room, obj=obj0), sc


def test_instance_record_reads_start_goal_and_held_object_poses():
    tro, sc = _camera("camera_relocalization-standard", (1.0, 2.0, 0.5), (4.0, 5.0, 1.0), cam=([4.1, 5.0, 1.4], _q_yaw(1.0)))
    r = SEP.record_from_dicts(tro, sc)
    assert (r.task, r.alias, r.scene, r.room) == (SEP.CAMERA, "camera_relocalization-standard", "house_single_floor", "living_room_0")
    assert r.start_base == pytest.approx([1.0, 2.0, 0.5]) and r.goal_base == pytest.approx([4.0, 5.0, 1.0])
    assert [p.name for p in r.start] == ["start_base"] and [p.name for p in r.goal] == ["goal_camera"]
    del tro["ispatialgym"]["pose_targets"][SEP.AGENT]
    assert [p.name for p in SEP.record_from_dicts(tro, sc).goal] == ["goal_base"]
    # no TRO robot pose: the sidecar's init stance
    tro2 = dict(tro, robot_poses={})
    assert SEP.record_from_dicts(tro2, sc).start_base == pytest.approx([1.0, 2.0, 0.5])
    with pytest.raises(ValueError):
        SEP.record_from_dicts(tro2, {})
    # held object: the base never moves; the object goal = goal eef o object-in-eef (+0.1 m along the eef z)
    q90 = [math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4)]  # 90 deg about x: eef z -> world -y
    tro, sc = _held("mental_rotation-mug", (3.0, 3.0, 0.0), ([3.5, 3.0, 1.2], [0, 0, 0, 1]), ([3.5, 3.1, 1.2], q90))
    r = SEP.record_from_dicts(tro, sc)
    assert r.goal_base == r.start_base and [p.name for p in r.start] == ["start_base", "object_start"]
    g = r.goal[0]
    assert g.name == "object_goal" and g.p == pytest.approx([3.5, 3.0, 1.2], abs=1e-9) and SEP.quat_gap_deg(g.q, q90) < 1e-6


def test_separation_check_finds_close_pairs_per_task_scene_and_room():
    recs = [SEP.record_from_dicts(*_camera("cam-a", (0.0, 0.0, 0.0), (5.0, 5.0, 0.0))),
            SEP.record_from_dicts(*_camera("cam-b", (0.6, 0.0, 0.0), (9.0, 9.0, 0.0))),   # start 0.6 m from a
            SEP.record_from_dicts(*_camera("cam-c", (5.0, 0.0, 0.0), (5.3, 5.0, 0.0))),   # goal 0.3 m from a
            SEP.record_from_dicts(*_camera("cam-d", (0.1, 0.0, 0.0), (5.0, 5.1, 0.0), room="dining_room_0")),  # another room
            SEP.record_from_dicts(*_held("mental_rotation-x", (0.0, 0.1, 0.0), ([0, 0, 1], [0, 0, 0, 1]), ([0, 0, 1], [0, 0, 0, 1]))),  # another task
            SEP.record_from_dicts(*_held("mental_rotation-y", (5.0, 5.0, 0.0), ([0, 0, 1], [0, 0, 0, 1]), ([0, 0, 1], [0, 0, 0, 1])))]
    off = SEP.separation_offenders(recs, 1.0)
    assert sorted((o.a.alias, o.b.alias, o.kind) for o in off) == [("cam-a", "cam-b", "start"), ("cam-a", "cam-c", "goal")]
    assert SEP.separation_offenders(recs, 0.25) == []
    # held-object pairs compare their start bases only
    recs.append(SEP.record_from_dicts(*_held("mental_rotation-z", (5.5, 5.0, 2.0), ([9, 9, 1], [0, 0, 0, 1]), ([9, 9, 1], [0, 0, 0, 1]))))
    assert [(o.a.alias, o.b.alias, o.kind) for o in SEP.separation_offenders(recs, 1.0) if o.a.task != SEP.CAMERA] == [
        ("mental_rotation-y", "mental_rotation-z", "start")]
    # the later alias of a pair is re-sampled; an alias whose earlier partner is re-sampled is kept
    off = SEP.separation_offenders(recs[:3], 1.0)
    assert set(SEP.resample_choice(off)) == {"cam-b", "cam-c"}
    assert set(SEP.resample_choice(off, order=["cam-c", "cam-b", "cam-a"])) == {"cam-a"}
    # the next seed index: one past the split index, 0 for a legacy --seed run
    recs[1].split = {"split": "eval", "index": 2, "seed": 7, "seed_source": "split"}
    assert SEP.next_seed_index(recs, "cam-b")[0] == 3 and SEP.next_seed_index(recs, "cam-c")[0] == 0


def _write(dirpath, iid, tro, sidecar, meta=None):
    inst = dirpath / "instances"
    inst.mkdir(parents=True, exist_ok=True)
    (inst / f"{iid}-tro_state.json").write_text(json.dumps(tro))
    (inst / f"{iid}.oracle.json").write_text(json.dumps(sidecar))
    if meta is not None:
        (dirpath / "template_meta.json").write_text(json.dumps(meta))
    return str(inst / f"{iid}-tro_state.json")


def test_separation_check_cli_on_run_tags(tmp_path, capsys):
    runs = tmp_path / "runs"
    for alias, start, goal in (("cam-a", (0.0, 0.0, 0.0), (5.0, 5.0, 0.0)), ("cam-b", (0.5, 0.0, 0.0), (9.0, 9.0, 0.0))):
        tro, sc = _camera(alias, start, goal)
        tpl = runs / "t1" / alias / "inst" / alias / "house_single_floor"
        _write(tpl, f"{alias}__house_single_floor__0000", tro, sc, {"split": {"split": "eval", "index": 0, "seed": 1, "seed_source": "split"}})
        _write(runs / "t1" / alias / "replay_inst", f"{alias}__house_single_floor__0000", tro, sc)  # the job's replay copy is not an instance
    # an alias's run set aside (renamed) is skipped
    tro, sc = _camera("cam-a", (9.0, 9.0, 0.0), (0.0, 0.0, 0.0))
    _write(runs / "t1" / "cam-a.hung_1" / "inst" / "cam-a" / "house_single_floor", "cam-a__house_single_floor__0000", tro, sc)
    assert SEP.main(["t1", "--write-index", str(tmp_path / "idx.json")], runs_root=str(runs)) == 1
    out = capsys.readouterr().out
    assert "2 instances of 2 aliases" in out and "RESAMPLE cam-b --seed-split eval --seed-index 1" in out and "SEPARATION FAIL pairs=1" in out
    assert len(SEP.load_instances([str(tmp_path / "idx.json")])) == 2
    # the re-sampled alias in a later run replaces its instance: clean
    tro, sc = _camera("cam-b", (3.0, 0.0, 0.0), (9.0, 9.0, 0.0))
    _write(runs / "t2" / "cam-b" / "inst" / "cam-b" / "house_single_floor", "cam-b__house_single_floor__0000", tro, sc)
    assert SEP.main(["t1", "t2"], runs_root=str(runs)) == 0
    assert "SEPARATION OK" in capsys.readouterr().out


# ----------------------------------------------------------------------------------------------------------------
# train-repeat rule
# ----------------------------------------------------------------------------------------------------------------


def test_train_repeat_needs_both_start_and_goal_within_tolerance():
    cam = ([4.0, 5.0, 1.4], _q_yaw(1.0))
    ev = SEP.record_from_dicts(*_camera("cam-a", (1.0, 2.0, 0.5), (4.0, 5.0, 1.0), cam=cam))

    def train(start, cam_goal, scene="house_single_floor"):
        return SEP.record_from_dicts(*_camera("cam-a", start, (0.0, 0.0, 0.0), scene=scene, cam=cam_goal))

    hit = SEP.find_repeat(train((1.15, 2.1, 0.5 + math.radians(10)), ([4.1, 5.1, 1.4], _q_yaw(1.0 + math.radians(10)))), [ev])
    assert hit is not None and hit[0] is ev and set(hit[1]) == {"start_base", "goal_camera"}
    assert SEP.find_repeat(train((1.15, 2.1, 0.5), ([4.3, 5.0, 1.4], _q_yaw(1.0))), [ev]) is None  # goal 0.3 m away
    assert SEP.find_repeat(train((1.0, 2.0, 0.5 + math.radians(20)), cam), [ev]) is None  # start turned 20 deg
    assert SEP.find_repeat(train((3.0, 2.0, 0.5), cam), [ev]) is None  # start far, goal equal: similar is fine
    assert SEP.find_repeat(train((1.0, 2.0, 0.5), cam, scene="Rs_int"), [ev]) is None  # another scene
    assert SEP.find_repeat(train((1.0, 2.0, 0.5), ([4.0, 5.0, 1.4], _q_yaw(1.0))), [ev], pos_m=1e-9, ang_deg=1e-3) is not None
    # held object: start base, object start and object goal all within
    q0 = [0.0, 0.0, 0.0, 1.0]
    evh = SEP.record_from_dicts(*_held("mental_rotation-mug", (3.0, 3.0, 0.0), ([3.5, 3.0, 1.2], q0), ([3.5, 3.0, 1.1], q0)))
    same = SEP.record_from_dicts(*_held("mental_rotation-mug", (3.1, 3.0, 0.1), ([3.6, 3.0, 1.2], q0), ([3.6, 3.0, 1.1], q0)))
    turned = SEP.record_from_dicts(*_held("mental_rotation-mug", (3.1, 3.0, 0.1), ([3.6, 3.0, 1.2], q0), ([3.6, 3.0, 1.1], _q_yaw(math.radians(30)))))
    assert SEP.find_repeat(same, [evh]) is not None and SEP.find_repeat(turned, [evh]) is None
    assert SEP.find_repeat(same, [ev]) is None  # another task


def test_repeat_filter_reads_the_instance_just_written(tmp_path):
    cam = ([4.0, 5.0, 1.4], _q_yaw(1.0))
    _write(tmp_path / "eval" / "cam-a" / "house_single_floor", "cam-a__house_single_floor__0000", *_camera("cam-a", (1.0, 2.0, 0.5), (4.0, 5.0, 1.0), cam=cam))
    evals = SEP.load_instances([str(tmp_path / "eval")])
    assert len(evals) == 1
    filt = SEP.repeat_filter(evals, task=SEP.CAMERA)
    p_rep = _write(tmp_path / "train" / "cam-a" / "house_single_floor", "cam-a__house_single_floor__0001", *_camera("cam-a", (1.1, 2.0, 0.5), (4.0, 5.0, 1.0), cam=cam))
    p_new = _write(tmp_path / "train" / "cam-a" / "house_single_floor", "cam-a__house_single_floor__0002", *_camera("cam-a", (2.0, 2.0, 0.5), (4.0, 5.0, 1.0), cam=cam))
    why = filt(p_rep)
    assert why.startswith("repeats eval instance cam-a__house_single_floor__0000 (cam-a)") and "start_base 0.10 m" in why
    assert filt(p_new) is None
    assert SEP.repeat_filter(evals, task="object_scaling")(p_rep) is None


def _filter_sampler(tmp_path, outcomes, instance_filter):
    """A ``SamplerV2`` whose attempts all succeed and write real files; ``instance_filter`` judges them."""
    from types import SimpleNamespace as NS

    from ispatialgym.sampling.v2.core import SamplerV2

    s = object.__new__(SamplerV2)
    noop = lambda *a, **k: None  # noqa: E731
    s.ctx = NS(base_seed=0)
    s.kit = NS(stats_snapshot=lambda: {}, stats_delta=lambda a, b: {})
    s.world = NS(reset_scene_objects=lambda names=None, context="": 0)
    s.objects, s.counters, s.template_meta, s.rejections, s.instance_seeds = {}, {}, {"stances": {}}, {}, []
    s.template_paths = None
    s.ikc = NS(solutions={}, attempt=None)
    s.gp = NS(alias="alias-test")
    s.reasons = []
    s.rejection_log = NS(add=lambda reason, *a, **k: s.reasons.append(reason), counts={})
    s.sample = lambda rng: NS(summary={})
    for name in ("park_robot", "apply_draft", "settle_and_validate", "check_stances_with_objects", "check_goal_feasible", "after_goal_check",
                 "verify_carry_legs", "program_check", "save_template", "check_goal_occlusion", "constraints_write_back", "write_template_meta", "on_reject"):
        setattr(s, name, noop)
    s.instance_id = lambda index: f"iid{index:04d}"

    def write_instance(draft, index, seed):
        p = tmp_path / f"iid{index:04d}-tro_state.json"
        for suffix in ("-tro_state.json", ".oracle.json", ".summary.json"):
            (tmp_path / f"iid{index:04d}{suffix}").write_text("{}")
        return str(p)

    s.write_instance = write_instance
    verdicts = iter(outcomes)
    s.instance_filter = (lambda path: next(verdicts)) if instance_filter else None
    return s


def test_instance_filter_rejects_and_removes_a_repeat_and_explicit_indices(tmp_path):
    s = _filter_sampler(tmp_path, ["repeats eval instance X", None, None], True)
    written = s.run_instances(2, max_attempts=3, indices=[3, 7])
    assert [os.path.basename(p) for p in written] == ["iid0003-tro_state.json", "iid0007-tro_state.json"]
    assert s.reasons == ["repeats_eval"]
    assert s.instance_seeds == [0 * 1000003 + 3 * 1009 + 1, 7 * 1009]
    assert sorted(os.listdir(tmp_path)) == sorted(f"iid{i:04d}{x}" for i in (3, 7) for x in ("-tro_state.json", ".oracle.json", ".summary.json"))
    # no filter: unchanged behaviour, indices from start_index
    s = _filter_sampler(tmp_path / "b", [], False)
    (tmp_path / "b").mkdir()
    assert [os.path.basename(p) for p in s.run_instances(2, max_attempts=1, start_index=5)] == ["iid0005-tro_state.json", "iid0006-tro_state.json"]


# ----------------------------------------------------------------------------------------------------------------
# problem: the room override, the template id, the write-back
# ----------------------------------------------------------------------------------------------------------------


def test_room_override_template_id_and_write_back(tmp_path):
    customs = tmp_path / "customs.json"
    customs.write_text(json.dumps({"object_scaling-apple": _held_entry(), "_meta": {"x": 1}}))
    gp = GroundedProblem("object_scaling-apple", json.loads(customs.read_text())["object_scaling-apple"], customs_path=str(customs))
    assert gp.floor_only() and gp.room_tag("house_single_floor") is None
    # a scene the entry does not list: the block is synthesized from the entry's first block
    assert gp.override_scene_room("Rs_int", "living_room_0", note="test") is True
    assert gp.scene_names() == ["house_single_floor", "Rs_int"] and gp.rooms("Rs_int") == ["living_room_0"]
    assert gp.bindings("Rs_int") == {"floor": {"room": "living_room_0"}} and gp.whitelist("Rs_int", "target") == {"apple": ["a"]}
    assert gp.blacklist("Rs_int", "target") == {"apple": {"b": "heldout"}} and gp.room_tag("Rs_int") == "living_room_0"
    choice = next(iter_template_choices(gp, "Rs_int", np.random.default_rng(0)))
    assert choice.template_id == "Rs_int__room-living_room_0__target-apple-a" and choice.room_tag == "living_room_0"
    assert choice.bindings == {"floor": {"room": "living_room_0"}} and choice.rooms == ["living_room_0"]
    back = TemplateChoice.from_json(json.loads(json.dumps(choice.to_json())))
    assert back.template_id == choice.template_id and back.room_tag == "living_room_0"
    legacy = {k: v for k, v in choice.to_json().items() if k != "room_tag"}
    assert TemplateChoice.from_json(legacy).template_id == "Rs_int__target-apple-a"
    # another room of a listed scene, then that scene's own room again (the entry's block comes back, no tag)
    spec0 = gp.spec_hash()
    assert gp.override_scene_room("house_single_floor", "kitchen_0") is True and gp.room_tag("house_single_floor") == "kitchen_0"
    assert gp.spec_hash() != spec0
    gp.add_blacklist("house_single_floor", "target", "apple", "a", "failed in the kitchen")
    gp.save_customs(str(tmp_path / "copy.json"))
    gp.save_customs()
    src = json.loads(customs.read_text())
    assert src["_meta"] == {"x": 1} and list(src["object_scaling-apple"]["scenes"]) == ["house_single_floor"]
    assert src["object_scaling-apple"]["scenes"]["house_single_floor"] == _held_entry()["scenes"]["house_single_floor"]
    cp = json.loads((tmp_path / "copy.json").read_text())["object_scaling-apple"]["scenes"]
    assert cp["house_single_floor"]["rooms"] == ["kitchen_0"] and cp["house_single_floor"]["_synthesized"] and cp["Rs_int"]["rooms"] == ["living_room_0"]
    assert gp.override_scene_room("house_single_floor", "living_room_0") is False
    assert gp.entry["scenes"]["house_single_floor"] == _held_entry()["scenes"]["house_single_floor"] and gp.room_tag("house_single_floor") is None
    assert gp.override_scene_room("house_single_floor", "living_room_0") is False  # no override: nothing to do
    # aliases with other scene roles cannot move rooms
    iors = _held_entry()
    iors["roles"]["support"] = {"category": "breakfast_table", "sampleable": False}
    with pytest.raises(GroundingError):
        GroundedProblem("object_scaling-apple", iors).override_scene_room("Rs_int", "living_room_0")


# ----------------------------------------------------------------------------------------------------------------
# sample_isg / multiply_isg plumbing
# ----------------------------------------------------------------------------------------------------------------


def _args(**kw):
    d = {"seed": None, "seed_split": None, "seed_index": 0, "scene": None, "room": None, "scene_from_split": False}
    d.update(kw)
    return SimpleNamespace(**d)


def test_sample_isg_alias_split_seed_scene_and_room():
    from ispatialgym.sampling import sample_isg as SI

    held = GroundedProblem("object_scaling-apple", _held_entry())
    seed, rec = SI.alias_split(_args(seed_split="eval", seed_index=1, scene_from_split=True), held)
    scene, room = SS.draw_scene_room("eval", "object_scaling-apple", 1)
    assert (scene, room) == SS.EVAL_ROOMS["object_scaling-apple"]  # the re-sample (k = 1) keeps the alias's room
    assert seed == SS.split_seed("eval", "object_scaling-apple", 1)
    assert rec == {"split": "eval", "index": 1, "seed": seed, "seed_source": "split", "scene": scene, "room": room, "scene_source": "split",
                   "room_override": (scene, room) != ("house_single_floor", "living_room_0")}
    assert held.rooms(scene) == [room] and held.bindings(scene) == {"floor": {"room": room}}
    cam = GroundedProblem("camera_relocalization-standard", _camera_entry())
    seed, rec = SI.alias_split(_args(seed_split="eval", scene_from_split=True), cam)
    assert seed == SS.split_seed("eval", "camera_relocalization-standard") and rec["scene"] is None and rec["scene_source"].startswith("customs")
    assert SI.alias_split(_args(seed=5, seed_split="eval"), cam)[1]["seed_source"] == "--seed"
    assert SI.alias_split(_args(), cam)[0] == 0
    seed, rec = SI.alias_split(_args(scene="hotel_suite_large", room="bedroom_0"), cam)
    assert rec["scene_source"] == "--scene --room" and cam.rooms("hotel_suite_large") == ["bedroom_0"] and cam.room_tag("hotel_suite_large") == "bedroom_0"


def test_sample_isg_existing_templates_are_per_room(tmp_path):
    from ispatialgym.sampling import sample_isg as SI

    gp = GroundedProblem("object_scaling-apple", _held_entry())
    choice = next(iter_template_choices(gp, "house_single_floor", np.random.default_rng(0)))
    d = tmp_path / gp.alias / choice.template_id
    d.mkdir(parents=True)
    (d / "template_meta.json").write_text(json.dumps({"scene": "house_single_floor", "rooms": ["living_room_0"], "choice": choice.to_json()}))
    assert SI._existing_templates(str(tmp_path), gp, "house_single_floor") == [{"target": "apple/a"}]
    gp.override_scene_room("house_single_floor", "kitchen_0")
    assert SI._existing_templates(str(tmp_path), gp, "house_single_floor") == []


def test_multiply_train_plan_indices_and_rooms(tmp_path):
    from ispatialgym.sampling import multiply_isg as MI

    gp = GroundedProblem("object_scaling-apple", _held_entry())
    choice = next(iter_template_choices(gp, "house_single_floor", np.random.default_rng(0)))
    plan = MI.train_plan("object_scaling-apple", choice, 4, 2, True)
    assert [g for g, _, _ in plan] == [2, 3, 4, 5] and all((s, r) == SS.draw_scene_room("train", "object_scaling-apple", g) for g, s, r in plan)
    assert all((s, r) in SS.eligible_rooms() for _, s, r in plan)
    assert MI.train_plan("object_scaling-apple", choice, 2, 0, False) == [(0, "house_single_floor", "living_room_0"), (1, "house_single_floor", "living_room_0")]
    moved = MI.choice_in_room(gp, choice, "restaurant_diner", "kitchen_0")
    assert moved.template_id == "restaurant_diner__room-kitchen_0__target-apple-a" and moved.models == choice.models
    assert MI.choice_in_room(gp, choice, "house_single_floor", "living_room_0").template_id == choice.template_id
    # the template's own room is re-applied from its meta (a template sampled with --room)
    gp2 = GroundedProblem("object_scaling-apple", _held_entry())
    assert MI.choice_for_template(gp2, {"choice": moved.to_json()}).template_id == moved.template_id and gp2.rooms("restaurant_diner") == ["kitchen_0"]
    # instance indices: one past the largest, gaps kept; train indices per alias over its templates
    for tid, idx in (("t1", 0), ("t1", 3), ("t2", 5)):
        p = tmp_path / "object_scaling-apple" / tid / "instances"
        p.mkdir(parents=True, exist_ok=True)
        (p / f"object_scaling-apple__{tid}__{idx:04d}-tro_state.json").write_text("{}")
    assert MI.next_index(str(tmp_path / "object_scaling-apple" / "t1")) == 4 and MI.next_index(str(tmp_path / "nowhere")) == 0
    assert MI.next_train_index(str(tmp_path), "object_scaling-apple") == 6 and MI.next_train_index(str(tmp_path), "other") == 0
    assert MI.template_seed(SimpleNamespace(seed=None, split="train", seed_index=0), "object_scaling-apple")[0] == SS.split_seed("train", "object_scaling-apple")
    assert MI.template_seed(SimpleNamespace(seed=None, split=None, seed_index=0), "x")[0] == 1


def test_no_garden_or_outdoor_room_is_ever_drawn(monkeypatch):
    """owner 2026-10-11 13:31: object scaling / mental rotation never draw a garden or any other outdoor room, eval and train,
    however much floor it has (the eval assignment and multiply_isg's train draws share eligible_rooms)."""
    from ispatialgym.sampling import split_seeds as S

    assert S.is_outdoor("garden_0") and S.is_outdoor("balcony_1") and S.is_outdoor("back_garden_0") and not S.is_outdoor("living_room_0")
    table = {"house_single_floor": {"garden_0": (400.0, 300.0, 250.0), "living_room_0": (30.0, 20.0, 10.0)}}
    assert S.eligible_rooms(table=table) == [("house_single_floor", "living_room_0")]
    assert S.train_weights(table)[0] == [("house_single_floor", "living_room_0")]
    monkeypatch.setattr(S, "ROOM_FLOOR_M2", table)  # the largest room is a garden: never drawn
    assert {S.draw_scene_room("train", "object_scaling-apple", g) for g in range(64)} == {("house_single_floor", "living_room_0")}
    monkeypatch.undo()
    assert not any(S.is_outdoor(r) for _, r in S.eligible_rooms())
    aliases = ["object_scaling-apple", "mental_rotation-crab", "object_scaling-bell_pepper", "mental_rotation-teaspoon"]
    for split in S.SPLITS:
        for a in aliases:
            for k in range(64):
                sc, room = S.draw_scene_room(split, a, k)
                assert not S.is_outdoor(room) and (sc, room) in S.eligible_rooms()
