"""Simulator-free tests of the v2 sampler core: the §9 model draw, the §3.8 stand-off enumeration, the §7 path record
schema and the §3.5 summary layout (``pytest ispatialgym/tests/test_v2_core.py``)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest
import torch as th

from ispatialgym.sampling.v2 import navigation as NAV
from ispatialgym.sampling.v2 import program_check as PC
from ispatialgym.sampling.v2.problem import (
    GroundedProblem,
    LookalikeIndex,
    ModelCycler,
    RejectionLog,
    TemplateChoice,
    draw_group,
    iter_template_choices,
    new_run_summary,
    subtask_of_alias,
    template_summary,
)
from ispatialgym.sampling.v2.stances import ContactRecord, EdgeFrame, GraspRecord, PathRecord, SidecarDraft, Stance, Transition, path_length


def _entry(target_models, partner_models, groups=None, templates_max=20):
    return {
        "iddl": "inter_object_relation_satisfaction/ontop_small_scale.iddl",
        "definition_id": 0,
        "roles": {
            "target": {"category": "apple", "sampleable": True, "require": {"grasps": "tabletop"}, "representative": target_models[0]},
            "partner": {"category": "plate", "sampleable": True, "representative": partner_models[0]},
            "support": {"category": "breakfast_table", "sampleable": False},
            "floor": {"sampleable": False},
        },
        "scenes": {
            "house_single_floor": {
                "rooms": ["dining_room_0"],
                "bindings": {"support": ["breakfast_table_rhjoby_0"], "floor": {"room": "dining_room_0"}},
                "whitelist": {"target": {"apple": list(target_models)}, "partner": {"plate": list(partner_models)}},
                "blacklist": {},
            }
        },
        "generation": {"templates_max": templates_max, "instances_per_template": 1, "tolerances": {"position": 0.03, "orientation": 0.2},
                       "groups": groups or {}, "path_buckets": None},
        "constraints": [],
    }


def _problem(entry, lookalikes=None):
    return GroundedProblem("inter_object_relation_satisfaction-ontop_small_scale-apple-plate-breakfast_table", entry, lookalikes=lookalikes)


# ----------------------------------------------------------------------------------------------------------------
# model draw (§9, §3.12)
# ----------------------------------------------------------------------------------------------------------------


def test_subtask_of_alias():
    assert subtask_of_alias("inter_object_relation_satisfaction-ontop_small_scale-apple-plate-breakfast_table") == "inter_object_relation_satisfaction-ontop_small_scale"
    assert subtask_of_alias("configuration_matching-block_design-block_4-breakfast_table") == "configuration_matching-block_design"
    assert subtask_of_alias("x", "inter_object_relation_satisfaction/nextto.iddl") == "inter_object_relation_satisfaction-nextto"


def test_cycler_iterates_all_models_before_reuse():
    rng = np.random.default_rng(0)
    cy = ModelCycler("target", {"apple": ["a", "b", "c"]}, rng, LookalikeIndex())
    seen = [cy.next().model for _ in range(6)]
    assert sorted(seen[:3]) == ["a", "b", "c"]
    assert sorted(seen[3:]) == ["a", "b", "c"]
    assert cy.n_passes == 2


def test_cycler_counts_exact_duplicates_once_and_variants_separately():
    la = LookalikeIndex(duplicate_of={("apple", "b"): ("apple", "a")}, geometry_group={("apple", "c"): "g1", ("apple", "d"): "g1"})
    cy = ModelCycler("target", {"apple": ["a", "b", "c", "d"]}, np.random.default_rng(0), la)
    assert len(cy) == 3  # a (b is its exact duplicate), c, d (texture variants stay distinct)
    assert {m for _, m in cy.pool} == {"a", "c", "d"}


def test_templates_unique_across_roles_and_no_lookalike_pairs():
    # apple/x and plate/x share a look-alike group: a template must never pair them
    la = LookalikeIndex(groups=[{("apple", "x"), ("plate", "x")}])
    gp = _problem(_entry(["x", "y"], ["x", "z"]), la)
    rng = np.random.default_rng(1)
    templates = list(iter_template_choices(gp, "house_single_floor", rng))
    assert templates, "no templates drawn"
    for t in templates:
        assert not la.lookalikes(("apple", t.models["target"].model), ("plate", t.models["partner"].model))
        canon = {la.canonical(c.category, c.model) for c in t.models.values()}
        assert len(canon) == len(t.models)
    ids = [t.template_id for t in templates]
    assert len(ids) == len(set(ids))
    assert len(templates) <= 4


def test_templates_respect_templates_max_and_exclude():
    gp = _problem(_entry(["a", "b", "c"], ["p", "q"], templates_max=3))
    rng = np.random.default_rng(2)
    templates = list(iter_template_choices(gp, "house_single_floor", rng))
    assert len(templates) == 3
    excl = [{"target": f"apple/{t.models['target'].model}", "partner": f"plate/{t.models['partner'].model}"} for t in templates[:1]]
    more = list(iter_template_choices(gp, "house_single_floor", np.random.default_rng(2), exclude=excl))
    keys = {(t.models["target"].model, t.models["partner"].model) for t in more}
    assert (templates[0].models["target"].model, templates[0].models["partner"].model) not in keys
    assert len(more) == 2


def test_blacklist_prunes_whitelist_and_hard_records():
    gp = _problem(_entry(["a", "b"], ["p"]))
    gp.add_blacklist("house_single_floor", "target", "apple", "a", "sample_isg: grasp_unreachable")
    assert gp.whitelist("house_single_floor", "target") == {"apple": ["b"]}
    assert gp.blacklist("house_single_floor", "target")["apple"]["a"].startswith("sample_isg")
    gp.add_constraint({"type": "blacklist", "scope": {"scene": "house_single_floor", "target": "apple/b"}, "hard": True})
    assert list(iter_template_choices(gp, "house_single_floor", np.random.default_rng(0))) == []


def test_group_draw_same_model_and_unique():
    entry = _entry(["a"], ["p"], groups={"targets": {"size": 4, "draw": "same_model"}, "distractors": {"size": [1, 3], "draw": "unique"}})
    entry["roles"]["targets"] = {"category": "block", "sampleable": True}
    entry["roles"]["distractors"] = {"category": "beaker", "sampleable": True}
    entry["scenes"]["house_single_floor"]["whitelist"]["targets"] = {"block": ["k", "l"]}
    entry["scenes"]["house_single_floor"]["whitelist"]["distractors"] = {"beaker": ["d1", "d2", "d3", "d4"]}
    la = LookalikeIndex(groups=[{("beaker", "d1"), ("beaker", "d2")}])
    gp = _problem(entry, la)
    assert gp.group_roles() == ["targets", "distractors"]
    rng = np.random.default_rng(3)
    blocks = draw_group(gp, "targets", "house_single_floor", rng, set())
    assert len(blocks) == 4 and len({b.model for b in blocks}) == 1
    for _ in range(20):
        dis = draw_group(gp, "distractors", "house_single_floor", rng, set())
        assert dis is not None and 1 <= len(dis) <= 3
        models = [d.model for d in dis]
        assert len(set(models)) == len(models)
        assert not ({"d1", "d2"} <= set(models)), "two members of one look-alike group"
    t = next(iter_template_choices(gp, "house_single_floor", rng))
    assert len(t.groups["targets"]) == 4 and "targets-block_4" in t.template_id
    assert "targets[0]" in t.choices


def test_group_members_per_category_follow_the_alias():
    # D1 mixed-category groups: ``members`` fixes the count per category (the grounded BDDL declares exactly those)
    entry = _entry(["a"], ["p"], groups={"distractors": {"size": 3, "draw": "unique", "members": {"bottle_of_beer": 2, "bottle_of_sake": 1}}})
    entry["roles"]["distractors"] = {"category": ["bottle_of_beer", "bottle_of_sake"], "sampleable": True}
    entry["scenes"]["house_single_floor"]["whitelist"]["distractors"] = {"bottle_of_beer": ["b1", "b2", "b3"], "bottle_of_sake": ["s1", "s2"]}
    gp = _problem(entry)
    assert gp.roles["distractors"].categories == ["bottle_of_beer", "bottle_of_sake"]
    rng = np.random.default_rng(4)
    for _ in range(10):
        members = draw_group(gp, "distractors", "house_single_floor", rng, set())
        assert [m.category for m in members] == ["bottle_of_beer", "bottle_of_beer", "bottle_of_sake"]
        assert len({m.model for m in members}) == 3
    t = next(iter_template_choices(gp, "house_single_floor", rng))
    assert "distractors-bottle_of_beer_2-bottle_of_sake_1" in t.template_id
    assert subtask_of_alias("configuration_matching-articulation_open_large_scale-bottom_cabinet") == "configuration_matching-articulation_open_large_scale"


def test_stance_sidecar_keeps_extras_and_recorded_configuration():
    edge = EdgeFrame.from_edge("+u", [1.0, 0.0], [1.0, 0.0], 1.6)
    st = Stance("init", (1.45, 0.0, math.pi), 0.5, 0.0, 0.45, edge, arms="reset", carried="bowl.n.01_1",
                extras={"goal_cell": [0.3, -0.1], "bearing_deg": 12.0, "_private": 1})
    d = st.to_sidecar()
    assert d["goal_cell"] == [0.3, -0.1] and d["bearing_deg"] == 12.0 and "_private" not in d
    assert d["arms"] == "reset" and d["carried"] == "bowl.n.01_1" and d["edge"] == "+u"
    assert st.posture == "tucked" or st.posture == "reset"  # the reset arms map to the hanging (zero) posture
    st2 = st.with_trunk(0.75, -0.3)
    assert st2.carried == "bowl.n.01_1" and st2.fingers is None and st2.extras["goal_cell"] == [0.3, -0.1]


def test_constraint_records_and_scope_query():
    gp = _problem(_entry(["a"], ["p"]))
    rec = gp.add_constraint({"type": "stance", "scope": {"scene": "house_single_floor", "support": "breakfast_table_rhjoby_0"}, "edges": ["+u"]},
                            source_instance="iid_0001", sampler_version="2.0.0-dev")
    assert rec["source"]["instance"] == "iid_0001" and rec["hard"] is False and "date" in rec["source"]
    assert gp.constraints_for("stance", scene="house_single_floor") == [rec]
    assert gp.constraints_for("stance", scene="Rs_int") == []
    with pytest.raises(Exception):
        gp.add_constraint({"scope": {}})


def test_v1_facade_roles_and_scene_view():
    gp = _problem(_entry(["a"], ["p"]))
    assert gp.activity == "inter_object_relation_satisfaction" and gp.name == gp.alias
    assert {r.name for r in gp.spawned_roles()} == {"target", "partner"}
    assert {r.name for r in gp.fixed_roles()} == {"support", "floor"}
    ss = gp.scenes["house_single_floor"]
    assert ss.rooms == ["dining_room_0"] and ss.fixed["support"] == ["breakfast_table_rhjoby_0"]
    assert gp.gen("instances_per_template") == 1
    tc = TemplateChoice.from_json(json.loads(json.dumps(next(iter_template_choices(gp, "house_single_floor", np.random.default_rng(0))).to_json())))
    assert tc.template_id == "house_single_floor__partner-plate-p__target-apple-a"


# ----------------------------------------------------------------------------------------------------------------
# transitions (§3.8), paths (§3.7)
# ----------------------------------------------------------------------------------------------------------------


def test_transition_standoff_enumeration():
    assert NAV.transition_standoffs(0.45) == [0.45, 0.55, 0.65, 0.75, 0.85, 0.95]
    assert NAV.transition_standoffs(0.35)[0] == 0.35 and len(NAV.transition_standoffs(0.35)) == 6
    sweep = NAV.trunk_sweep((0.5, 0.0), (0.75, -0.4), 5)
    assert len(sweep) == 5 and sweep[0] == (0.5, 0.0) and sweep[-1] == pytest.approx((0.75, -0.4))
    assert Transition("init", "manipulation", 0.55).to_sidecar() == {"from": "init", "to": "manipulation", "stand_off": 0.55,
                                                                     "order": ["drive-back", "trunk", "drive-in"]}


def test_stride_poses_yaw_envelope_covers_the_trackers_lag_on_a_turn():
    """Owner 2026-09-25 (shampoo 0925-211757: the finger brushed a wardrobe with the base yaw 6 deg behind the interpolated one on a
    57 deg turn): with ``yaw_envelope`` every stride pose is repeated at the yaws the reference held over the last 0.25 m of path
    (their hull with the interpolated yaw, every 10 deg); a straight path with a constant yaw adds nothing."""
    import math

    straight = NAV.stride_poses([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], 0.1, yaw_envelope=True)
    assert straight == NAV.stride_poses([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], 0.1)
    # a 36 cm segment turning 57 deg (146 -> 89), then 90 deg in place: the shampoo shuffle
    wps = [(23.155, -0.68, math.radians(137.0)), (23.1, -0.7, math.radians(146.0)), (22.8, -0.5, math.radians(89.0)), (22.801, -0.447, math.radians(180.0))]
    plain = NAV.stride_poses(wps, 0.05)
    env = NAV.stride_poses(wps, 0.05, yaw_envelope=True)
    assert len(env) > len(plain) and set(plain) <= set(env)
    # the stride ~25 % into the second segment (interpolated yaw ~132 deg) is also swept at the yaws of the last 0.25 m: the whole 5.8 cm
    # first segment (137 -> 146 deg) lies within - the demo's 141.9 deg is inside that envelope
    seg2 = [p for p in env if abs(p[0] - (23.1 - 0.3 * 0.2)) < 0.02 and abs(p[1] - (-0.7 + 0.2 * 0.2)) < 0.02]
    yaws = sorted(math.degrees(p[2]) for p in seg2)
    assert yaws and min(yaws) <= 136.5 and max(yaws) >= 141.9, yaws
    assert all(b - a <= 10.0 + 1e-6 for a, b in zip(yaws, yaws[1:])), yaws
    # the lag is capped at YAW_LAG_MAX_RAD (12 deg) from the interpolated yaw, whatever the look-back hull holds
    interp = [p for p in plain if abs(p[0] - seg2[0][0]) < 1e-9 and abs(p[1] - seg2[0][1]) < 1e-9][0][2]
    assert max(yaws) <= math.degrees(interp) + 12.0 + 1e-6 and min(yaws) >= math.degrees(interp) - 12.0 - 1e-6, (yaws, math.degrees(interp))


def test_stride_poses_bound_the_heading_step_between_checked_poses():
    """Owner 2026-10-13 ~04:30 item 3a (under#03 144 pitched c1_k1, 17836644 / 17838303: 'held:base_link in contact with desk_mdhelw_1'
    at navigate[5]): the recorded B -> C carry ends with a 12.7 cm segment turning 134 deg (0.80 -> pi), which the 10 cm strides
    checked at 46, 113 and 180 deg only; the base was at 137 deg at (18.455, -8.343) when the can swept into the desk. No two
    consecutive checked poses differ by more than YAW_STEP_MAX_RAD (3 deg) now - in the strides, the yaw-lag envelope and the
    face_travel poses - and less when the farthest robot / carried point would move more than 3 cm; a straight run is unchanged."""
    wps = [(18.3761, -8.6936, 1.9002), (18.3, -8.7, 1.249), (18.4, -8.4, 0.8008), (18.4884, -8.3088, -3.1416)]
    wrap = lambda a: (a + math.pi) % (2 * math.pi) - math.pi  # noqa: E731
    old, new = NAV.stride_poses(wps, 0.1, max_yaw_step=None), NAV.stride_poses(wps, 0.1)
    in_last = lambda ps: [p for p in ps if p[0] >= 18.4 - 1e-9 and p[1] >= -8.4 - 1e-9]  # noqa: E731
    assert [round(math.degrees(p[2])) for p in in_last(old)] == [46, 113, -180]
    assert len(in_last(new)) >= 45 and len(new) > len(old) and set(wps) <= set(new)
    assert max(abs(wrap(b[2] - a[2])) for a, b in zip(new[:-1], new[1:])) <= NAV.YAW_STEP_MAX_RAD + 1e-9
    fail = (18.455, -8.343, math.radians(137.0))

    def nearest(ps):
        return min(ps, key=lambda p: 100.0 * math.hypot(p[0] - fail[0], p[1] - fail[1]) + abs(math.degrees(wrap(p[2] - fail[2]))))

    p = nearest(new)
    assert math.hypot(p[0] - fail[0], p[1] - fail[1]) < 0.01 and abs(math.degrees(wrap(p[2] - fail[2]))) < 2.0, p
    assert abs(math.degrees(wrap(nearest(old)[2] - fail[2]))) > 20.0
    # a straight run (constant or slowly turning yaw) keeps the 10 cm strides
    straight = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    assert NAV.stride_poses(straight, 0.1) == NAV.stride_poses(straight, 0.1, max_yaw_step=None)
    assert len(NAV.stride_poses([(0.0, 0.0, 0.0), (1.0, 0.0, math.radians(20.0))], 0.1)) == 11
    assert NAV.stride_poses(straight, 0.1, yaw_envelope=True) == NAV.stride_poses(straight, 0.1)
    # the lag envelope follows: the plain strides plus, at every stride, its lag yaws <= 3 deg apart, <= 12 deg behind
    env = NAV.stride_poses(wps, 0.1, yaw_envelope=True)
    assert set(new) <= set(env) and len(env) > len(NAV.stride_poses(wps, 0.1, yaw_envelope=True, max_yaw_step=None))
    by_xy = {}
    for x, y, yaw in env:
        by_xy.setdefault((x, y), []).append(yaw)
    for (x, y), yaws in by_xy.items():
        ys = sorted(yaws)
        assert all(b - a <= NAV.YAW_STEP_MAX_RAD + 1e-9 for a, b in zip(ys, ys[1:])), ((x, y), ys)
        assert ys[-1] - ys[0] <= 2 * NAV.YAW_LAG_MAX_RAD + 1e-9
    # the farthest robot / carried point moves <= 3 cm per step: 0.73 m (the can) -> 2.35 deg; a smaller radius keeps 3 deg
    assert NAV.yaw_step_bound(None) == NAV.YAW_STEP_MAX_RAD == NAV.yaw_step_bound(0.33)
    assert NAV.yaw_step_bound(0.73) == pytest.approx(0.03 / 0.73)
    fine = NAV.stride_poses(wps, 0.1, max_yaw_step=NAV.yaw_step_bound(0.73))
    assert len(in_last(fine)) >= 57 and max(0.73 * abs(wrap(b[2] - a[2])) for a, b in zip(fine[:-1], fine[1:])) <= 0.03 + 1e-9
    # an on-the-spot turn (two waypoints at one spot) is swept, not checked at its ends alone
    spot = NAV.stride_poses([(0.0, 0.0, 0.0), (0.0, 0.0, math.radians(90.0))], 0.1)
    assert len(spot) == 31 and all(abs(b[2] - a[2]) <= NAV.YAW_STEP_MAX_RAD + 1e-9 for a, b in zip(spot[:-1], spot[1:]))
    # the face_travel schedule and its envelope: the same bound (the log-only probe of the leg: 50 swept poses before)
    ft = NAV.face_travel_poses(wps)
    assert len(NAV.yaw_lag_envelope(NAV.face_travel_poses(wps, max_yaw_step=None), max_yaw_step=None)) == 50
    assert max(abs(wrap(b[2] - a[2])) for a, b in zip(ft[:-1], ft[1:])) <= NAV.YAW_STEP_MAX_RAD + 1e-9
    assert len(ft) > len(NAV.face_travel_poses(wps, max_yaw_step=None)) and set(ft) <= set(NAV.yaw_lag_envelope(ft))
    # a sparse sequence handed to the envelope is filled in first
    sparse = [(0.0, 0.0, 0.0), (0.1, 0.0, math.radians(30.0))]
    filled = NAV.bound_yaw_steps(sparse)
    assert len(filled) == 11 and filled[0] == sparse[0] and filled[-1] == sparse[-1]
    assert set(filled) <= set(NAV.yaw_lag_envelope(sparse))


def test_sweep_radius_is_the_farthest_collision_sphere_and_plan_path_checks_with_its_bound(monkeypatch):
    """Owner 2026-10-13 item 3a: plan_path takes the heading step from the farthest collision sphere of the navigation posture
    (the robot's and the attached object's, ``robot_spheres_world``) - 3 cm of travel per step at most - and the angular bound
    alone when the backend has no sphere model."""
    from types import SimpleNamespace

    spheres = th.tensor([[[0.2, 0.1, 0.5, 0.05], [0.0, -0.9, 0.7, 0.1], [3.0, 3.0, 0.0, -1.0]]])  # x y z r; r <= 0 = disabled
    kit = SimpleNamespace(q_config=lambda **kw: th.zeros(3), q_at_base_pose=lambda q, x, y, yaw: th.tensor([x, y, yaw]),
                          backend=SimpleNamespace(robot_spheres_world=lambda q, emb_name="whole_body": (spheres, ["a", "b", "c"])))
    assert NAV.sweep_radius(kit, (0.5, 0.0)) == pytest.approx(1.0)
    assert NAV.sweep_radius(SimpleNamespace(q_config=kit.q_config, q_at_base_pose=kit.q_at_base_pose, backend=object()), (0.5, 0.0)) is None
    monkeypatch.setenv("ISG_ATTACHED_CHECKS", "off")
    monkeypatch.setattr(NAV, "local_map", lambda kit, rooms=None, blocked=(): th.ones(30, 30, dtype=th.bool))
    monkeypatch.setattr(NAV, "_to_map", lambda kit, xy: (int(round(float(xy[1]) / 0.05)), int(round(float(xy[0]) / 0.05))))
    monkeypatch.setattr(NAV, "_to_world", lambda kit, ij: th.tensor([ij[1] * 0.05, ij[0] * 0.05]))
    monkeypatch.setattr(NAV, "nearest_free_cell", lambda free, ij, r=15: ij)
    monkeypatch.setattr(NAV, "same_region", lambda free, a, b: True)
    monkeypatch.setattr(NAV, "grid_shortest_path", lambda free, a, b: [(0, 0), (0, 20), (20, 20)])
    monkeypatch.setattr(NAV, "simplify_path", lambda cells, free: list(cells))
    seen = []
    monkeypatch.setattr(NAV, "path_collision_free", lambda kit, poses, trunk, posture="untucked", margin=0.01, q_arms=None, fingers=None: (seen.append(list(poses)), (True, -1))[1])
    st = lambda name, x, y, yaw: SimpleNamespace(name=name, base_xy=th.tensor([x, y]), yaw=yaw, base_pose2d=(x, y, yaw), translate=0.0, tilt=0.0)  # noqa: E731
    a, b = st("init", 0.0, 0.0, 0.0), st("manipulation", 1.0, 1.0, 1.5708)
    rec = NAV.plan_path(kit, a, b, arm_q={})
    swept = seen[-1]
    assert swept == NAV.stride_poses(rec.waypoints, 0.1, yaw_envelope=True, max_yaw_step=0.03)
    assert rec.n_checked == len(NAV.stride_poses(rec.waypoints, 0.1, max_yaw_step=0.03)) > len(NAV.stride_poses(rec.waypoints, 0.1))
    seen.clear()
    NAV.plan_path(None, a, b, arm_q={})  # no sphere model: 3 deg
    assert seen[-1] == NAV.stride_poses(rec.waypoints, 0.1, yaw_envelope=True)


def test_plan_path_face_travel_style_records_the_schedule_or_falls_back_to_recorded(monkeypatch):
    """Owner 2026-09-26 (item 6): ``style="face_travel"`` checks the shared base_path schedule (with the yaw-lag envelope, same
    posture / margin) and records ``face_travel`` when it clears; when a stride of the schedule collides the polyline check
    decides and the record says ``recorded`` - never a rejection because of the schedule. A carried leg stays recorded and
    its schedule is only probed at 1 cm (log + tally)."""
    from types import SimpleNamespace

    monkeypatch.setenv("ISG_ATTACHED_CHECKS", "off")
    # an L-shaped grid path (0,0) -> (0,20) -> (20,20) on a 5 cm grid: a 90 deg corner at (1.0, 0.0)
    monkeypatch.setattr(NAV, "local_map", lambda kit, rooms=None, blocked=(): th.ones(30, 30, dtype=th.bool))
    monkeypatch.setattr(NAV, "_to_map", lambda kit, xy: (int(round(float(xy[1]) / 0.05)), int(round(float(xy[0]) / 0.05))))
    monkeypatch.setattr(NAV, "_to_world", lambda kit, ij: th.tensor([ij[1] * 0.05, ij[0] * 0.05]))
    monkeypatch.setattr(NAV, "nearest_free_cell", lambda free, ij, r=15: ij)
    monkeypatch.setattr(NAV, "same_region", lambda free, a, b: True)
    monkeypatch.setattr(NAV, "grid_shortest_path", lambda free, a, b: [(0, 0), (0, 20), (20, 20)])
    monkeypatch.setattr(NAV, "simplify_path", lambda cells, free: list(cells))
    calls = []
    verdicts = {"face_travel": True}

    def fake_free(kit, poses, trunk, posture="untucked", margin=0.01, q_arms=None, fingers=None):
        n = len(calls)
        calls.append((len(poses), margin))
        ok = verdicts["face_travel"] if n == 0 else True
        return ok, (0 if ok else 3)

    monkeypatch.setattr(NAV, "path_collision_free", fake_free)
    st = lambda name, x, y, yaw: SimpleNamespace(name=name, base_xy=th.tensor([x, y]), yaw=yaw, base_pose2d=(x, y, yaw), translate=0.0, tilt=0.0)  # noqa: E731
    a, b = st("init", 0.0, 0.0, 0.0), st("manipulation", 1.0, 1.0, 1.5708)
    rec = NAV.plan_path(None, a, b, arm_q={}, style="face_travel")
    assert rec.style == "face_travel" and len(calls) == 1 and rec.n_checked > 0
    assert rec.to_sidecar()["style"] == "face_travel"
    # the schedule collides: the polyline decides and the record says "recorded"
    calls.clear()
    verdicts["face_travel"] = False
    rec = NAV.plan_path(None, a, b, arm_q={}, style="face_travel")
    assert rec.style == "recorded" and len(calls) == 2 and rec.to_sidecar()["style"] == "recorded"
    # the default style is untouched: no style key, one (polyline) check
    calls.clear()
    verdicts["face_travel"] = True
    rec = NAV.plan_path(None, a, b, arm_q={})
    assert rec.style is None and "style" not in rec.to_sidecar() and len(calls) == 1
    # a carried leg stays recorded; the schedule is probed at 1 cm after the polyline check, tallied, never used
    calls.clear()
    before = dict(NAV.FACE_TRAVEL_CARRIED_PROBE)
    verdicts["face_travel"] = True
    rec = NAV.plan_path(None, a, b, arm_q={}, carried="sandal.n.01_1", carried_obj=object(), style="recorded", probe_face_travel=True, margin=0.03)
    assert rec.style is None and len(calls) == 2 and calls[0][1] == 0.03 and calls[1][1] == 0.01
    assert NAV.FACE_TRAVEL_CARRIED_PROBE["checked"] == before["checked"] + 1
    # the schedule's poses carry the yaw-lag envelope like the polyline's
    ft = NAV.face_travel_poses(rec.waypoints)
    env = NAV.yaw_lag_envelope(ft)
    assert len(env) >= len(ft) and set(ft) <= set(env)


def test_stride_poses_and_grid_path():
    poses = NAV.stride_poses([(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], 0.1)
    assert len(poses) == 11 and poses[-1] == (1.0, 0.0, 0.0)
    free = th.ones(10, 10, dtype=th.bool)
    free[:, 5] = False
    free[0, 5] = True
    path = NAV.grid_shortest_path(free, (9, 0), (9, 9))
    assert path is not None and path[0] == (9, 0) and path[-1] == (9, 9) and (0, 5) in path
    assert NAV.same_region(free, (9, 0), (9, 9))
    free[0, 5] = False
    assert NAV.grid_shortest_path(free, (9, 0), (9, 9)) is None
    assert not NAV.same_region(free, (9, 0), (9, 9))
    simple = NAV.simplify_path([(0, 0), (1, 1), (2, 2), (3, 3)], th.ones(4, 4, dtype=th.bool))
    assert simple == [(0, 0), (3, 3)]


def test_path_record_schema():
    p = PathRecord("init", "manipulation", "local", [(0.0, 0.0, 0.0), (0.3, 0.4, 0.1)], path_length([(0, 0), (0.3, 0.4)]), "untucked", None, (0.5, 0.0))
    d = p.to_sidecar()
    assert PathRecord.schema_ok(d) and d["length_m"] == pytest.approx(0.5) and d["carried"] is None and d["mode"] == "local"
    assert set(d) == {"from", "to", "mode", "waypoints", "length_m", "posture", "carried"}
    assert not PathRecord.schema_ok({**d, "mode": "teleport"})
    assert not PathRecord.schema_ok({k: v for k, v in d.items() if k != "carried"})
    assert PathRecord.schema_ok({**d, "mode": "edge_slide"})  # owner 2026-09-24: the same-edge slide is a third planning mode


def test_edge_slide_waypoints():
    """Owner 2026-09-24 (disembedding carry): two stances at the same edge are joined by a slide with the body facing the
    edge - every waypoint after the first carries the arriving yaw; equal stand-offs give a straight two-point slide, a
    stand-off difference an L (drive back to the larger stand-off, slide, drive in); ``back_m`` adds clearance; a
    different edge, a different support or a turned body is not an edge slide."""
    # edge "-u" of a table: midpoint (5, 3), inward +y (the robot stands at y < 3 facing +y = yaw 90 deg)
    edge = EdgeFrame("-u", th.tensor([5.0, 3.0]), th.tensor([0.0, 1.0]), 2.0)
    yaw = math.pi / 2
    manip = Stance("manipulation", (4.6, 3.0 - 0.45, yaw), 0.25, 0.0, stand_off=0.45, edge=edge, lateral=-0.4)
    place = Stance("placement", (5.6, 3.0 - 0.45, yaw), 0.25, 0.0, stand_off=0.45, edge=edge, lateral=0.6)
    assert NAV.same_edge(manip, place)
    wps = NAV.edge_slide_waypoints(manip, place)
    assert len(wps) == 2 and wps[0] == pytest.approx((4.6, 2.55, yaw)) and wps[1] == pytest.approx((5.6, 2.55, yaw))
    assert all(w[2] == pytest.approx(yaw) for w in wps)  # no heading change anywhere: the tracker slides
    assert path_length(wps) == pytest.approx(1.0)
    # the stride poses keep the yaw too (what the whole-body check validates = what the oracle drives)
    assert all(p[2] == pytest.approx(yaw) for p in NAV.stride_poses(wps, 0.1))
    # a deeper placement stance: back to the larger stand-off, slide, drive in
    place2 = Stance("placement", (5.6, 3.0 - 0.60, yaw), 0.25, 0.0, stand_off=0.60, edge=edge, lateral=0.6)
    wps2 = NAV.edge_slide_waypoints(manip, place2)
    assert len(wps2) == 3 and wps2[1] == pytest.approx((4.6, 2.40, yaw)) and wps2[2] == pytest.approx((5.6, 2.40, yaw))
    manip2 = Stance("manipulation", (4.6, 3.0 - 0.60, yaw), 0.25, 0.0, stand_off=0.60, edge=edge, lateral=-0.4)
    wps3 = NAV.edge_slide_waypoints(manip2, place)
    assert len(wps3) == 3 and wps3[1] == pytest.approx((5.6, 2.40, yaw)) and wps3[2] == pytest.approx((5.6, 2.55, yaw))
    # extra clearance for the slide
    wps4 = NAV.edge_slide_waypoints(manip, place, back_m=0.1)
    assert len(wps4) == 4 and wps4[1] == pytest.approx((4.6, 2.45, yaw)) and wps4[2] == pytest.approx((5.6, 2.45, yaw)) and wps4[3] == pytest.approx((5.6, 2.55, yaw))
    # not the same edge: another edge name, another support (midpoint), or a body that does not face the edge
    other = EdgeFrame("+v", th.tensor([6.0, 4.0]), th.tensor([-1.0, 0.0]), 1.0)
    assert not NAV.same_edge(manip, Stance("placement", (6.45, 4.0, math.pi), 0.25, 0.0, stand_off=0.45, edge=other))
    far = EdgeFrame("-u", th.tensor([9.0, 3.0]), th.tensor([0.0, 1.0]), 2.0)
    assert not NAV.same_edge(manip, Stance("placement", (9.0, 2.55, yaw), 0.25, 0.0, stand_off=0.45, edge=far))
    assert not NAV.same_edge(manip, Stance("placement", (5.6, 2.55, yaw + math.radians(10)), 0.25, 0.0, stand_off=0.45, edge=edge))
    assert not NAV.same_edge(manip, Stance("init", (5.6, 2.55, yaw), 0.25, 0.0, stand_off=0.45))  # no edge frame


# ----------------------------------------------------------------------------------------------------------------
# stances / sidecar (§7)
# ----------------------------------------------------------------------------------------------------------------


def test_edge_frame_and_stance_local_world_roundtrip():
    edge = EdgeFrame.from_edge("+u", mid=[1.0, 0.0], outward_normal=[1.0, 0.0], length=1.6, overhang=0.0)
    assert th.allclose(edge.inward, th.tensor([-1.0, 0.0])) and th.allclose(edge.tangent, th.tensor([0.0, -1.0]))
    xy = edge.base_xy(0.45, 0.2)
    st = Stance("init", (float(xy[0]), float(xy[1]), edge.yaw), 0.5, 0.0, 0.45, edge, lateral=0.2)
    assert st.edge_depth == pytest.approx(0.45)
    depth, lat = st.local(edge.mid + edge.inward * 0.3 + edge.tangent * 0.1)
    assert depth == pytest.approx(0.3) and lat == pytest.approx(0.1 - 0.2)
    back = st.world(depth, lat)
    assert th.allclose(back, edge.mid + edge.inward * 0.3 + edge.tangent * 0.1, atol=1e-6)
    manip = st.shifted(stand_off=0.35, name="manipulation").with_trunk(0.75, -0.4, "manipulation")
    assert manip.stand_off == 0.35 and manip.translate == 0.75 and manip.lateral == 0.2
    assert manip.to_sidecar()["edge"] == "+u" and manip.to_sidecar()["tilt"] == -0.4


def test_sidecar_draft_layout(tmp_path):
    sd = SidecarDraft("alias-x", "inter_object_relation_satisfaction")
    edge = EdgeFrame.from_edge("+u", [1.0, 0.0], [1.0, 0.0], 1.6)
    init = Stance("init", (1.45, 0.0, math.pi), 0.5, 0.0, 0.45, edge)
    manip = Stance("manipulation", (1.35, 0.0, math.pi), 0.75, -0.4, 0.35, edge, focus=th.tensor([0.7, 0.0, 0.83]), arms=["right"])
    sd.add_stance(init)
    sd.add_stance(manip)
    sd.add_transition(Transition("init", "manipulation", 0.55))
    pose = (th.tensor([0.7, 0.0, 0.95]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    sd.add_grasp(GraspRecord("apple.n.01_1", "right", "apple__a__t", "grasp_1", 0.3, pose, pose, pose, pose))
    sd.set_goal(manip.base_pose2d, 0.75, -0.4)
    d = sd.to_dict()
    assert d["schema"] == "isg-oracle-v1" and set(d["stances"]) == {"init", "manipulation"}
    assert d["stances"]["manipulation"]["focus"] == pytest.approx([0.7, 0.0, 0.83]) and d["stances"]["init"]["arms"] == "untucked"
    assert d["grasps"][0]["grasp"]["position"] == pytest.approx([0.7, 0.0, 0.95])
    assert d["waypoint_policy"] == "checked" and "contacts" not in d and "paths" not in d
    assert PC.check_blocks("inter_object_relation_satisfaction-ontop_small_scale", d) == []
    problems = PC.check_blocks("inter_object_relation_satisfaction-touching", d)
    # touching: the push is required, the brace optional (owner 2026-09-16 17:10, the axial rule)
    assert any("contacts.push" in p for p in problems) and not any("contacts.brace" in p for p in problems)
    tro = tmp_path / "x-tro_state.json"
    tro.write_text("{}")
    out = sd.finalize("iid", str(tro))
    assert out["tro_sha256"] == "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a" and out["instance_id"] == "iid"
    sd.reset_instance()
    assert sd.stances == {} and sd.grasps == [] and sd.tro_sha256 is None


def test_program_check_target_extraction():
    pose = {"position": [0.0, 0.0, 0.0], "orientation": [0.0, 0.0, 0.0, 1.0]}
    targets = {"pre_grasp": pose, "approach": [pose, pose, pose], "base_pose2d": [0.0, 0.0, 0.0], "nested": {"place": pose}}
    found = dict(PC.iter_pose_targets(targets))
    assert set(found) == {"pre_grasp", "approach", "nested.place"} and len(found["approach"]) == 3


def test_program_check_held_start_leaves_from_the_recorded_configuration():
    """A held-object init (the object in the hand since the init stance, the arm holding it not moved yet): the in-hand
    chain is seeded with the stance's recorded configuration, as the sampler's draw check does; every other case keeps
    the fresh solve - among them an object carried to a place stance that records it (iors / attachment), whose arm-end
    record a base motion cleared."""
    from types import SimpleNamespace

    q_hold = th.arange(28, dtype=th.float32)
    sampler = SimpleNamespace(ikc=SimpleNamespace(q_locked=lambda stance, fingers=None: q_hold))
    init, grasp, held = Stance("init", (0.0, 0.0, 0.0), 0.5, arms=["right"], carried="banana.n.02_1"), SimpleNamespace(arm="right"), "banana.n.02_1"
    q = PC._held_start_q(sampler, init, grasp, held, "right", None, True, None, None, None)
    assert q is not None and th.equal(q, q_hold) and q is not q_hold
    moved = th.zeros(28)
    place = Stance("place", (1.0, 0.0, 0.0), 0.5, arms=["right"], carried=held)
    not_held = [
        (place, grasp, held, "right", False, None, None, None),  # grasped during the program, carried to its place stance
        (init, grasp, None, "right", True, None, None, None),  # nothing carried
        (init, grasp, held, "left", True, None, None, None),  # the other arm acts
        (init, None, held, "right", True, None, None, None),  # no grasp record
        (Stance("manip", (0.0, 0.0, 0.0), 0.5), grasp, held, "right", True, None, None, None),  # the stance does not record it
        (init, grasp, held, "right", True, SimpleNamespace(), None, None),  # the base moved (a follow's stance)
        (init, grasp, held, "right", True, None, moved, None),  # seeded from a previous action's end
        (init, grasp, held, "right", True, None, None, moved),  # the arm already moved
    ]
    for stance, g, carried, arm, since_init, stance_obj, prev, end in not_held:
        assert PC._held_start_q(sampler, stance, g, carried, arm, None, since_init, stance_obj, prev, end) is None


def test_loaded_stance_keeps_the_recorded_arms():
    """``validate_instances`` rebuilds stances from the sidecar: the arm joint values the sampler's checks used (``arms_q``),
    the fingers and the idle posture come back, and ``stance_arm_vectors`` uses the recorded arms instead of the default
    postures (a held-object init: the hold and the reset left arm, not both arms untucked - mental_rotation-banana 0002's
    untucked left arm reached into the gas fireplace)."""
    from types import SimpleNamespace

    from ispatialgym.sampling.v2.ik import stance_arm_vectors

    hold = [-1.3561, -0.0534, 0.5577, -0.5514, -0.0924, -1.044, 1.0925]
    d = {"base_pose2d": [17.1, -0.7, -1.093], "translate": 0.5, "tilt": 0.0, "arms": ["right"], "carried": "banana.n.02_1", "idle_posture": "untucked",
         "arms_q": {"right": hold, "left": [0.0] * 7}, "fingers": 0.0228, "hold": {"depth_m": 0.5}}
    st = Stance.from_sidecar("init", d)
    assert st.idle_posture == "untucked" and abs(st.fingers - 0.0228) < 1e-9 and st.carried == "banana.n.02_1"
    assert st.extras["arms_q"]["right"] == hold and st.extras["hold"] == {"depth_m": 0.5} and "base_pose2d" not in st.extras and "idle_posture" not in st.extras
    untucked = th.full((7,), 0.5)
    kit = SimpleNamespace(robot=SimpleNamespace(arm_control_idx={"right": th.arange(7), "left": th.arange(7, 14)}, gripper_control_idx={"right": th.arange(14, 16)}),
                          arm_q=lambda name, arm: untucked.clone())
    r, l, f = stance_arm_vectors(kit, st)
    assert th.allclose(r, th.tensor(hold)) and th.equal(l, th.zeros(7)) and abs(f - 0.0228) < 1e-9
    bare = Stance.from_sidecar("init", {k: v for k, v in d.items() if k != "arms_q"})  # no record: the default postures
    r, l, _ = stance_arm_vectors(kit, bare)
    assert th.equal(r, untucked) and th.equal(l, untucked)


def test_program_check_held_start_heads_the_chain():
    """The hand's recorded start pose heads a held-object init's chain when the action's waypoints leave it out (a
    reorient's orbit starts a quarter turn in), so the chain is checked as the draw check's ``[hold] + waypoints``."""
    from types import SimpleNamespace

    start = _pose(0.3)
    reaches = lambda q, pose, arm: float(th.linalg.norm(pose[0] - start[0])) < 1e-6  # noqa: E731 - FK of q_held is `start`
    sampler = SimpleNamespace(ikc=SimpleNamespace(reaches=reaches), kit=SimpleNamespace(eef_pose=lambda q, arm="right": start))
    q_held = th.zeros(28)
    orbit = [_pose(0.4), _pose(0.5), _pose(0.6)]
    chain, prepended = PC._with_held_start(orbit, q_held, sampler, "right")
    assert prepended and len(chain) == 4 and chain[0] is start and chain[1:] == orbit
    chain, prepended = PC._with_held_start([start] + orbit, q_held, sampler, "right")  # already leaves from the start
    assert not prepended and len(chain) == 4
    chain, prepended = PC._with_held_start(orbit, None, sampler, "right")  # not a held start
    assert not prepended and chain == orbit


def test_program_check_tracker_handle_grasp_carries_nothing():
    """Large articulation (Rs_int drawer bamfsz_1): a kind="handle" grasp must not set ``carried`` - the cabinet stays in the
    world, only its handle link follows the hand; library / cube grasps carry their object until the place."""
    sd = SidecarDraft("alias", "configuration_matching")
    sd.add_stance(Stance("init", (0.0, 0.0, 0.0), 0.5))
    tr = PC._Tracker(sd)

    class A:
        def __init__(self, name, **kw):
            self.name = name
            self.__dict__.update(kw)

    tr.after(A("grasp", object="cabinet.n.01_1", kind="handle"))
    assert tr.carried is None
    tr.after(A("grasp", object="apple.n.01_1", kind="library"))
    assert tr.carried == "apple.n.01_1"
    tr.after(A("place", object="apple.n.01_1"))
    assert tr.carried is None
    tr.after(A("grasp", object="block.n.01_1", kind="cube"))
    assert tr.carried == "block.n.01_1"


def test_program_check_tracker_locks_the_brace_arm_until_its_retreat():
    """Owner 2026‑09‑16 22:55 item 3 (wall horizontal 0916‑212037: push witnesses solved with the brace arm at zeros ran the
    push arm through the braced arm): after a ``brace`` the bracing arm is locked at its contact witness for every action
    of the other arm (push, the push arm's retreat); its own retreat releases it; the acting arm is never locked."""
    sd = SidecarDraft("alias", "alignment")
    sd.add_stance(Stance("init", (0.0, 0.0, 0.0), 0.5))
    tr = PC._Tracker(sd)

    class A:
        def __init__(self, name, arms, **kw):
            self.name = name
            self.arms = tuple(arms)
            self.__dict__.update(kw)

    class Robot:
        arm_control_idx = {"left": th.tensor([10, 12, 14, 16, 18, 20, 22]), "right": th.tensor([11, 13, 15, 17, 19, 21, 23])}

    class Kit:
        robot = Robot()

    class Sampler:
        kit = Kit()

    q = [0.0] * 28
    for i, v in zip(Robot.arm_control_idx["right"].tolist(), [0.1, -2.5, -0.5, -1.6, 2.2, 0.6, -0.05]):
        q[i] = v
    brace, push = A("brace", ["right"]), A("push", ["left"])
    assert tr.locked_arms(brace) == {} and tr.locked_arms(push) == {}
    tr.after(brace, {"pre": {"q": [0.0] * 28}, "contact": {"q": q}, "approach": {"q": q}}, Sampler())
    locked = tr.locked_arms(push)
    assert list(locked) == ["right"] and th.allclose(locked["right"], th.tensor([0.1, -2.5, -0.5, -1.6, 2.2, 0.6, -0.05]))
    assert tr.locked_arms(A("retreat", ["right"])) == {}  # the bracing arm's own actions are never locked against itself
    tr.after(push, {"pre": {"q": [0.0] * 28}}, Sampler())
    assert list(tr.locked_arms(A("retreat", ["left"]))) == ["right"]  # the push arm retreats while the brace still holds
    tr.after(A("retreat", ["left"]), {"retreat": {"q": [0.0] * 28}}, Sampler())
    assert list(tr.locked_arms(A("retreat", ["right"], release=True))) == []
    tr.after(A("retreat", ["right"], release=True), {"retreat": {"q": [0.0] * 28}}, Sampler())
    assert tr.arm_q == {}


def test_program_check_ignore_sets_mirror_the_sampler():
    """Large-scale A4/A5/A8: a brace approach runs along the wall -> the supports are removed for every waypoint (as for a
    push), the braced object only at the contact; explicit ``ignore_every`` / ``ignore_last`` on the record win."""

    class Obj:
        def __init__(self, name):
            self.name = name

    wall, target, partner = Obj("wall"), Obj("painting"), Obj("can")

    class Sampler:
        objects = {"painting": target, "can": partner}

        @staticmethod
        def support_objects():
            return [wall]

    class Contact:
        def __init__(self, obj, extra=None):
            self.object = obj
            self.extra = extra or {}

    class Action:
        def __init__(self, name, contact):
            self.name, self.contact = name, contact

    last, every = PC._ignore_for(Action("brace", Contact("painting")), Sampler, None)
    assert every == [wall] and last == [target]
    last, every = PC._ignore_for(Action("push", Contact("painting")), Sampler, None)
    assert target in every and wall in every and last == []
    # a module that recorded what it ignored while solving (names or the token "supports")
    last, every = PC._ignore_for(Action("brace", Contact("painting", {"ignore_every": ["supports", "can"], "ignore_last": []})), Sampler, None)
    assert every == [wall, partner] and last == []


def test_program_check_ignores_the_objects_a_grasp_may_touch():
    """Owner 2026-10-12 (A2b): disembedding's grasp record names the distractors / fixed-layout parts its grasp may touch
    (``grasps[].may_touch``): out of the world with the supports for the grasp pose and along the lift (the fingers start among
    them); an instance the sampler does not know is skipped and a record without the key changes nothing."""
    from types import SimpleNamespace

    table, tray, box = SimpleNamespace(name="table"), SimpleNamespace(name="tray"), SimpleNamespace(name="box")

    class Sampler:
        objects = {"bottle": SimpleNamespace(name="bottle"), "box": box, "tray": tray}

        @staticmethod
        def support_objects():
            return [table, tray]

    rec = SimpleNamespace(extra={"may_touch": {"objects": ["box", "ghost"], "stay_on": "tray"}})
    grasp = SimpleNamespace(name="grasp", object="bottle", grasp=rec)
    assert PC._ignore_for(grasp, Sampler, None) == ([table, tray, box], [])
    assert PC._ignore_for(SimpleNamespace(name="lift", object="bottle", after=grasp), Sampler, None) == ([], [table, tray, box])
    plain = SimpleNamespace(name="grasp", object="bottle", grasp=SimpleNamespace(extra={}))
    assert PC._ignore_for(plain, Sampler, None) == ([table, tray], [])
    assert PC._ignore_for(SimpleNamespace(name="lift", object="bottle", after=plain), Sampler, None) == ([], [table, tray])
    # the fingers may meet them on the way down: removed along the whole approach (the pre-grasp itself keeps them) - no spacing keeps
    # them out of the finger path any more
    bottle = Sampler.objects["bottle"]
    assert PC._approached_for(grasp, Sampler) == [bottle, box] and PC._approached_for(plain, Sampler) == [bottle]
    # the puzzle's fixed-layout parts (``may_touch.pre_grasp``): out of the world for the pre-grasp too, i.e. for every pose of the pick
    puzzle = SimpleNamespace(name="grasp", object="bottle", grasp=SimpleNamespace(extra={"may_touch": {"objects": ["box"], "stay_on": "tray", "pre_grasp": ["box"]}}))
    assert PC._ignore_for(puzzle, Sampler, None) == ([table, tray, box], [box])


def test_per_role_scale_default_and_template_id_tag():
    """Owner item 4b: ``roles.<role>.scale = {"<cat>/<model>": s, "default": s}`` (or a bare scalar) applies to every model
    of the role at spawn; the template id carries it (``-s1.25``) so scaled and unscaled draws never share a directory."""
    from ispatialgym.sampling.spec import ModelChoice
    from ispatialgym.sampling.v2.problem import TemplateChoice, resolve_scale, scale_tag

    assert resolve_scale({"tray/gsxbym": 1.25, "default": 1.25}, "tray", "other") == [1.25, 1.25, 1.25]
    assert resolve_scale({"tray/gsxbym": [1.0, 1.2, 1.0], "default": 1.25}, "tray", "gsxbym") == [1.0, 1.2, 1.0]
    assert resolve_scale(1.25, "tray", "gsxbym") == [1.25, 1.25, 1.25] and resolve_scale(None, "tray", "gsxbym") is None
    assert resolve_scale({"bag_of_chips/dwkdko": 0.7}, "bag_of_chips", "zzzzzz") is None  # no default: unscaled
    assert scale_tag(None) == "" and scale_tag([1.0, 1.0, 1.0]) == "" and scale_tag([1.25, 1.25, 1.25]) == "-s1.25" and scale_tag([1.0, 1.2, 1.0]) == "-s1x1.2x1"
    scaled = TemplateChoice("house_single_floor", {"source": ModelChoice("tray", "gsxbym", scale=[1.25, 1.25, 1.25]), "target": ModelChoice("bar_soap", "abc")}, {}, {}, [])
    plain = TemplateChoice("house_single_floor", {"source": ModelChoice("tray", "gsxbym"), "target": ModelChoice("bar_soap", "abc")}, {}, {}, [])
    assert scaled.template_id == "house_single_floor__source-tray-gsxbym-s1.25__target-bar_soap-abc" and plain.template_id == "house_single_floor__source-tray-gsxbym__target-bar_soap-abc"
    assert TemplateChoice.from_json(scaled.to_json()).template_id == scaled.template_id


def test_open_jaw_footprint_clearance_and_release_opening():
    """Owner 2026-09-16 20:29 (nextto re-run: the opening finger swept the pencil 9 cm): the partner must lie outside the
    swept open-jaw footprint at the place; the largest release opening that clears it by 1 cm is a fact for the oracle."""
    from ispatialgym.robot import conventions as C
    from ispatialgym.sampling.v2.tasks import iors_small as M

    cls = [c for c in vars(M).values() if isinstance(c, type) and hasattr(c, "open_jaw_clearance")][0]
    table = 0.8126
    q = C.eef_quat(th.tensor([0.0, 0.0, -1.0]), th.tensor([0.0, 1.0, 0.0]))  # top-down pinch, jaw opening along +-y
    place = (th.tensor([0.0, 0.0, table + C.FINGERTIP_LENGTH + 0.004]), q)  # fingertips 4 mm above the table
    pencil = th.tensor([[x, 0.03, table + z] for x in (-0.09, 0.0, 0.09) for z in (0.0, 0.007)])  # 3 cm off along the jaw
    assert cls.open_jaw_clearance(None, place, pencil) < 0
    assert cls.open_jaw_opening_max(None, place, pencil) < 0.017 + 0.01  # even the least releasing opening sweeps it
    far = pencil + th.tensor([0.0, 0.06, 0.0])  # 9 cm off: outside the 7.2 cm half jaw
    assert cls.open_jaw_clearance(None, place, far) > 0 and cls.open_jaw_opening_max(None, place, far) == pytest.approx(C.GRIPPER_OPEN_M)
    q_turned = C.eef_quat(th.tensor([0.0, 0.0, -1.0]), th.tensor([1.0, 0.0, 0.0]))  # jaw turned 90 deg away from the partner
    assert cls.open_jaw_clearance(None, (place[0], q_turned), pencil) > 0


def test_robust_under_ontop_wrapper_branches(monkeypatch):
    """D-007 option A: OmniGibson's verdict, then the contact-patch rays, then the geometric branch; the branch is logged."""
    import omnigibson.utils.bddl_utils as BU
    from ispatialgym.sampling import predicates as PRED

    class Obj:
        def __init__(self, name):
            self.name, self.links, self.initialized = name, {}, True

        def get_position_orientation(self):
            return th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])

    class Under:  # stands in for bddl's predicate class (only the name is used by the wrapper)
        pass

    class NextTo:
        pass

    saucer, cup = Obj("saucer"), Obj("cup")
    monkeypatch.setattr(BU, "evaluate_bddl_predicate", lambda cls, *ents: False)
    monkeypatch.setattr(PRED, "support_rays", lambda lower, upper: {"ok": True, "n_hits_on_upper": 3})
    PRED.reset_branches()
    assert PRED.evaluate_predicate(Under, saucer, cup) is True
    assert PRED.branches()[-1]["branch"] == "rays" and PRED.decided_by() == {"under(saucer, cup)": "rays"}
    monkeypatch.setattr(PRED, "support_rays", lambda lower, upper: {"ok": False})
    monkeypatch.setattr(PRED, "support_geometry", lambda lower, upper: {"ok": True, "bottom_gap_mm": 1.0})
    assert PRED.evaluate_predicate(Under, saucer, cup) is True and PRED.branches()[-1]["branch"] == "geometry"
    monkeypatch.setattr(PRED, "support_geometry", lambda lower, upper: {"ok": False})
    assert PRED.evaluate_predicate(Under, saucer, cup) is False and PRED.branches()[-1]["branch"] == "none"
    # ontop(cup, saucer): the saucer supports the cup - the same branches with the roles swapped
    seen = {}
    monkeypatch.setattr(PRED, "support_rays", lambda lower, upper: seen.update(lower=lower.name, upper=upper.name) or {"ok": True})
    assert PRED.evaluate_predicate(type("OnTop", (), {}), cup, saucer) is True and seen == {"lower": "saucer", "upper": "cup"}
    # other predicates pass straight through (no re-check), OmniGibson True ends the evaluation
    monkeypatch.setattr(BU, "evaluate_bddl_predicate", lambda cls, *ents: True)
    assert PRED.evaluate_predicate(NextTo, cup, saucer) is True and PRED.branches()[-1]["branch"] == "omnigibson"


def test_pose_settled_of_the_robot_is_judged_by_configuration_drift(monkeypatch):
    """Owner 2026-09-24: PoseSettled's 1 mm/s root-velocity gate never holds for the R1Pro at rest (the wheel-contact solver
    residual, pod probe), so after an OmniGibson False the robot's ``pose_settled`` is decided by the drift of its PoseSettled
    window - the rule the attached objects already had. A plain object keeps the stock rule unless ``DRIFT_SETTLED_ENTITIES``
    says ``all``; the deciding branch and the entity class are logged."""
    import omnigibson.utils.bddl_utils as BU
    from omnigibson.object_states import AttachedTo, PoseSettled
    from ispatialgym.sampling import predicates as PRED

    class Settled:  # the PoseSettled window as the wrapper reads it
        def __init__(self, drift_pos=0.0, drift_joint=0.0, n=10, joints=28):
            self.is_active, self.n_steps = True, n
            self._pos_tolerance, self._ori_tolerance, self._joint_tolerance = 0.01, 0.05, 0.05
            q = th.zeros(joints)
            self._history = [(t, th.tensor([drift_pos * t / (n - 1), 0.0, 0.0]), th.tensor([0.0, 0.0, 0.0, 1.0]), q + drift_joint * t / (n - 1), False) for t in range(n)]

    class Obj:
        def __init__(self, name, states, robot=False):
            self.name, self.links, self.states = name, {}, states
            if robot:
                self.controller_action_idx = {"base": th.arange(3)}  # what marks a robot for the wrapper without a simulator

        def get_position_orientation(self):
            return th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])

    class pose_settled:  # bddl's predicate class stand-in (the wrapper keys on the class name)
        pass

    monkeypatch.setattr(BU, "evaluate_bddl_predicate", lambda cls, *ents: False)  # the velocity gate fails, as it does for the robot at rest
    robot = Obj("robot_r1", {PoseSettled: Settled()}, robot=True)
    PRED.reset_branches()
    assert PRED.evaluate_predicate(pose_settled, robot) is True
    last = PRED.branches()[-1]
    assert last["branch"] == "drift" and last["detail"]["entity"] == "robot" and last["detail"]["velocity_still_steps"] == 0
    assert PRED.decided_by() == {"pose_settled(robot_r1)": "drift"}
    # a robot whose joints drift beyond the window tolerance is not settled (a creeping joint), and the window must be full
    assert PRED.evaluate_predicate(pose_settled, Obj("robot_r1", {PoseSettled: Settled(drift_joint=0.1)}, robot=True)) is False
    assert PRED.branches()[-1]["branch"] == "none"
    short = Settled()
    short._history = short._history[:5]
    assert PRED.evaluate_predicate(pose_settled, Obj("robot_r1", {PoseSettled: short}, robot=True)) is False
    assert PRED.branches()[-1]["detail"]["reason"] == "window not full"
    # every object gets the drift rule by default (owner 2026-09-24: the resting-contact residual fails 86 / 228 still objects);
    # ``ISG_DRIFT_SETTLED=attached,robot`` restores the stock velocity-gated verdict for a plain object
    plain = Obj("platter_0", {PoseSettled: Settled()})
    assert "all" in PRED.DRIFT_SETTLED_ENTITIES
    assert PRED.evaluate_predicate(pose_settled, plain) is True and PRED.branches()[-1]["detail"]["entity"] == "all"
    monkeypatch.setattr(PRED, "DRIFT_SETTLED_ENTITIES", frozenset({"attached", "robot"}))
    assert PRED.evaluate_predicate(pose_settled, plain) is False and PRED.branches()[-1]["branch"] == "omnigibson"
    assert PRED.evaluate_predicate(pose_settled, robot) is True and PRED.branches()[-1]["detail"]["entity"] == "robot"
    # the attached child keeps its drift rule (owner 2026-09-20)
    attached = Obj("lid_0", {PoseSettled: Settled(), AttachedTo: type("AT", (), {"parent": object(), "children": {}})()})
    assert PRED.evaluate_predicate(pose_settled, attached) is True and PRED.branches()[-1]["detail"]["entity"] == "attached"
    # a drifting plain object under the default rule is not settled: the window carries the rest criterion
    monkeypatch.setattr(PRED, "DRIFT_SETTLED_ENTITIES", frozenset({"all"}))
    assert PRED.evaluate_predicate(pose_settled, Obj("ball_0", {PoseSettled: Settled(drift_pos=0.02)})) is False
    assert PRED.branches()[-1]["branch"] == "none" and PRED.branches()[-1]["detail"]["max_pos_drift_m"] == pytest.approx(0.02)
    # the task's default window: the keys PoseSettled.set_config accepts, 1.5 s at 30 Hz, tolerances tighter than the stock state
    import inspect
    from ispatialgym.tasks.ispatialgym_task import DEFAULT_POSE_SETTLED_CONFIG

    assert set(DEFAULT_POSE_SETTLED_CONFIG) <= set(inspect.signature(PoseSettled.set_config).parameters)
    assert DEFAULT_POSE_SETTLED_CONFIG["n_steps"] == 45 and DEFAULT_POSE_SETTLED_CONFIG["pos_tolerance"] == 0.002
    assert DEFAULT_POSE_SETTLED_CONFIG["ori_tolerance"] == 0.005 and DEFAULT_POSE_SETTLED_CONFIG["joint_tolerance"] == 0.005
    # OmniGibson True ends the evaluation for everyone
    monkeypatch.setattr(BU, "evaluate_bddl_predicate", lambda cls, *ents: True)
    assert PRED.evaluate_predicate(pose_settled, robot) is True and PRED.branches()[-1]["branch"] == "omnigibson"


def test_straight_approach_detection_and_reject_code():
    """Branch consistency (oracle finding): pre -> end segments are tracked from the pre solution; the failure carries the
    distinct ``approach_untrackable`` code for the callers' Reject."""
    from ispatialgym.robot import conventions as C
    from ispatialgym.sampling.v2 import ik as IK

    q = C.eef_quat(th.tensor([0.0, 0.0, -1.0]), th.tensor([0.0, 1.0, 0.0]))
    grasp = (th.tensor([1.0, 0.0, 0.9]), q)
    pre = IK.pre_pose(grasp)  # 10 cm back along -a, same orientation
    assert IK.is_approach(pre, grasp) and IK.is_approach(grasp, pre)
    assert not IK.is_approach(grasp, grasp)  # zero length
    far = (th.tensor([1.0, 0.0, 1.3]), q)
    assert not IK.is_approach(far, grasp)  # 40 cm: a carry, not an approach
    turned = (th.tensor([1.0, 0.0, 1.0]), C.eef_quat(th.tensor([0.0, 0.0, -1.0]), th.tensor([1.0, 0.0, 0.0])))
    assert not IK.is_approach(turned, grasp)  # 90 deg wrist turn: not a straight same-orientation segment
    assert IK.IKResult(False, None, "approach_untrackable: joint jump 2.9 rad at waypoint 4").code == "approach_untrackable"
    assert IK.IKResult(False, None, "ik_joint_limit: right_arm_joint2 within 3 mrad of its limit at the pose").code == "ik_joint_limit"
    assert IK.IKResult(False, None, "exact IK failed at pose 2 of 2").code == ""


def test_transition_resweep_never_lowers_the_recorded_standoff():
    """The re-sweep with the objects placed starts at the recorded s_T; the path's end gives the extra drive-back start."""
    from ispatialgym.sampling.v2 import navigation as NAV
    from ispatialgym.sampling.v2.core import SamplerV2

    assert NAV.transition_standoffs(0.4) == [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    edge = EdgeFrame("south", th.tensor([1.5, 1.0]), th.tensor([0.0, 1.0]), 2.0)  # inward = +y: the robot stands at y < 1 facing +y
    to = Stance("manipulation", (1.5, 0.5, math.pi / 2), 0.5, 0.0, 0.5, edge=edge)  # base 0.5 m in front of the edge

    class Fake:
        sidecar = SidecarDraft("alias", "alignment")

    # no path recorded: the path ends at the stance -> s
    assert SamplerV2.path_end_standoff(Fake, Stance("init", (0.0, 0.0, 0.0), 0.5), to) == 0.5
    # prompt0.connect planned the path to the drive-back pose 0.2 m behind the stance (along -inward = -y)
    Fake.sidecar.add_path(PathRecord("init", "manipulation", "local", [(0.0, 0.0, 0.0), (1.5, 0.3, math.pi / 2)], 1.0, "untucked", None))
    assert abs(SamplerV2.path_end_standoff(Fake, Stance("init", (0.0, 0.0, 0.0), 0.5), to) - 0.7) < 1e-6


# ----------------------------------------------------------------------------------------------------------------
# CLIs (§9): path buckets, whitelist pruning after a failed template, exclusion of produced templates
# ----------------------------------------------------------------------------------------------------------------


def test_multiply_path_buckets():
    from ispatialgym.sampling.multiply_isg import parse_path_buckets

    assert parse_path_buckets("3:10,5:10,10:5") == [{"max_m": 3.0, "n": 10}, {"max_m": 5.0, "n": 10}, {"max_m": 10.0, "n": 5}]
    assert parse_path_buckets(None) is None


def test_sample_isg_blacklist_after_failure_and_exclusion(tmp_path):
    from ispatialgym.sampling.sample_isg import _existing_templates, blacklist_after_failure

    gp = _problem(_entry(["a", "b"], ["p", "q"]))
    choice = next(iter_template_choices(gp, "house_single_floor", np.random.default_rng(0)))
    # a grasp failure blames the target only; a stance failure blames every sampleable model of the pair
    out = blacklist_after_failure(gp, choice, {"rejections": {"grasp_unreachable": 5, "no_transition": 1}})
    assert out == [f"target=apple/{choice.models['target'].model}"]
    assert gp.blacklist("house_single_floor", "target")["apple"][choice.models["target"].model].endswith("grasp_unreachable")
    out2 = blacklist_after_failure(gp, choice, {"rejections": {}, "reject_reason": "no_free_edge"})
    assert len(out2) == 2 and gp.whitelist("house_single_floor", "partner") == {"plate": [m for m in ["p", "q"] if m != choice.models["partner"].model]}
    # templates already on disk are excluded from the draw
    d = tmp_path / gp.alias / choice.template_id
    d.mkdir(parents=True)
    (d / "template_meta.json").write_text(json.dumps({"scene": "house_single_floor", "choice": choice.to_json()}))
    excl = _existing_templates(str(tmp_path), gp, "house_single_floor")
    assert excl == [{r: f"{c.category}/{c.model}" for r, c in choice.models.items()}]
    gp2 = _problem(_entry(["a", "b"], ["p", "q"]))
    rest = list(iter_template_choices(gp2, "house_single_floor", np.random.default_rng(0), exclude=excl))
    assert choice.template_id not in {t.template_id for t in rest} and len(rest) == 3


# ----------------------------------------------------------------------------------------------------------------
# summary layout (§3.5)
# ----------------------------------------------------------------------------------------------------------------


def test_a1_joint_band_gate_with_a_synthetic_band():
    from ispatialgym.sampling.v2 import stances as IS

    right = np.array([[x, lat] for x in (0.5, 0.58, 0.66) for lat in (-0.3, -0.22, -0.14)])
    left = np.array([[x, lat] for x in (0.5, 0.58, 0.66) for lat in (0.18, 0.26, 0.34)])
    assert IS.in_band((0.6, -0.2), right) and not IS.in_band((0.6, 0.2), right) and not IS.in_band((0.6, -0.2), np.zeros((0, 2)))
    assert IS.brace_placement_admissible((0.6, -0.18), (0.6, 0.3), right, left)
    assert not IS.brace_placement_admissible((0.6, -0.18), (0.6, 0.6), right, left)  # brace point outside the left band
    assert not IS.brace_placement_admissible((0.9, -0.18), (0.6, 0.3), right, left)  # work point outside the right band


def test_nextto_goal_takes_the_partner_side_facing_the_init():
    """Owner review 2026-09-16: jar (target) at lateral 0.00, can (partner) at -0.16 (robot's right) -> the goal is on the
    can's left (facing the jar), not lifted over it to -0.28."""
    from ispatialgym.sampling.v2.stances import nearer_side_lateral

    partner = (-0.16 - 0.04, -0.16 + 0.04)  # 8 cm wide can centred at -0.16
    target = (-0.04, 0.04)  # 8 cm wide jar about its origin
    side, lat = nearer_side_lateral(0.0, partner, target, gap=0.02)
    assert side == "left" and abs(lat - (-0.12 + 0.02 + 0.04)) < 1e-9  # near face 2 cm off the can's left face, at -0.06
    side, lat = nearer_side_lateral(-0.40, partner, target, gap=0.02)
    assert side == "right" and abs(lat - (-0.20 - 0.02 - 0.04)) < 1e-9
    # touching (gap 0): partner on the robot's left of the target -> the target goes on the partner's right side
    side, lat = nearer_side_lateral(0.0, (0.10, 0.18), target, gap=0.0)
    assert side == "right" and abs(lat - (0.10 - 0.04)) < 1e-9


def test_open_gripper_lowest_point_for_top_down_and_side_grasps():
    from ispatialgym.robot import conventions as C
    from ispatialgym.sampling.v2.tasks.iors_small import FINGER_THICKNESS, gripper_corners_world

    # top-down: approach a = -z -> the fingertips are the lowest points, FINGERTIP_LENGTH below the eef origin
    q_down = C.eef_quat(th.tensor([0.0, 0.0, -1.0]), th.tensor([0.0, 1.0, 0.0]))
    low = float(gripper_corners_world((th.tensor([0.0, 0.0, 1.0]), q_down))[:, 2].min())
    assert abs(low - (1.0 - C.FINGERTIP_LENGTH)) < 1e-6
    # side grasp with a vertical closing axis: the lower open finger is (open / 2 + thickness) below the origin
    q_side = C.eef_quat(th.tensor([1.0, 0.0, 0.0]), th.tensor([0.0, 0.0, 1.0]))
    low = float(gripper_corners_world((th.tensor([0.0, 0.0, 1.0]), q_side))[:, 2].min())
    assert abs(low - (1.0 - 0.5 * C.GRIPPER_OPEN_M - FINGER_THICKNESS)) < 1e-6
    # grasp record extras and the carry waypoint (FOV rule b) reach the sidecar block
    g = GraspRecord("obj", "right", "k", "n", 0.0, grasp=(th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])), extras={"place_kind": "drop", "rim_z": 0.8},
                    carry=[(th.tensor([0.5, 0.0, 1.1]), th.tensor([0.0, 0.0, 0.0, 1.0]))])
    d = g.to_sidecar()
    assert d["place_kind"] == "drop" and d["rim_z"] == 0.8 and d["carry"][0]["position"] == pytest.approx([0.5, 0.0, 1.1])


def test_stance_idle_posture_and_world_key_string():
    from ispatialgym.sampling.sample_isg import world_key_str

    edge = EdgeFrame.from_edge("+u", [1.0, 0.0], [1.0, 0.0], 1.6)
    manip = Stance("manipulation", (1.35, 0.0, math.pi), 0.75, -0.4, 0.35, edge, arms=["right"])
    assert manip.working_arms == ["right"] and manip.idle_arm == "left" and manip.idle_posture is None
    assert manip.to_sidecar()["idle_posture"] == "untucked"  # the default, always written for a one-arm stance
    manip.idle_posture = "ready"
    d = manip.with_trunk(0.5, -0.2).to_sidecar()
    assert d["idle_posture"] == "ready" and d["arms"] == ["right"]
    init = Stance("init", (1.45, 0.0, math.pi), 0.5, 0.0, 0.45, edge)
    assert init.idle_arm is None and "idle_posture" not in init.to_sidecar()
    assert world_key_str(("house_single_floor", ("dining_room_0", "living_room_0"), ("chair_a", "chair_b"))) == \
        "house_single_floor|rooms=dining_room_0,living_room_0|removed=chair_a,chair_b"


def test_rejection_log_and_summary_layout():
    rl = RejectionLog(keep=3)
    for i in range(5):
        rl.add("grasp_unreachable", f"attempt {i}", index=0, attempt=i)
    rl.add("no_transition", "init -> manipulation")
    d = rl.to_json()
    assert d["counts"] == {"grasp_unreachable": 5, "no_transition": 1} and len(d["examples"]["grasp_unreachable"]) == 3
    assert d["examples"]["grasp_unreachable"][0]["attempt"] == 0 and rl.total == 6
    entry = template_summary(2, 123.456, rl, seeds=[7, 8], extra={"counters": {"scene_reset": 12}})
    assert entry["n_written"] == 2 and entry["time_s"] == 123.46 and entry["rejections"]["no_transition"] == 1 and entry["seeds"] == [7, 8]
    assert entry["counters"] == {"scene_reset": 12} and "scene_reset" not in entry["rejections"]
    rl.counts["scene_reset"] = 3  # a legacy log that still counted resets as rejections
    entry2 = template_summary(0, 1.0, rl)
    assert entry2["counters"] == {"scene_reset": 3} and "scene_reset" not in entry2["rejections"]
    run = new_run_summary("2.0.0-dev", "customs.json", "/tmp/out")
    run["problems"].setdefault("alias", {})["scene__target-apple-a"] = entry
    # the layout bench_report.py reads: problems.<problem>.<template>.{n_written, time_s}
    for temps in run["problems"].values():
        for e in temps.values():
            assert {"n_written", "time_s"} <= set(e)
    json.dumps(run)


def test_contact_height_extra_knob():
    """A/B knob ``generation.contact_height_extra_m``: pads not tips for grasps (lift until the fingertip is x above the
    support, never above the pinch band), face-top clips for blades / pads, the full offset for releases, 0 = off."""
    from ispatialgym.sampling.v2 import core as CORE

    class Fake:
        def __init__(self, x):
            self.x = x

        def gen(self, key, default=None):
            return self.x if key == "contact_height_extra_m" else default

    off = Fake(0.0)
    assert CORE.requested_height_extra(off) == 0.0 and CORE.contact_height_extra(off, "grasp") == (0.0, "off")
    assert CORE.contact_height_extra(Fake(None), "release") == (0.0, "off") and CORE.contact_height_extra(Fake("bad"), "release") == (0.0, "off")
    k = Fake(0.005)
    # a lying marker: fingertip 3 mm above the table, pad centre 8.5 mm, hull top 17 mm → needs 2 mm to clear 5 mm, room 3.5 mm
    eff, why = CORE.contact_height_extra(k, "grasp", fingertip_z=0.803, support_z=0.8, band_top_z=0.817, pad_center_z=0.8085)
    assert abs(eff - 0.002) < 1e-9 and why == "partial: fingertip clearance"
    # a fingertip already 1 cm clear needs nothing
    assert CORE.contact_height_extra(k, "grasp", fingertip_z=0.81, support_z=0.8, band_top_z=0.9, pad_center_z=0.82) == (0.0, "fingertip already clear")
    # a tall object with the tip on the table: the full 5 mm
    assert CORE.contact_height_extra(k, "grasp", fingertip_z=0.8, support_z=0.8, band_top_z=0.9, pad_center_z=0.82) == (0.005, "full")
    # the pinch band clips: pad centre 2 mm below the top-minus-margin line
    eff, why = CORE.contact_height_extra(k, "grasp", fingertip_z=0.8, support_z=0.8, band_top_z=0.83, pad_center_z=0.823)
    assert abs(eff - 0.002) < 1e-9 and why == "clipped: pinch band"
    # blades: fingertip 5 mm above the table beside a 2.5 cm cube → room 1 cm - 5 mm = 1 cm... top 0.825 - 0.01 - 0.805 = 0.01 ≥ 5 mm → full
    assert CORE.contact_height_extra(k, "blade", fingertip_z=0.805, band_top_z=0.825) == (0.005, "full")
    eff, why = CORE.contact_height_extra(Fake(0.02), "blade", fingertip_z=0.805, band_top_z=0.825)
    assert abs(eff - 0.01) < 1e-9 and why == "clipped: face top"
    # pads on a face: centre + half height below the top
    eff, why = CORE.contact_height_extra(Fake(0.02), "push", pad_center_z=0.85, band_top_z=0.87)
    assert abs(eff - 0.008) < 1e-9 and why == "clipped: face top"
    assert CORE.contact_height_extra(Fake(0.02), "release") == (0.02, "full")
    # the one-line module form + the record annotation
    s = Fake(0.02)
    z = CORE.lift_contact_height(s, 0.83, "push", face_top_z=0.85)  # fingertip 0.83 - 0.0188 = 0.8112 → room 0.85 - 0.01 - 0.8112 = 0.0288 ≥ 2 cm
    assert abs(z - 0.85) < 1e-9 and s.height_extra_last["height_extra_clip_reason"] == "full"
    rec = ContactRecord("push", "right", ((0, 0, 0), (0, 0, 0, 1)), ((0, 0, 0), (0, 0, 0, 1)), None, ((0, 0, 0), (0, 0, 0, 1)), (1, 0, 0), 0.1, object="t")
    CORE.annotate_height_extra(rec, s)
    d = rec.to_sidecar()
    assert d["height_extra_m"] == 0.02 and d["height_extra_effective_m"] == 0.02 and d["height_extra_clip_reason"] == "full"
    off_rec = CORE.annotate_height_extra(ContactRecord("push", "right", ((0, 0, 0), (0, 0, 0, 1)), ((0, 0, 0), (0, 0, 0, 1)), None, ((0, 0, 0), (0, 0, 0, 1)), (1, 0, 0), 0.1), off)
    assert off_rec.extras["height_extra_clip_reason"] == "off" and off_rec.extras["height_extra_effective_m"] == 0.0


# ----------------------------------------------------------------------------------------------------------------
# owner 2026-09-24: legs tracked from where the arm IS (IKChecker.continue_chain, program_check._chain_order) and the
# travel driven with the arm where the previous motion left it (program_check.choose_departure_arms)
# ----------------------------------------------------------------------------------------------------------------


def _pose(x, y=0.0, z=0.0):
    return (th.tensor([x, y, z], dtype=th.float32), th.tensor([0.0, 0.0, 0.0, 1.0], dtype=th.float32))


def test_chain_order_defers_a_single_that_ends_a_chain_and_keeps_a_chain_start_first():
    pre, place, clear = _pose(0.0, 0.0, 0.10), _pose(0.0), _pose(0.0, 0.0, 0.15)
    items = [("pre_place", [pre]), ("place", [place]), ("descent", [pre, _pose(0.0, 0.0, 0.05), place]), ("retreat", [place, _pose(0.0, 0.0, 0.05), clear])]
    order = [k for k, _ in PC._chain_order(items)]
    assert order == ["pre_place", "descent", "place", "retreat"]
    # nothing to reorder without chains / singles
    assert [k for k, _ in PC._chain_order([("pre", [pre])])] == ["pre"]
    assert [k for k, _ in PC._chain_order([("lift", [place, clear])])] == ["lift"]
    # a single that is a chain's START (and not the end of another) keeps its place
    items2 = [("approach", [pre, place]), ("pre_grasp", [pre]), ("grasp", [place])]
    assert [k for k, _ in PC._chain_order(items2)] == ["approach", "grasp", "pre_grasp"]


def test_chain_starts_recover_the_start_line_waypoints_leave_out():
    """``line_waypoints`` lists n poses AFTER the start: the leg [pre + 1/3, pre + 2/3, place] leaves from ``pre`` = 2 p0 - p1."""
    pre, place = _pose(0.0, 0.0, 0.30), _pose(0.0, 0.0, 0.0)
    leg = [_pose(0.0, 0.0, 0.20), _pose(0.0, 0.0, 0.10), place]
    starts = PC._chain_starts(leg)
    assert len(starts) == 2 and starts[0][1] is False and starts[1][1] is True
    assert PC._same_pose(starts[1][0], pre)
    # a one-pose chain has only itself; a chain turning between its first two poses has no straight-line start
    assert len(PC._chain_starts([place])) == 1
    turned = [(_pose(0.0, 0.0, 0.2)[0], th.tensor([0.0, 0.0, 0.3827, 0.9239])), _pose(0.0, 0.0, 0.1)]
    assert len(PC._chain_starts(turned)) == 1
    # the real Place layout: pre_place is where the DESCENT leaves from (unlisted) and where the RETREAT ends; it stays first
    back = pre
    items = [("pre_place", [pre]), ("place", [place]), ("descent", leg), ("retreat", [_pose(0.0, 0.0, 0.10), _pose(0.0, 0.0, 0.20), back])]
    assert [k for k, _ in PC._chain_order(items)] == ["pre_place", "descent", "place", "retreat"]
    # a Grasp: approach leaves from pre_grasp (unlisted) and ends at grasp
    items = [("pre_grasp", [pre]), ("grasp", [place]), ("approach", leg)]
    assert [k for k, _ in PC._chain_order(items)] == ["pre_grasp", "approach", "grasp"]


class _ToyKit:
    """A 7-joint 'arm' whose eef position is the first three joint values (identity orientation): differential-IK tracking
    reaches any x up to ``reach``; the IK of a pose beyond the stall lands in the same branch or 3.2 rad away (``far_branch``)."""

    def __init__(self, reach: float = 10.0, far_branch: bool = False):
        self.robot = type("R", (), {})()
        self.robot.arm_control_idx = {"right": th.arange(7, 14), "left": th.arange(14, 21)}
        self.reach, self.far_branch = float(reach), bool(far_branch)
        self.stats = {}
        kit = self

        class _Backend:
            def solve_ik_path(self, name, targets, q_start, check_collisions=False, sync_world=False, max_joint_step=0.2, max_iters=60, **kw):
                idx = kit.robot.arm_control_idx["right"]
                traj = [q_start.clone()]
                for k, t in enumerate(targets):
                    p = list(t.values())[0][0]
                    if float(p[0]) > kit.reach + 1e-9:
                        rem = float(th.linalg.norm(p - traj[-1][idx][:3]))
                        return type("Res", (), {"success": False, "q_traj": th.stack(traj), "status": f"stall at {k}", "diagnostics": {"stage": "diff_ik", "pos_err": rem, "fail_index": k}})()
                    q = traj[-1].clone()
                    q[idx[:3]] = p
                    traj.append(q)
                return type("Res", (), {"success": True, "q_traj": th.stack(traj), "status": "ok", "diagnostics": {}})()

        self.backend = _Backend()

    def _ensure_synced(self):
        pass

    def fk(self, links, q):
        idx = self.robot.arm_control_idx["right"]
        return {links[0]: (q[idx][:3].clone(), th.tensor([0.0, 0.0, 0.0, 1.0]))}

    def collision_free(self, q, margin=0.0):
        n = int(q.shape[0]) if q.dim() == 2 else 1
        return th.ones(n, dtype=th.bool)

    def ik(self, pose, q_seed, arm="right"):
        idx = self.robot.arm_control_idx[arm]
        q = q_seed.clone()
        q[idx[:3]] = pose[0]
        if self.far_branch:
            q[idx[3]] = q_seed[idx[3]] + 3.2
        return q


def _q_at(x: float) -> th.Tensor:
    q = th.zeros(28)
    q[7] = x
    return q


def test_continue_chain_tracks_from_the_given_configuration():
    from ispatialgym.sampling.v2.ik import IKChecker

    ikc = IKChecker(_ToyKit())
    chain = [_pose(0.0), _pose(0.5), _pose(1.0)]
    res = ikc.continue_chain(_q_at(0.0), chain, "right", soft_stall_m=0.03)
    assert res.ok and res.reason.startswith("tracked continuation") and abs(float(res.q[7]) - 1.0) < 1e-6
    # the start must BE at the chain's first pose: no re-solve of the first pose
    res = ikc.continue_chain(_q_at(0.3), chain, "right")
    assert not res.ok and res.reason.startswith("continuation")
    # a one-pose chain: the arm is already there
    res = ikc.continue_chain(_q_at(0.0), [_pose(0.0)], "right")
    assert res.ok and abs(float(res.q[7])) < 1e-9


def test_continue_chain_finishes_a_small_stall_in_branch_and_refuses_another_branch():
    from ispatialgym.sampling.v2.ik import IKChecker

    chain = [_pose(0.0), _pose(1.5)]
    # the tracking stalls 1 cm short (reach 1.49): finished by the IK of the end pose seeded at the stall, same branch -> ok
    ikc = IKChecker(_ToyKit(reach=1.49))
    res = ikc.continue_chain(_q_at(0.0), chain, "right", soft_stall_m=ikc.FREE_LEG_STALL_M)
    assert res.ok and "stall finished" in res.reason and abs(float(res.q[7]) - 1.5) < 1e-6
    assert ikc.last_track_stall is not None and 0.0 < ikc.last_track_stall["remaining_m"] < 0.03 and ikc.last_track_stall["free"]
    # the same stall whose end-pose IK lies 3.2 rad away: the oracle's finish would be rejected (> 0.6 rad) -> the chain fails
    ikc = IKChecker(_ToyKit(reach=1.49, far_branch=True))
    res = ikc.continue_chain(_q_at(0.0), chain, "right", soft_stall_m=ikc.FREE_LEG_STALL_M)
    assert not res.ok and "> 0.60" in res.reason and "another IK branch" in res.reason
    assert ikc.last_track_stall is not None and 0.0 < ikc.last_track_stall["remaining_m"] < 0.03
    # a stall far from the end with a small residual is finished too (the oracle's free-leg finish allows 0.6 m remaining); the
    # remaining distance is recorded for the callers' partial rule
    ikc = IKChecker(_ToyKit(reach=1.2))
    res = ikc.continue_chain(_q_at(0.0), chain, "right", soft_stall_m=ikc.FREE_LEG_STALL_M)
    assert res.ok and "stall finished" in res.reason and ikc.last_track_stall is not None and ikc.last_track_stall["remaining_m"] > 0.25
    # without a soft stall (a reorient) any stall fails
    ikc = IKChecker(_ToyKit(reach=1.49))
    assert not ikc.continue_chain(_q_at(0.0), chain, "right", soft_stall_m=None).ok


def test_away_from_limits_re_solves_from_random_configurations_only_when_the_seeded_retries_fail():
    """Owner 2026-09-26 (bar soap 0926-011542: the place pose's solutions all within 2 mrad of right_arm_joint6's limit on one pod,
    a valid solution for the same candidate on another): after the seeded re-solves stay within the margin, re-solves from random
    arm configurations (reproducible: seeded by the pose) find the solution away from the limit; a solution already clear of the
    limits is kept without any re-solve."""
    from ispatialgym.sampling.v2.ik import IKChecker

    kit = _ToyKit()
    kit.robot.joint_lower_limits = th.full((28,), -3.0)
    kit.robot.joint_upper_limits = th.full((28,), 3.0)
    ikc = IKChecker(kit)
    idx = kit.robot.arm_control_idx["right"]
    calls = []

    def ik(pose, q_seed, arm="right"):
        # seeds near q0 (|arm joints| < 1) land on the limit branch; a seed far from it reaches the free branch
        calls.append(q_seed.clone())
        q = th.zeros(28)
        q[idx] = 2.999 if float(q_seed[idx].abs().max()) < 1.0 else 1.0
        return q

    kit.ik = ik
    ikc._retry_seeds = lambda q0, pose, arm, chain=(): []
    q_lim = th.zeros(28)
    q_lim[idx] = 2.999
    pose = _pose(0.3)
    q, why = ikc.away_from_limits(q_lim, pose, "right", th.zeros(28))
    assert q is not None and why == "" and float(q[idx].max()) == 1.0
    assert 3 < len(calls) <= 3 + IKChecker.LIMIT_RETRY_RANDOM  # 3 seeded re-solves (no remembered seeds: q0 x 3), then random ones
    n_first = len(calls)
    calls.clear()
    q2, _ = ikc.away_from_limits(q_lim, pose, "right", th.zeros(28))
    assert len(calls) == n_first and th.allclose(q, q2)  # the random seeds are reproducible (seeded by the pose)
    # no random re-solve when the seeded ones succeed; none at all when the first solution is clear of the limits
    calls.clear()
    q_ok = th.zeros(28)
    q_ok[idx] = 0.5
    assert ikc.away_from_limits(q_ok, pose, "right", th.zeros(28))[1] == "" and calls == []
    # attempt 0 of an instance solves as before the random re-solves existed (sandal 0926-023744: a rescued solution is remembered
    # as a seed and pulls the attempt's later candidates into its branch); the attempts after it get them
    ikc.attempt = 0
    calls.clear()
    q0_, why0 = ikc.away_from_limits(q_lim, pose, "right", th.zeros(28))
    assert q0_ is None and why0.startswith("ik_joint_limit") and len(calls) == 3  # the seeded re-solves only
    ikc.attempt = IKChecker.LIMIT_RETRY_RANDOM_FROM_ATTEMPT
    q1_, why1 = ikc.away_from_limits(q_lim, pose, "right", th.zeros(28))
    assert q1_ is not None and why1 == "" and th.allclose(q1_, q)
    ikc.attempt = None
    # with the random re-solves switched off the candidate is rejected as before
    ikc.LIMIT_RETRY_RANDOM = 0
    q3, why3 = ikc.away_from_limits(q_lim, pose, "right", th.zeros(28))
    assert q3 is None and why3.startswith("ik_joint_limit")


def test_run_instances_tells_the_ik_checker_the_attempt():
    """``SamplerV2.run_instances`` sets ``ikc.attempt`` before each attempt's sampling (the random joint-limit re-solves start at
    ``IKChecker.LIMIT_RETRY_RANDOM_FROM_ATTEMPT``)."""
    import inspect

    from ispatialgym.sampling.v2.core import SamplerV2

    src = inspect.getsource(SamplerV2.run_instances)
    i_attempt, i_clear, i_sample = src.index("self.ikc.attempt = attempt"), src.index("self.ikc.solutions.clear()"), src.index("self.sample(rng)")
    assert i_clear < i_attempt < i_sample


def test_exact_leaves_from_the_given_first_configuration():
    """``q_first``: the chain's first pose is taken at the recorded witness (FK-verified) instead of a fresh solve."""
    from ispatialgym.sampling.v2.ik import IKChecker

    kit = _ToyKit()
    ikc = IKChecker(kit)
    calls = {"ik": 0}
    real_ik = kit.ik

    def counting_ik(pose, q_seed, arm="right"):
        calls["ik"] += 1
        return real_ik(pose, q_seed, arm)

    kit.ik = counting_ik
    ikc.q_locked = lambda stance, **kw: th.zeros(28)  # the stance configuration (no Stance needed for the toy)
    ikc.away_from_limits = lambda q, pose, arm, q0, tries=4: (q, "")
    ikc.joint_limit_margin = lambda q, arm: (1.0, "none")
    def fake_track_approach(q_pre, pre, end, arm, ignore_end=(), ignore_mid=(), q_hint=None):
        ikc._last_track_start = q_pre.clone()
        return kit.ik(end, q_pre), "ok"

    ikc.track_approach = fake_track_approach
    q_first = _q_at(0.0)
    q_first[10] = 0.7  # a branch marker the solver would not produce
    res = ikc.exact([_pose(0.0), _pose(0.2)], stance=None, arm="right", q_seed=th.zeros(28), cache=False, q_first=q_first)
    assert res.ok and calls["ik"] == 1 and abs(float(res.q[10]) - 0.7) < 1e-6, "the chain left from q_first (only the tracked end was solved)"
    # a q_first that does not reach the first pose is ignored; the seed (zeros) holds the first pose itself and is pinned as the
    # chain's start (2026-09-25, the held-object hold posture), so only the tracked end is solved
    calls["ik"] = 0
    res = ikc.exact([_pose(0.0), _pose(0.2)], stance=None, arm="right", q_seed=th.zeros(28), cache=False, q_first=_q_at(0.5))
    assert res.ok and calls["ik"] == 1
    # a seed away from the first pose is only a seed: both poses are solved as before
    calls["ik"] = 0
    res = ikc.exact([_pose(0.0), _pose(0.2)], stance=None, arm="right", q_seed=_q_at(0.9), cache=False, q_first=_q_at(0.5))
    assert res.ok and calls["ik"] == 2


def test_sampler_state_eef_pose_is_the_fk_of_the_arm_end_and_raises_when_unknown():
    """Owner 2026-09-25 item 1: ``_SamplerState.eef_pose(arm)`` = FK of ``tracker.arm_end_q[arm]`` so ``Place.targets()`` builds the
    carry leg the oracle tracks; unknown (after a travel / a push) -> raises, and ``targets()`` keeps its recorded-pose fallbacks."""
    from ispatialgym.oracle.actions import Place, Pose, Lift, Grasp

    kit = _ToyKit()
    sampler = type("S", (), {})()
    sampler.kit = kit
    sidecar = type("SC", (), {"stances": {}, "blocks": None, "grasps": []})()
    tr = PC._Tracker(sidecar)
    st = PC._SamplerState(sampler, tr)
    with pytest.raises(KeyError):
        st.eef_pose("right")
    tr.arm_end_q["right"] = _q_at(0.4)
    p, q = st.eef_pose("right")
    assert abs(float(p[0]) - 0.4) < 1e-6 and abs(float(q[3]) - 1.0) < 1e-6
    # Place.targets() through the state: the carry leg from where the hand is to the pre-place, a pose every 5 cm (start left out)
    pre = Pose([0.7, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])
    act = Place(kind="support", arm="right", object="teacup.n.02_1", place=Pose([0.7, 0.0, -0.1], [0.0, 0.0, 0.0, 1.0]), pre_place=pre, on="saucer.n.02_1")
    t = act.targets(sidecar, st)
    assert len(t["carry"]) == 6 and abs(t["carry"][-1].position[0] - 0.7) < 1e-6 and abs(t["carry"][0].position[0] - 0.45) < 1e-6
    # without a configuration the carry is empty (the place after a carried travel: as before)
    tr.arm_end_q.clear()
    assert act.targets(sidecar, st)["carry"] == []


def test_exact_pins_a_seed_that_already_holds_the_first_pose_without_the_limit_margin():
    """object_scaling-beaker 0925-172725: the held-object chain leaves from the HOLD posture (q_seed = q_init, FK(q_init) = the first
    pose); it is the chain's start, not a solution to pick, so the joint-limit margin (about the branch an approach would stall in)
    is not applied to it - every precomputed (grasp, goal) had been refused with 'right_arm_joint6 within 3 mrad of its limit'."""
    from ispatialgym.sampling.v2.ik import IKChecker

    kit = _ToyKit()
    ikc = IKChecker(kit)
    ikc.q_locked = lambda stance, **kw: th.zeros(28)
    # the hold posture sits 3 mrad from a limit: away_from_limits would refuse every solution of the first pose
    ikc.joint_limit_margin = lambda q, arm: (0.003, "right_arm_joint6")
    ikc.away_from_limits = lambda q, pose, arm, q0, tries=4: (None, "ik_joint_limit: right_arm_joint6 within 3 mrad of its limit at the pose")
    q_hold = _q_at(0.0)
    q_hold[10] = 0.7  # a marker the solver would not produce
    res = ikc.exact([_pose(0.0)], stance=None, arm="right", q_seed=q_hold, cache=False)
    assert res.ok and abs(float(res.q[10]) - 0.7) < 1e-6, res.reason
    # a seed that does NOT hold the first pose is only a seed: the margin rule applies to the solution
    res = ikc.exact([_pose(0.3)], stance=None, arm="right", q_seed=q_hold, cache=False)
    assert not res.ok and res.reason.startswith("ik_joint_limit")


def test_tracker_keeps_the_holding_arm_through_a_carried_travel_and_drops_the_rest():
    """Owner 2026-09-25: a carried navigate / transition keeps ``arm_end_q`` of the arm holding the object (the oracle freezes it
    through carried legs) and drops the other arms; an empty-handed travel drops everything unless the record keeps the arms."""
    sidecar = type("SC", (), {"stances": {}, "blocks": None, "grasps": []})()
    tr = PC._Tracker(sidecar)
    grasp = type("A", (), {"name": "grasp", "arms": ("right",), "object": "teacup.n.02_1", "kind": "library"})()
    lift = type("A", (), {"name": "lift", "arms": ("right",), "object": "teacup.n.02_1"})()
    brace = type("A", (), {"name": "brace", "arms": ("left",), "object": "x"})()
    nav = type("A", (), {"name": "navigate", "arms": (), "path": None})()
    place = type("A", (), {"name": "place", "arms": ("right",), "object": "teacup.n.02_1"})()
    q_lift = _q_at(0.4)
    tr.after(grasp); tr.note_arm_end(grasp, _q_at(0.1))
    tr.after(lift); tr.note_arm_end(lift, q_lift)
    tr.arm_end_q["left"] = _q_at(0.2)  # some other arm's configuration
    assert tr.carried == "teacup.n.02_1" and tr.carrying_arm() == "right"
    tr.after(nav); tr.note_arm_end(nav, None)
    assert list(tr.arm_end_q) == ["right"] and th.equal(tr.arm_end_q["right"], q_lift), "the holding arm survives the carried travel, the other arm is dropped"
    tr.after(place); tr.note_arm_end(place, _q_at(0.9))
    assert tr.carried is None and tr.carrying_arm() is None
    tr.after(nav); tr.note_arm_end(nav, None)
    assert tr.arm_end_q == {}, "an empty-handed travel forgets the arms"
    # the record keeps the arms (departure arms): everything stays
    tr.note_arm_end(lift, q_lift)
    nav_keep = type("A", (), {"name": "navigate", "arms": (), "path": None, "_keeps_arms": True})()
    tr.after(nav_keep); tr.note_arm_end(nav_keep, None)
    assert th.equal(tr.arm_end_q["right"], q_lift)
    # a held-object init: the carrying arm comes from the sidecar's grasp record
    sidecar2 = type("SC", (), {"stances": {"init": type("St", (), {"carried": "bottle.n.01_1"})()}, "blocks": None,
                                "grasps": [type("G", (), {"object": "bottle.n.01_1", "arm": "left"})()]})()
    tr2 = PC._Tracker(sidecar2)
    assert tr2.carried == "bottle.n.01_1" and tr2.carrying_arm() == "left"


def test_place_order_checks_the_carry_first_with_its_start_prepended():
    """The place's keys in motion order: carry (start prepended) -> pre_place (its end) -> descent -> place -> retreat; ``carry_via``
    is part of the carry; without a carry leg the generic ``_chain_order`` applies."""
    kit = _ToyKit()
    sampler = type("S", (), {})()
    sampler.kit = kit
    pre, place, clear = _pose(0.7), _pose(0.7, 0.0, -0.1), _pose(0.7, 0.0, 0.05)
    carry = [_pose(0.45), _pose(0.5), _pose(0.55), _pose(0.6), _pose(0.65), pre]
    items = [("pre_place", [pre]), ("place", [place]), ("descent", [_pose(0.7, 0.0, -0.033), _pose(0.7, 0.0, -0.066), place]),
             ("retreat", [_pose(0.7, 0.0, -0.05), _pose(0.7, 0.0, 0.0), clear]), ("carry", carry), ("carry_via", [_pose(0.5)])]
    out = PC._place_order(items, _q_at(0.4), sampler, "right")
    assert [k for k, _ in out] == ["carry", "pre_place", "descent", "place", "retreat"]
    assert len(out[0][1]) == 7 and PC._same_pose(out[0][1][0], _pose(0.4)) and PC._same_pose(out[0][1][-1], pre)
    # no carry leg (no configuration): the generic order, carry_via dropped
    out = PC._place_order([(k, p) for k, p in items if k != "carry"], None, sampler, "right")
    assert [k for k, _ in out] == ["pre_place", "descent", "place", "retreat"]
    out = PC._place_order([(k, p) if k != "carry" else (k, []) for k, p in items], _q_at(0.4), sampler, "right")
    assert [k for k, _ in out] == ["pre_place", "descent", "place", "retreat"]


def test_reorient_derived_poses_are_not_checked_as_a_second_chain_when_the_key_poses_are_recorded():
    """``reorient.derived`` re-derives the tracked motion from the start pose and is never executed: with recorded ``waypoints``
    only those are checked (shampoo 0925-210939 failed the re-derived floor-class carry from a fresh IK seed); without recorded
    key poses ``derived`` IS the tracked list and stays. ``base_follow.handle_poses`` keep their skip."""
    targets = {"waypoints": [_pose(0.5)], "derived": [_pose(0.5)], "pivot": [0, 0, 0], "end": _pose(0.5)}
    recorded = type("R", (), {"waypoints": [_pose(0.5)]})()
    assert PC._skipped_target_keys("reorient", recorded, targets) == {"derived"}
    unrecorded = type("R", (), {"waypoints": None})()
    assert PC._skipped_target_keys("reorient", unrecorded, targets) == set()
    assert PC._skipped_target_keys("reorient", recorded, [_pose(0.5)]) == set()
    assert PC._skipped_target_keys("base_follow", None, {"handle_poses": []}) == {"handle_poses"}
    assert PC._skipped_target_keys("lift", None, targets) == set()


def test_path_record_and_transition_carry_arms_q():
    p = PathRecord("a", "b", "local", [(0.0, 0.0, 0.0), (0.1, 0.0, 0.0)], 0.1, arms_q={"right": [0.1] * 7})
    d = p.to_sidecar()
    assert d["arms_q"] == {"right": [0.1] * 7} and PathRecord.schema_ok(d)
    assert "arms_q" not in PathRecord("a", "b", "local", [(0.0, 0.0, 0.0)], 0.0).to_sidecar()
    t = Transition("a", "b", 0.45, extras={"arms_q": {"right": [0.2] * 7}, "arms_source": "where the arm is"})
    assert t.to_sidecar()["arms_q"] == {"right": [0.2] * 7} and t.to_sidecar()["arms_source"] == "where the arm is"


def test_choose_departure_arms_records_the_first_free_candidate_on_every_leg(monkeypatch):
    """The empty-handed travel after a retreat: the arm as left is tried first, then lifted 5 / 10 cm; the first candidate that
    sweeps every leg free is written as ``arms_q`` on the path and the transitions (sampler records + the oracle-side objects)
    and the tracker keeps the arm's configuration through the travel."""
    sd = SidecarDraft("alias", "configuration_matching")
    sd.add_stance(Stance("manipulation_2", (0.0, 0.0, 0.0), 0.75, -0.3, 0.45))
    sd.add_stance(Stance("manipulation_3", (0.0, -0.07, 0.0), 0.75, -0.3, 0.45))
    dep = sd.add_transition(Transition("manipulation_2", "manipulation_3", 0.55, trunk_from=(0.75, -0.3), trunk_to=(0.75, 0.0), at="departure"))
    path = sd.add_path(PathRecord("manipulation_2", "manipulation_3", "local", [(-0.1, 0.0, 0.0), (-0.1, -0.07, 0.0)], 0.07, trunk=(0.75, 0.0)))
    arr = sd.add_transition(Transition("manipulation_2", "manipulation_3", 0.55, trunk_from=(0.75, 0.0), trunk_to=(0.75, -0.3)))

    class Side:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    a_dep = Side(name="transition", transition=Side(frm="manipulation_2", to="manipulation_3", departure=True, extra={}), carried=None)
    a_nav = Side(name="navigate", path=Side(frm="manipulation_2", to="manipulation_3", carried=None, extra={}))
    a_arr = Side(name="transition", transition=Side(frm="manipulation_2", to="manipulation_3", departure=False, extra={}), carried=None)
    actions = [Side(name="retreat", arms=("right",), object="toy_dice.n.01_3"), a_dep, a_nav, a_arr, Side(name="grasp", arms=("right",))]
    assert PC._travel_group(actions, 1) == [1, 2, 3] and PC._travel_group(actions, 4) == []
    assert PC._draft_record(sd, a_dep) is dep and PC._draft_record(sd, a_nav) is path and PC._draft_record(sd, a_arr) is arr

    kit = _ToyKit()
    kit.sync_world = lambda: None
    q_end = _q_at(0.3)  # where the retreat left the right arm (7 arm joints: x = 0.3)

    class IKC:
        def track_segment(self, q, p_from, p_to, arm, **kw):
            q2 = q.clone()
            j = kit.robot.arm_control_idx[arm][2]
            q2[j] = q2[j] + float(p_to[0][2] - p_from[0][2])  # the toy lifts along joint 3
            return q2, "ok"

    sampler = Side(kit=kit, ikc=IKC(), objects={})
    tracker = PC._Tracker(sd)
    tracker.arm_end_q["right"] = q_end.clone()
    seen = []

    def fake_sweep(sidecar, sampler_, action, override):
        seen.append((action.name, {a: float(q[2]) for a, q in override.items()}))
        lift = float(override["right"][2])
        ok = lift >= 0.05 - 1e-9  # the arm as left collides on the path; lifted 5 cm it is free
        return ok, "" if ok else f"{action.name} collides", dict(override)

    monkeypatch.setattr(PC, "_sweep_travel_record", fake_sweep)
    out = PC.choose_departure_arms(sd, sampler, actions, [1, 2, 3], tracker)
    assert out["choice"] == "lifted 0.05 m" and out["legs"] == [1, 2, 3] and out["arms"] == ["right"]
    assert seen[0][0] == "transition" and abs(seen[0][1]["right"]) < 1e-9 and seen[1][0] == "transition" and abs(seen[1][1]["right"] - 0.05) < 1e-6
    assert [s[0] for s in seen] == ["transition", "transition", "navigate", "transition"]  # as left fails at the first leg; lifted 5 cm sweeps all three
    lifted = path.arms_q["right"]
    assert abs(lifted[2] - 0.05) < 1e-6 and abs(lifted[0] - 0.3) < 1e-6 and dep.extras["arms_q"]["right"] == lifted and arr.extras["arms_q"]["right"] == lifted
    assert dep.extras["arms_source"] == "lifted 0.05 m"
    assert a_nav.path.extra["arms_q"]["right"] == lifted and a_dep.transition.extra["arms_q"]["right"] == lifted and a_arr.transition.extra["arms_q"]["right"] == lifted
    assert all(getattr(a, "_keeps_arms", False) for a in (a_dep, a_nav, a_arr))
    assert path.to_sidecar()["arms_q"]["right"] == lifted and "arms_q" in dep.to_sidecar()
    # the tracker keeps the arm through such a travel, and forgets it through one without arms_q
    tracker.note_arm_end(a_nav, None)
    assert "right" in tracker.arm_end_q
    tracker.note_arm_end(Side(name="navigate", path=Side(frm="x", to="y", extra={})), None)
    assert tracker.arm_end_q == {}
    # no candidate free: the records stay as swept (no arms_q), the choice says so
    sd2 = SidecarDraft("alias", "configuration_matching")
    sd2.add_stance(Stance("manipulation_2", (0.0, 0.0, 0.0), 0.75, -0.3, 0.45))
    sd2.add_stance(Stance("manipulation_3", (0.0, -0.07, 0.0), 0.75, -0.3, 0.45))
    path2 = sd2.add_path(PathRecord("manipulation_2", "manipulation_3", "local", [(-0.1, 0.0, 0.0), (-0.1, -0.07, 0.0)], 0.07))
    a_nav2 = Side(name="navigate", path=Side(frm="manipulation_2", to="manipulation_3", carried=None, extra={}))
    tracker2 = PC._Tracker(sd2)
    tracker2.arm_end_q["right"] = q_end.clone()
    monkeypatch.setattr(PC, "_sweep_travel_record", lambda *a: (False, "collides", {}))
    out = PC.choose_departure_arms(sd2, sampler, [a_nav2], [0], tracker2)
    assert out["choice"].startswith("stance posture") and path2.arms_q is None and "arms_q" not in a_nav2.path.extra and not getattr(a_nav2, "_keeps_arms", False)
    assert len(out["tried"]) == 3  # as left, lifted 0.05, lifted 0.10
    # a braced arm / nothing left where it is: no selection at all
    tracker3 = PC._Tracker(sd2)
    assert PC.choose_departure_arms(sd2, sampler, [a_nav2], [0], tracker3) is None
    tracker3.arm_end_q["right"] = q_end.clone()
    tracker3.arm_q["left"] = th.zeros(7)
    assert PC.choose_departure_arms(sd2, sampler, [a_nav2], [0], tracker3) is None


def test_choose_departure_arms_keeps_an_arriving_stance_that_holds_the_next_pre_contact(monkeypatch):
    """A push-close re-stance records the fist at its next pre-contact (``stance.q``): the travel arriving there keeps the stance's
    recorded arms (no departure candidate, the records untouched), so the arm goes straight to the next contact; arriving arms that
    do not reach the next push's pre-contact (a posture) leave the departure choice as it was."""
    sd = SidecarDraft("alias", "configuration_matching")
    sd.add_stance(Stance("manipulation", (0.0, 0.0, 0.0), 0.25, -0.3, 0.45))
    st2 = sd.add_stance(Stance("manipulation2", (0.0, 0.1, 0.0), 0.25, -0.3, 0.45))
    tr = sd.add_transition(Transition("manipulation", "manipulation2", 0.45))
    kit = _ToyKit()
    st2.q = _q_at(0.62)
    st2.q[8], st2.q[9] = -0.17, 1.34  # the toy FK: the fist at (0.62, -0.17, 1.34), the next push's pre-contact

    class Side:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class IKC:
        def q_locked(self, st, **kw):
            return st.q.clone()

        def reaches(self, q, pose, arm):
            return float(th.linalg.norm(q[kit.robot.arm_control_idx[arm]][:3] - pose[0])) <= 0.005

        def track_segment(self, q, p_from, p_to, arm, **kw):
            return q.clone(), "ok"

    pre = Side(position=[0.62, -0.17, 1.34], orientation=[0.0, 0.0, 0.0, 1.0])
    a_tr = Side(name="transition", transition=Side(frm="manipulation", to="manipulation2", departure=False, extra={}), carried=None)
    actions = [Side(name="retreat", arms=("right",), object="blender.n.01_1"), a_tr, Side(name="push", arms=("right",), contact=Side(arm="right", pre=pre))]
    sampler = Side(kit=kit, ikc=IKC(), objects={})
    tracker = PC._Tracker(sd)
    tracker.arm_end_q["right"] = _q_at(0.3)
    monkeypatch.setattr(PC, "_sweep_travel_record", lambda sidecar, sampler_, action, override: (True, "", dict(override)))
    out = PC.choose_departure_arms(sd, sampler, actions, [1], tracker)
    assert out["choice"].startswith("arriving stance's arms") and out["legs"] == [1]
    assert "arms_q" not in tr.extras and "arms_q" not in a_tr.transition.extra and not getattr(a_tr, "_keeps_arms", False)
    st2.q = _q_at(0.0)
    out = PC.choose_departure_arms(sd, sampler, actions, [1], tracker)
    assert out["choice"] == "where the arm is" and tr.extras["arms_q"]["right"] is not None and a_tr._keeps_arms


def test_record_arms_q_reads_the_oracle_side_record():
    class Side:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    nav = Side(name="navigate", path=Side(extra={"arms_q": {"right": [0.1] * 7, "left": [0.0, 1.0]}}))
    got = PC._record_arms_q(nav)
    assert list(got) == ["right"] and got["right"].shape == (7,)  # the malformed left entry is skipped
    assert PC._record_arms_q(Side(name="transition", transition=Side(extra={}))) == {}
    assert PC._record_arms_q(Side(name="grasp")) == {}


def test_instance_id_stays_within_the_file_name_limit():
    from ispatialgym.sampling.tasks.base import INSTANCE_ID_MAX, make_instance_id

    alias = "disembedding-bottle_of_milk-bottle_of_milk_2_bottle_of_coconut_milk_1-breakfast_table"
    tpl = "house_single_floor__destination-tray-hvlfig__source-tray-uekqey__target-bottle_of_milk-mrejrs__distractors-bottle_of_coconut_milk_1-bottle_of_milk_2"
    short = make_instance_id("alias", "tpl", 3)
    assert short == "alias__tpl__0003"
    iid = make_instance_id(alias, tpl, 0)
    assert len(iid.encode()) <= INSTANCE_ID_MAX and len((iid + "-tro_state.json").encode()) <= 255
    assert iid.split("__")[0] == alias and iid.endswith("__0000")
    assert iid == make_instance_id(alias, tpl, 0)  # deterministic
    assert make_instance_id(alias, tpl + "-bottle_of_milk_3", 0) != iid  # distinct templates keep distinct ids


def test_flipped_push_chain_check_rejects_only_a_witness_in_another_branch():
    """Owner 2026-09-26 (axial 0926-043254 / 051919; wave 14 verification 0926-060408 / 060411): a free-object push whose recorded
    pre-contact witness lies more than ``FLIPPED_PUSH_WITNESS_RAD`` from the raw IK of the pre-contact seeded at the stance posture
    is rejected (``approach_untrackable``) - the oracle's closed-loop arc failed from either branch while the sampler's dense track
    passed; a witness in the stance posture's branch (or no raw IK) is not judged at all."""
    from types import SimpleNamespace

    from ispatialgym.sampling.v2 import program_check as PC
    from ispatialgym.sampling.v2.ik import IKResult

    idx = th.arange(7)
    q_wit = th.zeros(28)

    def make(q_raw):
        kit = SimpleNamespace(robot=SimpleNamespace(arm_control_idx={"right": idx}), ik=lambda pose, q_seed, arm="right": q_raw)
        ikc = SimpleNamespace(q_locked=lambda stance, fingers=None: th.zeros(28),
                              waypoints_ok=lambda *a, **k: (_ for _ in ()).throw(AssertionError("no dense track any more")))
        return SimpleNamespace(kit=kit, ikc=ikc)

    chain = [((0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))] * 3
    near = th.zeros(28)
    near[3] = 0.73  # world vertical's green instances (wave 15 0926-061803: rejected at the former 0.6)
    assert PC._flipped_push_chain_check(make(near), chain, None, "right", q_wit, [], [], None, None) is None
    assert PC._flipped_push_chain_check(make(None), chain, None, "right", q_wit, [], [], None, None) is None
    far = th.zeros(28)
    far[6] = 3.1  # the other branch (a wrist flip)
    ok, why = PC._flipped_push_chain_check(make(far), chain, None, "right", q_wit, [], [], None, None)
    assert not ok and why.startswith("approach_untrackable") and "3.10 rad" in why
    assert IKResult(False, None, why, 3).code == "approach_untrackable"


def _budget_sampler(outcomes):
    """A ``SamplerV2`` whose attempts follow ``outcomes`` (True = an instance is written, False = rejected), the rest stubbed."""
    from types import SimpleNamespace

    from ispatialgym.sampling.sim_utils import Reject
    from ispatialgym.sampling.v2.core import SamplerV2

    s = object.__new__(SamplerV2)
    seq = iter(outcomes)
    s.attempts = []

    def sample(rng):
        s.attempts.append(len(s.attempts))
        if not next(seq):
            raise Reject("goal_infeasible", "stub")
        return SimpleNamespace(summary={})

    noop = lambda *a, **k: None  # noqa: E731
    s.ctx = SimpleNamespace(base_seed=0)
    s.kit = SimpleNamespace(stats_snapshot=lambda: {}, stats_delta=lambda a, b: {})
    s.world = SimpleNamespace(reset_scene_objects=lambda names=None, context="": 0)
    s.objects, s.counters, s.template_meta, s.rejections, s.instance_seeds = {}, {}, {}, {}, []
    s.template_paths = None
    s.ikc = SimpleNamespace(solutions={}, attempt=None)
    s.gp = SimpleNamespace(alias="alias-test")
    s.rejection_log = SimpleNamespace(add=noop, counts={})
    s.sample = sample
    for name in ("park_robot", "apply_draft", "settle_and_validate", "check_stances_with_objects", "check_goal_feasible", "after_goal_check",
                 "verify_carry_legs", "program_check", "save_template", "check_goal_occlusion", "constraints_write_back", "write_template_meta", "on_reject"):
        setattr(s, name, noop)
    s.write_instance = lambda draft, index, seed: f"inst{index}"
    s.instance_id = lambda index: f"iid{index}"
    return s


def test_time_budget_stops_only_between_failing_attempts():
    """``sample_isg --time-budget-s`` (owner 2026-10-11): past the deadline no new attempt starts, but the attempt in progress
    and a template's first attempt always run; without a deadline every attempt runs."""
    import time

    s = _budget_sampler([False] * 5)
    assert s.run_instances(1, max_attempts=5, start_index=0) == [] and len(s.attempts) == 5 and "time_budget_stops" not in s.counters
    s = _budget_sampler([False] * 5)
    s.deadline = time.time() - 1.0
    assert s.run_instances(1, max_attempts=5, start_index=0) == [] and len(s.attempts) == 1 and s.counters["time_budget_stops"] == 1
    s = _budget_sampler([True, False, False])
    s.deadline = time.time() - 1.0
    assert s.run_instances(2, max_attempts=3, start_index=0) == ["inst0"] and len(s.attempts) == 1 and s.counters["time_budget_stops"] == 1
    s = _budget_sampler([False, False, True])
    s.deadline = time.time() + 3600.0
    assert s.run_instances(1, max_attempts=5, start_index=0) == ["inst0"] and len(s.attempts) == 3 and "time_budget_stops" not in s.counters


def test_block_design_has_one_and_a_half_times_the_time_budget():
    from types import SimpleNamespace

    from ispatialgym.sampling import sample_isg as SI

    gp = SimpleNamespace(subtask="configuration_matching-block_design", gen=lambda key, default=None: None)
    assert SI.time_budget_factor(gp) == 1.5
    assert SI.time_budget_factor(SimpleNamespace(subtask="mental_rotation", gen=lambda key, default=None: None)) == 1.0
    assert SI.time_budget_factor(SimpleNamespace(subtask="mental_rotation", gen=lambda key, default=None: 2.0)) == 2.0
