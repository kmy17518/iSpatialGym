"""Phase 2's split cone (``generation.phase2_split_cone``, cone A/B variant C): off leaves Phase 2 and the per-attempt draw
untouched, on keeps every (work stance, start position) pair inside both cones and draws the position uniformly, deterministically."""
import json
import math
import os
import tempfile
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import reach_bands as RB
from ispatialgym.sampling.geometry import CameraModel
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.stances import EdgeFrame, Stance

ON = {"phase2_split_cone": True, "init_view_class_deg": [20, 30], "phase2_lateral_shifts_m": [0.0, 0.1, -0.1, 0.2, -0.2, 0.3, -0.3],
      "phase2_tight_class_deg": 8, "phase2_loose_class_deg": 20}
SHIFTS = tuple(ON["phase2_lateral_shifts_m"])
STAND_OFFS = (0.35, 0.45, 0.55)


def _band(d):
    # one grid trunk (0.5, 0): an 8 cm lattice 0.48-1.12 m ahead of the base, +-0.40 m to the side
    cells = [(x, y) for x in np.arange(0.48, 1.13, 0.08) for y in np.arange(-0.40, 0.41, 0.08)]
    meta = {"scene": "s", "support": "t", "object": "apple/x", "grasp_key": "apple__x__t", "generic_pinch": False, "lattice": 0.08, "mode": "tabletop",
            "grasp_names": ["grasp_0"], "stance": {"base_pose2d": [0.55, 0.0, 0.0]}, "union_extent": [-0.40, 0.40],
            "trunks": [{"translate": 0.5, "tilt": 0.0, "collides": False}], "feasible_sets": []}
    name = "tabletop__s__t__apple-x__k"
    np.savez_compressed(os.path.join(d, name + ".npz"), **{"t0.50_k+0.0": np.asarray(cells, dtype=np.float32),
                                                           "t0.50_k+0.0__pairs": np.asarray([(i, 0, 0) for i in range(len(cells))], dtype=np.int32)})
    with open(os.path.join(d, name + ".json"), "w") as f:
        json.dump(meta, f)
    return RB.ReachBands.load("s", "t", "apple/x", "apple__x__t", dirs=[d])


def _camera(st):
    return C.camera_pose(C.torso_q(st.translate, st.tilt), base_pose=tuple(float(v) for v in st.base_pose2d))


def _scene(monkeypatch, gen):
    """A robot facing a top edge at x = 1.0 (init trunk 0.75 / 0, stand-off 0.45), the conventions camera FK, the 1080 px head camera."""
    monkeypatch.setattr(TT, "camera_fk", lambda s=None, hand_below=None: SimpleNamespace(tilt_limit=lambda tr: -0.6))
    cams = lambda qs: [C.camera_pose(C.torso_q(float(q[0]), float(q[1])), base_pose=tuple(float(v) for v in q[2:5])) for q in qs]  # noqa: E731
    s = SimpleNamespace(kit=SimpleNamespace(trav_free=lambda xy: True, camera_poses_batch=cams),
                        ikc=SimpleNamespace(stances_collision_free=lambda cands: th.ones(len(cands), dtype=th.bool)),
                        stance_q=lambda c: th.tensor([c.translate, c.tilt, *c.base_pose2d], dtype=th.float64),
                        world=SimpleNamespace(camera_model=CameraModel(1080, 1080, 459.0, 459.0, 540.0, 540.0)), gen=lambda k, d=None: gen.get(k, d))
    edge = EdgeFrame.from_edge("+x", th.tensor([1.0, 0.0]), th.tensor([-1.0, 0.0]), 1.6)
    init = Stance("init", (0.55, 0.0, 0.0), 0.75, 0.0, 0.45, edge)
    top = SimpleNamespace(contains=lambda c, erosion: True, z=0.75)
    return s, init, top


def _phase2(s, init, top, rb, split):
    return TT.phase2_manipulation_stance(s, init, 0.03, target_height=0.06, stand_offs=STAND_OFFS, translates=(0.5,), n_tilts=3, bands=rb, top=top, focus_z=0.78,
                                         split_cone=split)


def _key(c):
    return (tuple(round(float(v), 6) for v in c.base_pose2d), c.translate, round(c.tilt, 6), tuple(round(float(v), 6) for v in c.focus.tolist()),
            {k: (round(float(v), 9) if isinstance(v, float) else v) for k, v in c.extras.items() if k != "focus_info"})


def test_split_cone_off_leaves_phase2_and_the_draw_alone(monkeypatch):
    from ispatialgym.sampling.v2.core import SamplerV2

    s, init, top = _scene(monkeypatch, {})
    assert TT.split_cone_opts(s) is None
    with tempfile.TemporaryDirectory() as d:
        rb = _band(d)
        plain = TT.phase2_manipulation_stance(s, init, 0.03, target_height=0.06, stand_offs=STAND_OFFS, translates=(0.5,), n_tilts=3, bands=rb, top=top, focus_z=0.78)
        off = _phase2(s, init, top, rb, TT.split_cone_opts(s))
    assert plain and [_key(c) for c in plain] == [_key(c) for c in off]
    assert all("_position" not in c.extras and "init_view_angle" not in c.extras for c in off)
    # the per-attempt draw neither touches the candidates nor consumes the attempt's random stream
    s.manip_candidates = off
    rng = np.random.default_rng(7)
    before = json.dumps(rng.bit_generator.state, sort_keys=True)
    TT.split_cone_draw(s, rng)
    assert s.manip_candidates is off and json.dumps(rng.bit_generator.state, sort_keys=True) == before
    # init -> manipulation stays the plain §3.8 transition
    calls = []
    fake = SimpleNamespace(record_transition=lambda frm, to: calls.append((frm.name, to.name)) or "tr")
    manip = off[0].with_trunk(off[0].translate, off[0].tilt, "manipulation")
    assert SamplerV2.record_stance_travel(fake, init, manip) == "tr"
    assert calls == [("init", "manipulation")] and "init_travel" not in manip.extras


def test_split_cone_travel_records_the_move(monkeypatch):
    from ispatialgym.sampling.sim_utils import Reject
    from ispatialgym.sampling.v2 import ik as IK
    from ispatialgym.sampling.v2 import prompt0 as P0
    from ispatialgym.sampling.v2.core import SamplerV2

    _, init, _ = _scene(monkeypatch, ON)
    calls, fail_departure = [], []

    def departure(frm, to, travel, min_standoff=None):
        calls.append(("departure", travel, round(min_standoff, 6)))
        if fail_departure:
            raise Reject("no_transition", "blocked")
        return SimpleNamespace(stand_off=0.65)

    old = [SimpleNamespace(frm="init", to="manipulation"), SimpleNamespace(frm="manipulation", to="placement")]
    fake = SimpleNamespace(sidecar=SimpleNamespace(paths=list(old), transitions=list(old)), gen=lambda k, d=None: d, kit=None, gp=SimpleNamespace(alias="a"),
                           record_transition=lambda frm, to, trunk_from=None: calls.append(("transition", trunk_from)) or SimpleNamespace(stand_off=0.45),
                           record_departure_transition=departure,
                           record_path=lambda frm, to, **kw: calls.append(("path", kw["trunk"], tuple(round(v, 6) for v in kw["start_pose2d"]))) or SimpleNamespace(length_m=0.5))
    fake.drop_travel_records = lambda frm, to: SamplerV2.drop_travel_records(fake, frm, to)
    fake.record_edge_travel = lambda frm, to: SamplerV2.record_edge_travel(fake, frm, to)
    monkeypatch.setattr(IK, "stance_arm_vectors", lambda kit, st, posture=None: (th.zeros(7), th.zeros(7), None))
    monkeypatch.setattr(P0, "connect", lambda s, frm, to, path_max_m: calls.append(("connect", path_max_m)) or (SimpleNamespace(length_m=0.4123), SimpleNamespace(stand_off=0.55)))
    # same lateral: the transition alone, the pair's earlier records dropped
    same = init.shifted(stand_off=0.35, name="manipulation")
    same.extras["_position"] = (10, 0)
    assert SamplerV2.record_stance_travel(fake, init, same).stand_off == 0.45 and calls == [("transition", None)]
    assert same.extras["init_travel"] == {"lateral_m": 0.0, "base_m": 0.1, "path_m": None, "departure_stand_off_m": None, "transition_stand_off_m": 0.45}
    assert [(p.frm, p.to) for p in fake.sidecar.paths] == [("manipulation", "placement")]
    # shifted along the edge, both trunks upright: the path to the drive-back pose + the transition (prompt0.connect)
    calls.clear()
    moved = init.shifted(stand_off=0.45, lateral=init.lateral + 0.2, name="manipulation")
    moved.extras["_position"] = (10, 2)
    assert SamplerV2.record_stance_travel(fake, init, moved).stand_off == 0.55 and calls == [("connect", 3.0)]
    tr = moved.extras["init_travel"]
    assert abs(tr["lateral_m"]) == pytest.approx(0.2) and tr["base_m"] == pytest.approx(0.2) and tr["path_m"] == 0.4123 and tr["transition_stand_off_m"] == 0.55
    # the work trunk bows forward: drive back 0.10 m beyond the init stand-off first, the path upright at the init translate, then the arrival
    calls.clear()
    bowed = moved.with_trunk(0.5, -0.3, "manipulation")
    assert SamplerV2.record_stance_travel(fake, init, bowed).stand_off == 0.45
    assert calls == [("departure", (0.75, 0.0), 0.55), ("path", (0.75, 0.0), (0.35, 0.0, 0.0)), ("transition", (0.75, 0.0))]
    assert bowed.extras["init_travel"]["departure_stand_off_m"] == 0.65 and bowed.extras["init_travel"]["path_m"] == 0.5
    # no drive-back at either travel trunk: prompt0.connect
    calls.clear()
    fail_departure.append(True)
    assert SamplerV2.record_stance_travel(fake, init, bowed).stand_off == 0.55
    assert [c[0] for c in calls] == ["departure", "departure", "connect"] and bowed.extras["init_travel"]["departure_stand_off_m"] is None


def test_split_cone_reaches_every_tabletop_phase2_task():
    """Every small-scale task module that calls tabletop Phase 2 forwards the option and records the init -> manipulation move
    through ``record_stance_travel`` (block design: its own ``_travel_between``)."""
    tasks = os.path.join(os.path.dirname(TT.__file__), "tasks")
    for name in ("attachment", "alignment_axial", "iors_small", "disembedding", "block_design", "articulation_small"):
        with open(os.path.join(tasks, name + ".py")) as f:
            src = f.read()
        n = src.count("TT.phase2_manipulation_stance(")
        assert n > 0 and src.count("split_cone=") >= n, name
        assert "record_transition(init, " not in src and "record_transition(self.init_stance, " not in src, name


def test_split_cone_keeps_both_cones_and_draws_positions_uniformly(monkeypatch):
    s, init, top = _scene(monkeypatch, ON)
    opts = TT.split_cone_opts(s)
    assert opts == {"init_view_classes_deg": (20.0, 30.0), "lateral_shifts": SHIFTS, "work_classes_deg": (8.0, 20.0)}
    cm = s.world.camera_model
    with tempfile.TemporaryDirectory() as d:
        rb = _band(d)
        out = _phase2(s, init, top, rb, opts)
        cam0 = _camera(init)
        assert len(out) > 50
        for c in out:
            cam = _camera(c)
            a0, a = cm.angle_off_axis(cam0, c.focus), cm.angle_off_axis(cam, c.focus)
            assert a0 == pytest.approx(c.extras["init_view_angle"], abs=1e-5) and a == pytest.approx(c.extras["axis_angle"], abs=1e-5)
            assert c.extras["init_view_angle"] <= math.radians(30.0) and c.extras["axis_angle"] <= math.radians(20.0)
            assert c.extras["init_view_class"] == (0 if c.extras["init_view_angle"] <= math.radians(20.0) else 1)
            assert c.extras["angle_class"] == (0 if c.extras["axis_angle"] <= math.radians(8.0) else 1)
            assert TT.default_in_frame(s, init, cam0, c.focus, None, 0.03, 0.06, TT.IN_FRAME_MARGIN)
            assert TT.default_in_frame(s, c, cam, c.focus, None, 0.03, 0.06, TT.IN_FRAME_MARGIN)
            # the position is a band cell of its work stance; the work stance stands where its shift puts it along the edge
            cells = TT.band_cells_world(rb, c)
            assert float(th.linalg.norm(cells - c.focus[:2], dim=1).min()) < 1e-5
            assert c.extras["lateral_shift_m"] in SHIFTS and c.lateral == pytest.approx(init.lateral + c.extras["lateral_shift_m"]) and c.stand_off in STAND_OFFS
        assert {c.extras["lateral_shift_m"] for c in out} - {0.0}  # the search along the edge is used
    pool = TT.split_cone_positions(out)
    assert any(c.extras["init_view_class"] == 0 for c in out) and len(pool) > 5
    for key, cs in pool.items():
        assert all(c.extras["init_view_class"] == 0 and tuple(c.extras["_position"]) == key for c in cs)
        tight = [c for c in out if tuple(c.extras["_position"]) == key and c.extras["init_view_class"] == 0 and c.extras["angle_class"] == 0]
        assert {id(c) for c in cs} == ({id(c) for c in tight} if tight else {id(c) for c in out if tuple(c.extras["_position"]) == key and c.extras["init_view_class"] == 0})
    # one uniform draw of a position per attempt; its best-class work stances become the task's candidates
    s.manip_candidates = out
    counts = Counter()
    n = 300 * len(pool)
    for seed in range(n):
        TT.split_cone_draw(s, np.random.default_rng(seed))
        key = tuple(s.manip_candidates[0].extras["_position"])
        assert [id(c) for c in s.manip_candidates] == [id(c) for c in pool[key]]
        counts[key] += 1
    assert set(counts) == set(pool)
    assert max(counts.values()) < 1.3 * n / len(pool) and min(counts.values()) > 0.7 * n / len(pool)


def test_split_cone_is_deterministic(monkeypatch):
    s, init, top = _scene(monkeypatch, ON)
    with tempfile.TemporaryDirectory() as d:
        rb = _band(d)
        a, b = _phase2(s, init, top, rb, TT.split_cone_opts(s)), _phase2(s, init, top, rb, TT.split_cone_opts(s))
    assert [_key(c) for c in a] == [_key(c) for c in b]
    draws = []
    for cands in (a, b):
        t, _, _ = _scene(monkeypatch, ON)
        t.manip_candidates = cands
        seq = []
        for seed in range(40):
            rng = np.random.default_rng(seed)
            TT.split_cone_draw(t, rng)
            seq.append((tuple(t.manip_candidates[0].extras["_position"]), len(t.manip_candidates), float(rng.random())))
        draws.append(seq)
    assert draws[0] == draws[1]


def test_split_cone_d009b_admits_the_loose_class_per_position_when_every_tight_focus_is_deep():
    edge = EdgeFrame.from_edge("+x", th.tensor([1.0, 0.0]), th.tensor([-1.0, 0.0]), 1.6)

    def pair(depth, work_class, pos):
        st = Stance("manipulation", (0.55, 0.0, 0.0), 0.5, 0.0, 0.45, edge)
        st.focus = th.tensor([0.55 + depth, 0.0, 0.78])
        st.extras.update({"angle_class": work_class, "init_view_class": 0, "_position": pos})
        return st

    deep0, loose = pair(0.80, 0, (1, 0)), pair(0.60, 1, (1, 0))
    near0, loose2 = pair(0.60, 0, (2, 0)), pair(0.55, 1, (2, 0))
    pool = TT.split_cone_positions([deep0, loose, near0, loose2], class1_when_deep_m=0.65)
    assert pool[(1, 0)] == [deep0, loose] and pool[(2, 0)] == [near0]
    assert TT.split_cone_positions([deep0, loose])[(1, 0)] == [deep0]


def test_split_cone_on_from_the_subtask_defaults():
    """Owner 2026-10-11 (variant C in every tabletop subtask): the keys come from the subtask's generation_defaults when the alias's
    customs lack them; an alias's own keys still win (``phase2_split_cone: false`` turns it off); the other subtasks stay off."""
    def sampler(subtask, gen):
        return SimpleNamespace(subtask=subtask, gen=lambda k, d=None: gen.get(k, d))

    on = {"init_view_classes_deg": (20.0, 30.0), "lateral_shifts": SHIFTS, "work_classes_deg": (8.0, 20.0)}
    for sub in ("disembedding", "configuration_matching-block_design", "configuration_matching-articulation_open_small_scale",
                "configuration_matching-articulation_close_small_scale", "configuration_matching-attachment", "alignment-axial",
                "inter_object_relation_satisfaction-ontop_small_scale", "inter_object_relation_satisfaction-under_small_scale",
                "inter_object_relation_satisfaction-inside", "inter_object_relation_satisfaction-nextto", "inter_object_relation_satisfaction-touching"):
        fallback = (8.0, 20.0) if sub in ("alignment-axial", "inter_object_relation_satisfaction-inside") else (20.0, 30.0)
        assert TT.split_cone_opts(sampler(sub, {})) == {**on, "fallback_classes_deg": fallback}, sub
    assert TT.split_cone_opts(sampler("alignment-axial", {"phase2_split_cone": False})) is None
    assert TT.split_cone_opts(sampler("alignment-axial", {"phase2_tight_class_deg": 5}))["work_classes_deg"] == (5.0, 20.0)
    for sub in ("alignment-world_vertical", "configuration_matching-articulation_open_large_scale", "inter_object_relation_satisfaction-ontop_large_scale"):
        assert TT.split_cone_opts(sampler(sub, {})) is None, sub


def test_split_cone_draw_reads_the_subtask_defaults():
    """The per-attempt position draw is on when C comes from the subtask's generation_defaults (504df81e left it reading the
    alias's customs only, so the launched jobs kept every position's stances in the task's own order)."""
    edge = EdgeFrame.from_edge("+x", th.tensor([1.0, 0.0]), th.tensor([-1.0, 0.0]), 1.6)

    def cand(pos):
        st = Stance("manipulation", (0.55, 0.0, 0.0), 0.5, 0.0, 0.45, edge)
        st.focus = th.tensor([0.9, 0.1 * pos, 0.78])
        st.extras.update({"angle_class": 0, "init_view_class": 0, "_position": (pos, 0)})
        return st

    cands = [cand(p) for p in (1, 1, 2, 3)]
    s = SimpleNamespace(subtask="inter_object_relation_satisfaction-ontop_small_scale", gen=lambda k, d=None: d, manip_candidates=list(cands))
    TT.split_cone_draw(s, np.random.default_rng(0))
    assert len({tuple(c.extras["_position"]) for c in s.manip_candidates}) == 1 and len(s.manip_candidates) < len(cands)
    off = SimpleNamespace(subtask="inter_object_relation_satisfaction-ontop_small_scale", gen=lambda k, d=None: {"phase2_split_cone": False}.get(k, d),
                          manip_candidates=list(cands))
    TT.split_cone_draw(off, np.random.default_rng(0))
    assert off.manip_candidates == cands


def test_split_cone_without_a_pair_falls_back_to_the_plain_phase2(monkeypatch):
    """Owner 2026-10-11 ~20:10: a template whose split cone finds no (stance, position) pair (the 55's first jobs: 0 pairs on
    countertop_kelker_0 and four other supports) runs the plain Phase 2 with ``fallback_classes_deg`` instead of producing no
    instance; without the key it stays empty."""
    s, init, top = _scene(monkeypatch, {})
    monkeypatch.setattr(TT, "phase2_split_cone", lambda *a, **k: [])
    split = {"init_view_classes_deg": (20.0, 30.0), "lateral_shifts": SHIFTS, "work_classes_deg": (8.0, 20.0)}
    with tempfile.TemporaryDirectory() as d:
        rb = _band(d)
        plain = TT.phase2_manipulation_stance(s, init, 0.03, target_height=0.06, stand_offs=STAND_OFFS, translates=(0.5,), n_tilts=3, bands=rb, top=top,
                                              focus_z=0.78, angle_classes_deg=(20.0, 30.0))
        none = _phase2(s, init, top, rb, split)
        fb = _phase2(s, init, top, rb, {**split, "fallback_classes_deg": (20.0, 30.0)})
    assert none == [] and plain and [_key(c) for c in fb] == [_key(c) for c in plain]
