"""IORS small: a candidate whose init -> manipulation travel fails deterministically is dropped (owner 2026-10-12 item 7e; touching#02
c5_k0 redrew its only candidate 407 times, every attempt ``no_transition``): the determinism rule (a travel reason with every task object
at its park slot), the drop out of the template's draws (the split cone's position pools included), the start region exhausted ->
``TemplateExhausted``, and ``SamplerV2.run_instances`` ending the template's attempts on it. No simulator."""

from __future__ import annotations

import types
from types import SimpleNamespace

import numpy as np
import pytest
import torch as th

from ispatialgym.sampling.sim_utils import PARK_POSITION, Reject
from ispatialgym.sampling.v2 import core
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.problem import RejectionLog
from ispatialgym.sampling.v2.stances import EdgeFrame, Stance
from ispatialgym.sampling.v2.tasks import iors_small as IS

ONTOP = "inter_object_relation_satisfaction-ontop_small_scale"


class _Obj:
    def __init__(self, xyz):
        self.p = th.tensor(xyz, dtype=th.float32)

    def get_position_orientation(self):
        return self.p, th.tensor([0.0, 0.0, 0.0, 1.0])


def _parked_world(n=2):
    return SimpleNamespace(spawned={f"o{i}": _Obj((PARK_POSITION + th.tensor([float(i), 0.0, 0.0])).tolist()) for i in range(n)})


def test_a_travel_failure_is_deterministic_only_with_every_task_object_parked():
    world = _parked_world()
    for reason in IS.TRAVEL_REASONS:
        assert IS.travel_failure_deterministic(Reject(reason, "init -> manipulation: ..."), world)
    assert not IS.travel_failure_deterministic(Reject("grasp_unreachable", "..."), world)  # not a travel failure
    on_table = _parked_world()
    on_table.spawned["o1"] = _Obj([1.0, 2.0, 0.8])  # left on the table by a written instance
    assert not IS.travel_failure_deterministic(Reject("no_transition", "..."), on_table)
    swapped = _parked_world()
    swapped.spawned["o0"], swapped.spawned["o1"] = swapped.spawned["o1"], swapped.spawned["o0"]  # each object has its own slot
    assert not IS.travel_failure_deterministic(Reject("no_transition", "..."), swapped)


# ---------------------------------------------------------------------------------------------------------------- the drop


def _edge():
    return EdgeFrame.from_edge("+x", th.tensor([1.0, 0.0]), th.tensor([-1.0, 0.0]), 1.6)


def _cand(pos, sides, view=0, name=None):
    st = Stance("manipulation", (0.55, 0.1 * pos, 0.0), 0.5, 0.0, 0.45, _edge())
    st.focus = th.tensor([0.9, 0.1 * pos, 0.78])
    st.extras.update({"angle_class": 0, "init_view_class": view, "_position": (pos, 0), TT.START_REGION: {sd: [0.03, 0.1] for sd in sides},
                      "id": name or f"p{pos}{''.join(s[0] for s in sides)}{view}"})
    return st


def _sampler(cands, sides=(1.0, -1.0), **extra):
    s = IS.IORSSmall.__new__(IS.IORSSmall)
    s.subtask, s.spec, s.gp = ONTOP, SimpleNamespace(generation={}), SimpleNamespace(variant="a")
    s.manip_candidates, s.region_sides, s.region_side = list(cands), list(sides), None
    s.shelf_start = s.shelf_pose = None
    s.counters, s.template_meta = {}, {}
    for k, v in extra.items():
        setattr(s, k, v)
    return s


def test_a_dropped_candidate_leaves_every_draw_and_its_side_when_it_was_the_last():
    cands = [_cand(1, ["left"]), _cand(1, ["left", "right"]), _cand(2, ["right"]), _cand(3, ["left"]), _cand(4, ["left"], view=1)]
    s = _sampler(cands)
    for side in (1.0, -1.0):  # both pools built (split cone on for the subtask)
        s.region_side = side
        TT.split_cone_draw(s, np.random.default_rng(0))
    assert set(s._split_cone_pools[1.0]) == {(1, 0), (3, 0)}  # the init-view class 0 positions only
    s.drop_candidate(cands[3], Reject("no_transition", "init -> manipulation: no s_T ..."))
    assert IS.TRAVEL_DROPPED in cands[3].extras and "[no_transition]" in cands[3].extras[IS.TRAVEL_DROPPED]
    assert cands[3] not in s._split_cone_cands and cands[3] not in s.manip_candidates and set(s._split_cone_pools[1.0]) == {(1, 0)}
    assert s.region_sides == [1.0, -1.0] and s.counters["travel_dropped"] == 1 and len(s.template_meta["travel_dropped"]) == 1
    # every class-0 left stance dropped: the left pool goes and is rebuilt from the live candidates - the class-1 position
    s.drop_candidate(cands[0], Reject("path_collision", "..."))
    s.drop_candidate(cands[1], Reject("no_transition", "..."))
    assert 1.0 not in s._split_cone_pools and s.region_sides == [1.0, -1.0]
    s.region_side = 1.0
    TT.split_cone_draw(s, np.random.default_rng(1))
    assert s.manip_candidates == [cands[4]]
    # the right side lost (1, 0) with cands[1]; its last candidate goes, then the left's: nothing left to draw
    s.drop_candidate(cands[2], Reject("no_transition", "..."))
    assert s.region_sides == [1.0] and not s.start_region_exhausted()
    s.drop_candidate(cands[4], Reject("no_transition", "..."))
    assert s.region_sides == [] and s.start_region_exhausted() and s.counters["travel_dropped"] == 5


def test_without_the_split_cone_the_template_candidates_are_dropped_and_the_shelf_layout_has_no_sides():
    cands = [_cand(1, ["left"]), _cand(2, ["left"])]
    s = _sampler(cands, sides=(1.0,))
    s.drop_candidate(cands[0], Reject("no_transition", "..."))
    assert s.manip_candidates == [cands[1]] and s.live_candidates(1.0) == [cands[1]] and not s.start_region_exhausted()
    shelf = _sampler([_cand(1, []), _cand(2, [])], sides=(), shelf_start={"lean_side": 1.0})
    shelf.manip_candidates[0].extras.pop(TT.START_REGION)
    shelf.manip_candidates[1].extras.pop(TT.START_REGION)
    first, second = shelf.manip_candidates
    shelf.drop_candidate(first, Reject("no_transition", "..."))
    assert not shelf.start_region_exhausted()
    shelf.drop_candidate(second, Reject("no_transition", "..."))
    assert shelf.start_region_exhausted()


# ---------------------------------------------------------------------------------------------------------------- the attempt


def _attempt_sampler(cands, failing, parked=True, phase3_reason="grasp_unreachable"):
    """``sample_instance`` on a stand-in: the travel of the candidates named in ``failing`` raises ``no_transition``, the others pass and
    then fail the target draw with ``phase3_reason``; ``travelled`` records the candidates whose travel was checked."""
    travelled = []

    def travel(frm, to):
        travelled.append(to.extras["id"])
        if to.extras["id"] in failing:
            raise Reject("no_transition", f"init -> manipulation: no s_T passes ({to.extras['id']})")

    def phase3(manip, rng, arms=None):
        raise Reject(phase3_reason, "no feasible (yaw, grasp)")

    world = _parked_world()
    if not parked:
        world.spawned["o0"] = _Obj([1.0, 0.0, 0.8])
    s = _sampler(cands, sides=(1.0,), region_side=1.0, world=world, segment_arms=["right"], init_stance=Stance("init", (0.55, 0.0, 0.0), 0.5, 0.0, 0.45, _edge()),
                 bindings=SimpleNamespace(one=lambda role: SimpleNamespace(name=f"{role}.n.01_1")),
                 sidecar=SimpleNamespace(add_stance=lambda st: None, transitions=[], stances={}, grasps=[]))
    s.camera_pose_at = lambda st: None
    s.record_stance_travel = travel
    s.phase3_two_tier = phase3
    return s, travelled


def test_the_attempt_drops_a_deterministic_travel_failure_and_ends_the_template_when_nothing_is_left():
    cands = [_cand(1, ["left"], name="A"), _cand(2, ["left"], name="B")]
    s, travelled = _attempt_sampler(cands, failing={"A", "B"})
    with pytest.raises(core.TemplateExhausted) as e:
        s.sample_instance(np.random.default_rng(0))
    assert e.value.reason == "target_placement" and "start region exhausted" in str(e.value) and sorted(travelled) == ["A", "B"]
    assert all(IS.TRAVEL_DROPPED in c.extras for c in cands) and s.counters["travel_dropped"] == 2 and s.region_sides == []


def test_only_the_failing_candidate_is_dropped_and_a_later_attempt_never_checks_it_again():
    cands = [_cand(1, ["left"], name="A"), _cand(2, ["left"], name="B")]
    s, travelled = _attempt_sampler(cands, failing={"A"})
    with pytest.raises(Reject) as e:
        s.sample_instance(np.random.default_rng(0))
    assert not isinstance(e.value, core.TemplateExhausted) and e.value.reason == "target_placement"  # B failed its draw, not its travel
    assert IS.TRAVEL_DROPPED in cands[0].extras and IS.TRAVEL_DROPPED not in cands[1].extras and s.region_sides == [1.0]
    travelled.clear()
    for seed in range(3):
        with pytest.raises(Reject):
            s.sample_instance(np.random.default_rng(seed))
    assert travelled == ["B", "B", "B"]


def test_a_travel_failure_with_a_task_object_on_the_table_drops_nothing():
    cands = [_cand(1, ["left"], name="A")]
    s, travelled = _attempt_sampler(cands, failing={"A"}, parked=False)
    with pytest.raises(Reject) as e:
        s.sample_instance(np.random.default_rng(0))
    assert not isinstance(e.value, core.TemplateExhausted) and IS.TRAVEL_DROPPED not in cands[0].extras and s.counters == {}


# ---------------------------------------------------------------------------------------------------------------- run_instances


def _run_sampler(exc):
    calls = []

    def sample(rng):
        calls.append(1)
        raise exc

    s = core.SamplerV2.__new__(core.SamplerV2)
    s.deadline, s.objects, s.template_paths, s.template_meta, s.counters, s.rejections = None, {}, None, {}, {}, {}
    s.world = SimpleNamespace(reset_scene_objects=lambda *a, **k: 0)
    s.ikc, s.kit = SimpleNamespace(solutions={}, attempt=0), SimpleNamespace(stats_snapshot=lambda: {})
    s.gp, s.ctx, s.rejection_log = SimpleNamespace(variant="a"), SimpleNamespace(base_seed=0), RejectionLog(keep=3)
    s.park_robot = lambda: None
    s.on_reject = lambda e: None
    s.sample = sample
    return s, calls


def test_run_instances_ends_the_templates_attempts_on_template_exhausted():
    s, calls = _run_sampler(core.TemplateExhausted("target_placement", "start region exhausted"))
    assert s.run_instances(2, max_attempts=5, indices=[0, 1]) == []
    assert len(calls) == 1 and s.counters["template_exhausted"] == 1 and s.rejection_log.counts == {"target_placement": 1}
    plain, calls = _run_sampler(Reject("target_placement", "candidate 0: ..."))
    plain.run_instances(2, max_attempts=3, indices=[0, 1])
    assert len(calls) == 6 and "template_exhausted" not in plain.counters
