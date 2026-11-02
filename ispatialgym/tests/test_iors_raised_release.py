"""IORS small nextto, raised release (owner 2026-10-14 ~01:51, COMPLAINTS.md nextto#03: "try the higher release: lift the spoon about 2 cm
so the fingers open above the chopstick. The spoon then drops that far and may turn - but not too high - just so that the fingers don't
touch"; the release stays the open command +1): ``generation.raised_release`` raises the place over the resting goal to the lowest height at
which the fully open jaw clears the partner, the goal check drops the target from there and the settled pose is the recorded goal; the
oracle's place must not touch the partner. No simulator."""

from __future__ import annotations

import json
import math
import os
import types

import numpy as np
import pytest
import torch as th

from ispatialgym.oracle.actions import Place
from ispatialgym.oracle.programs import expand
from ispatialgym.oracle.sidecar import Sidecar
from ispatialgym.robot import conventions as C
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2.stances import GraspRecord
from ispatialgym.sampling.v2.tasks import iors_small as IS
from ispatialgym.tests.test_iors_pair_goal_align import _goal_sampler, _pose
from ispatialgym.tests.test_iors_start_region import _sampler

CUSTOMS = os.path.join(os.path.dirname(__file__), "..", "assets", "task_specs", "isg_customs.json")
SIDECARS = os.path.join(os.path.dirname(__file__), "data", "sidecars")
TOP_DOWN = C.eef_quat(th.tensor([0.0, 0.0, -1.0]), th.tensor([0.0, 1.0, 0.0]))  # approach down, the jaw opening along +-y


def _stick(near, table=0.75, r=0.0026, rings=(-0.12, -0.01, 0.10)):
    """A chopstick-like hull along x: rings of 12 points at its ends and middle only (as the chopstick's collision hull), lying on the
    table, its near side ``near`` m from x-axis along +y."""
    return th.tensor([[x, near + r + r * math.cos(a), table + r + r * math.sin(a)] for x in rings for a in [2 * math.pi * k / 12 for k in range(12)]],
                     dtype=th.float32)


def _top_down_grasp(z_tip_below_origin=0.0115):
    """A top-down pinch at the target's origin, the fingertips at the target's bottom (a 23 mm thick box hull centred on its origin)."""
    eio = (th.tensor([0.03, 0.0, -z_tip_below_origin + C.FINGERTIP_LENGTH]), TOP_DOWN)
    return GraspRecord("target", "right", "k", "g", 0.0, eef_in_object=eio)


def test_off_by_default_nextto_only_and_spec_errors():
    assert _sampler("nextto", gen={}).raised_release() is None
    assert _sampler("touching", gen={"raised_release": True}).raised_release() is None
    assert _sampler("nextto", gen={"raised_release": True}).raised_release() == IS.RAISED_RELEASE_DEFAULTS
    reg = _sampler("nextto", gen={"raised_release": {"max_raise_m": 0.02}}).raised_release()
    assert reg["max_raise_m"] == 0.02 and reg["partner_clearance_m"] == IS.RAISED_RELEASE_DEFAULTS["partner_clearance_m"]
    for bad in ({"height": 0.02}, "yes"):
        with pytest.raises(Reject) as e:
            _sampler("nextto", gen={"raised_release": bad}).raised_release()
        assert e.value.reason == "spec"


def test_hull_edge_points_put_points_where_a_slender_hull_has_none():
    """The chopstick's collision hull has vertex rings only at its ends and middle: the open jaw standing 4 cm from the middle ring sees no
    vertex, though the hull's edges pass under it."""
    s = IS.IORSSmall.__new__(IS.IORSSmall)
    stick = _stick(0.03, rings=(-0.12, 0.10))
    place = (th.tensor([0.0, 0.0, 0.75 + C.FINGERTIP_LENGTH + 0.002]), TOP_DOWN)
    assert s.open_jaw_clearance(place, stick) > 0.05  # the vertices: far along x, "clear"
    dense = IS.hull_edge_points(stick)
    assert len(dense) > 10 * len(stick) and s.open_jaw_clearance(place, dense) < 0  # its edges: inside the open jaw
    gaps = th.cdist(dense[None], dense[None])[0] + th.eye(len(dense))
    assert float(gaps.min(dim=1).values.max()) <= 0.003 + 1e-6


def test_release_raise_is_the_lowest_height_clearing_the_partner_by_the_margin(monkeypatch):
    """The fingertips at the table at rest; the stick 5.2 mm thick beside the target: the jaw clears it by 5 mm once the fingertips are 10.2
    mm above the table (1 mm steps); a cap below that rejects the draw."""
    s, manip = _goal_sampler(monkeypatch, "nextto03", g_nt=0.026, gen={"raised_release": True})
    grasp = _top_down_grasp()
    s.sidecar.grasp_of = lambda name: grasp
    pos = th.tensor([0.0, 0.0, 0.75 + 0.0115])
    stick = _stick(0.035)
    h = s.release_raise(pos, C.IDENTITY_QUAT.clone(), stick, s.raised_release())
    assert h == pytest.approx(0.011, abs=1e-6)  # 5.2 mm + 5 mm, the next 1 mm step
    f = s.raised_release_facts
    assert f["raise_m"] == h and f["jaw_clearance_m"] >= 0.005 and f["jaw_clearance_at_rest_m"] < 0 and f["partner_top_z"] == pytest.approx(0.7552, abs=1e-4)
    far = _stick(0.12)
    assert s.release_raise(pos, C.IDENTITY_QUAT.clone(), far, s.raised_release()) == 0.0  # nothing under the open jaw: no raise
    with pytest.raises(Reject) as e:
        s.release_raise(pos, C.IDENTITY_QUAT.clone(), stick, dict(s.raised_release(), max_raise_m=0.008))
    assert e.value.reason == "release_sweep"
    s.sidecar.grasp_of = lambda name: None
    assert s.release_raise(pos, C.IDENTITY_QUAT.clone(), stick, s.raised_release()) == 0.0


def test_goal_is_the_plain_nextto_goal_raised_and_without_the_key_unchanged(monkeypatch):
    """With the key the goal keeps the plain nextto gap (g_NT - 1 cm, no move along the bound) and rises by the raise; without it the goal
    rests on the table as before."""
    goals = {}
    for key in (None, True):
        s, manip = _goal_sampler(monkeypatch, "nextto03", g_nt=0.026, gen={} if key is None else {"raised_release": key})
        s.hull["partner"] = th.tensor([[x, y, z] for x in np.linspace(-0.123, 0.099, 3) for y in (-0.00435, 0.00435) for z in (-0.0026, 0.0026)], dtype=th.float32)
        grasp = _top_down_grasp()
        s.sidecar.grasp_of = lambda name, g=grasp: g
        if key is None:
            s.avoid_release_sweep = lambda st, depth, lat, q, *a, **k: (lat, q)  # the open-jaw rule of the plain release, not under test here
        t_pose = _pose(s, "target", s.pair_target_yaws(manip)[0], (1.31, -0.24))
        p_pose = s.place_partner(np.random.default_rng(3), t_pose, manip, None)
        goals[key] = (s.goal_pose(t_pose, p_pose), t_pose, getattr(s, "raised_release_facts", None))
    (g0, t0, _), (g1, t1, facts) = goals[None], goals[True]
    assert float(g0[0][2]) == pytest.approx(float(t0[0][2]), abs=1e-6)
    assert facts is not None and facts["raise_m"] > 0 and float(g1[0][2]) == pytest.approx(float(t1[0][2]) + facts["raise_m"], abs=1e-6)
    assert float(th.linalg.norm(g1[0][:2] - g0[0][:2])) < 1e-6 and G.quat_angle(g1[1], g0[1]) < 1e-6


def test_record_drop_and_reject_when_the_open_jaw_reaches_the_settled_partner(monkeypatch):
    """At the recorded place: the drop (release pose -> settled goal: height, slide, turn), the open jaw's clearance from the settled partner,
    and a reject when the partner settled under the open jaw."""
    s, manip = _goal_sampler(monkeypatch, "nextto03", g_nt=0.026, gen={"raised_release": True})
    grasp = _top_down_grasp()
    s.sidecar.grasp_of = lambda name: grasp
    rel = (th.tensor([0.0, 0.0, 0.75 + 0.0115 + 0.011]), C.IDENTITY_QUAT.clone())
    settled_q = G.quat_from_yaw(math.radians(2.0))
    draft = types.SimpleNamespace(goal_poses={"target": rel}, pose_targets={"target": {"position": [0.003, 0.004, 0.75 + 0.0115], "orientation": settled_q.tolist()}})
    s.release_raise(rel[0] - th.tensor([0.0, 0.0, 0.011]), rel[1], _stick(0.035), s.raised_release())
    place = IS.IK.grasp_pose(rel, grasp.eef_in_object)
    rec = s.raised_release_record(draft, "target", place, _stick(0.035), s.raised_release())
    assert rec["drop_m"] == pytest.approx(0.011, abs=1e-4) and rec["slide_m"] == pytest.approx(0.005, abs=1e-4)
    assert rec["turn_deg"] == pytest.approx(2.0, abs=0.01) and rec["yaw_deg"] == pytest.approx(2.0, abs=0.01)
    assert rec["jaw_clearance_at_place_m"] >= 0.005 and rec["raise_m"] == pytest.approx(0.011) and rec["key"] == s.raised_release()
    with pytest.raises(Reject) as e:  # the partner settled 8 mm higher, under the open jaw
        s.raised_release_record(draft, "target", place, _stick(0.035) + th.tensor([0.0, 0.0, 0.008]), s.raised_release())
    assert e.value.reason == "release_sweep"


def test_program_place_avoids_the_partner_only_for_a_raised_release():
    sc = Sidecar.load(os.path.join(SIDECARS, "inter_object_relation_satisfaction-nextto.json"))
    place = next(a for a in expand(sc.variant, sc) if isinstance(a, Place))
    assert place.kind == "support" and place.avoid == ()
    sc.grasps[0].extra["raised_release"] = {"raise_m": 0.013, "drop_m": 0.013}
    place = next(a for a in expand(sc.variant, sc) if isinstance(a, Place))
    assert place.kind == "support" and place.avoid == (sc.grasps[0].place_on,) and place.relation == "next to"


def test_nextto03_variant_alone_carries_the_key():
    customs = json.load(open(CUSTOMS))
    keyed = sorted(a for a, e in customs.items() if isinstance(e, dict) and (e.get("generation") or {}).get("raised_release") is not None)
    assert keyed == ["inter_object_relation_satisfaction-nextto-tablespoon-chopstick-breakfast_table"]
    gen = customs[keyed[0]]["generation"]
    assert set(gen["raised_release"]) <= set(IS.RAISED_RELEASE_DEFAULTS) and gen["pair_goal_align"]["facing"] == "away" and gen["init_align_support"] is True
