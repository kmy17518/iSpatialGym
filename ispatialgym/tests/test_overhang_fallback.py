"""i7-plateqtf2 (owner decision 2026-11-08 02:35 item 2): the overhang FALLBACK of a failed side grasp in the IORS small-scale draw - IORSLarge's
pick_overhang_push bound unchanged onto the small sampler (v2/overhang_fallback.py), fallback only (a side grasp, a support-policy failure),
counted in kit, recorded as contacts.push_pick; the small program pushes, retreats, then grasps. Simulator-free (the stubs of
test_iors_large_overhang)."""

import json
import os
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.oracle.programs import expand
from ispatialgym.oracle.sidecar import Sidecar
from ispatialgym.robot import conventions as C
from ispatialgym.sampling.v2 import overhang as OH
from ispatialgym.sampling.v2 import overhang_fallback as OF
from ispatialgym.sampling.v2.stances import SidecarDraft
from ispatialgym.tests import test_iors_large_overhang as T

POLICY = "support_policy: non-finger link touches the support"


class _Small:
    """What the IORSSmall draw hands the fallback: per-role hulls / rests / objects, the support top, the bindings, the sidecar."""

    def __init__(self, ikc=None, **gen):
        self.gp = T._GP(**gen)
        self.gen = self.gp.gen
        self._target = T._Target()
        self.world = T._World(self._target)
        self.top = SimpleNamespace(z=T.Z_TOP, contains=T._Top(T.EDGE_X).contains, name="countertop_top")
        self._support = SimpleNamespace(name="countertop_kelzer_0", category="countertop", links={})
        self.hull = {"target": T._plate_hull(), "partner": T._plate_hull()}
        self.rest = {"target": C.IDENTITY_QUAT.clone()}
        self.ikc = ikc or T._IKC()
        self.kit = SimpleNamespace(sync_world=lambda: None, stats={}, robot=SimpleNamespace(arm_control_idx={"right": th.arange(7), "left": th.arange(7, 14)}))
        self.robot = SimpleNamespace(links={})
        self.choice = SimpleNamespace(models={"target": SimpleNamespace(scale=None, category="plate", model="qtfzeq")})
        names = {"target": "plate.n.04_1", "support": "countertop.n.01_1"}
        self.bindings = SimpleNamespace(has=lambda role: role in names, one=lambda role: SimpleNamespace(name=names[role]))
        self.sidecar = SidecarDraft(f"{T.SUBTASK}-plate-drop_in_sink-countertop", "inter_object_relation_satisfaction")

    def obj(self, role):
        return self._target

    def support_objects(self):
        return [self._support]

    def camera_pose_at(self, st):
        return None


def _side_g(mode="side"):
    return SimpleNamespace(name="side_9", key="plate__qtfzeq", pose=(T.G5_POS, T.G5_QUAT), meta={"width": T.G5_WIDTH, "mode": mode}, finger_q=[0.004, 0.004])


@pytest.fixture
def setup(monkeypatch):
    monkeypatch.delenv(OF.ENV, raising=False)
    T._first_passing_choose(monkeypatch)
    s = _Small(ikc=T._IKC(grasp_ok=T._grasp_needs(T._first_rung())))
    return s, T._manip(), T._plate_pose()


def test_reason_classes_and_side_grasps():
    assert OF.reason_class(POLICY) == "non_finger_link"
    assert OF.reason_class("approach_untrackable: support policy on the approach tail (non-finger link touches the support, 0.018 m") == "non_finger_link"
    assert OF.reason_class("support_policy: palm_contact on breakfast_table_0") == "palm_contact"
    assert OF.reason_class("support_policy: finger_penetration on breakfast_table_0") == "finger_penetration"
    for fixable in ("exact IK failed at pose 2 of 2", "ik_joint_limit: right_arm_joint6", "collides with place_mat_0", "approach_untrackable: joint jump 0.21 rad"):
        assert OF.reason_class(fixable) is None
    assert OF.is_side_grasp(_side_g()) and not OF.is_side_grasp(_side_g("top")) and not OF.is_side_grasp(SimpleNamespace(name="g"))
    o = OF.options(SimpleNamespace(gen=lambda k, d=None: None))
    assert o["mode"] == "fallback" and o["grasp_back_offs_m"] == [] and o["finger_clearance_m"] == OH.DEFAULTS["finger_clearance_m"]


def test_side_grasp_policy_failure_falls_back_to_the_overhang_pick(setup):
    s, manip, pose = setup
    out = OF.try_fallback(s, manip, "right", _side_g(), pose, POLICY, False, "plate.n.04_1", lambda g: g.pose)
    assert out is not None
    t_pose, rec = out
    assert th.allclose(t_pose[0], pose[0]) and rec.arm == "right" and rec.grasp_name == "side_9" and rec.object == "plate.n.04_1"
    ov = rec._overhang
    # IORSLarge's own checks ran: the push to the edge, the grasp at the pushed pose with the support present, the push chain, the settle
    assert ov["facts"]["push_m"] == pytest.approx(T._first_rung(), abs=1e-4) and ov["facts"]["com_inside_m"] >= 0.05
    eef = C.grasp_pose_from_transform(ov["post_pose"], (T.G5_POS, T.G5_QUAT))
    assert float(th.linalg.norm(rec.grasp[0] - eef[0])) < 1e-5 and float(th.linalg.norm(rec.pre_grasp[0] - eef[0])) > 0.05
    assert all(c["ignore_for_last"] == [] for c in s.ikc.calls if c["n"] == 2) and any(c["n"] == 4 for c in s.ikc.calls)
    assert s.world.events[:4][0] == "snapshot" and "settle" in s.world.events
    assert rec.extras["overhang_fallback"]["trigger"] == "non_finger_link"
    st = s.kit.stats
    assert st["overhang_fallback.trigger.non_finger_link"] == 1 and st["overhang_fallback.tried"] == 1 and st["overhang_fallback.found"] == 1
    assert st["pick_overhang.tried"] == 1 and st["pick_overhang.found"] == 1
    # the records: contacts.push_pick + grasps[0].pick_overhang, removed again with a rejected candidate
    assert OF.record(s, rec)
    c = s.sidecar.contacts[OH.CONTACT_SLOT].to_sidecar()
    assert c["kind"] == "pick_overhang" and c["object"] == "plate.n.04_1" and c["support"] == "countertop.n.01_1" and c["distance_m"] == pytest.approx(ov["push_m"])
    assert rec.extras["pick_overhang"]["contact"] == "push_pick" and rec.extras["ignore_last"] == []
    json.dumps(rec.extras)
    json.dumps(OF.summary(rec), default=float)
    OF.drop_records(s)
    assert OH.CONTACT_SLOT not in s.sidecar.contacts
    assert not OF.record(s, SimpleNamespace(extras={}))  # a direct grasp records nothing


def test_fallback_only_for_side_grasps_and_unfixable_reasons(setup, monkeypatch):
    s, manip, pose = setup
    eof = lambda g: g.pose  # noqa: E731
    assert OF.try_fallback(s, manip, "right", _side_g("top"), pose, POLICY, False, "p", eof) is None  # not a side grasp
    assert OF.try_fallback(s, manip, "right", _side_g(), pose, "exact IK failed at pose 2 of 2", False, "p", eof) is None  # the sampler can fix it
    assert s.kit.stats == {}
    monkeypatch.setenv(OF.ENV, "0")
    assert OF.try_fallback(s, manip, "right", _side_g(), pose, POLICY, False, "p", eof) is None and s.kit.stats == {}


def test_a_failed_overhang_pick_is_remembered(setup):
    s, manip, pose = setup
    s.ikc = T._IKC(grasp_ok=lambda p: False)  # the grasp fails at every rung
    assert OF.try_fallback(s, manip, "right", _side_g(), pose, POLICY, False, "p", lambda g: g.pose) is None
    assert OF.try_fallback(s, manip, "right", _side_g(), pose, POLICY, False, "p", lambda g: g.pose) is None
    assert s.kit.stats["overhang_fallback.failed"] == 1 and s.kit.stats["overhang_fallback.memo_hit"] == 1 and s.kit.stats["overhang_fallback.tried"] == 1


def test_small_sampler_offers_the_fallback_except_for_touching(monkeypatch):
    from ispatialgym.sampling.v2.tasks import iors_small as IS

    monkeypatch.delenv(OF.ENV, raising=False)
    prop = IS.IORSSmall.overhang_fallback_enabled
    assert prop.fget(SimpleNamespace(relation="under")) and prop.fget(SimpleNamespace(relation="inside"))
    assert not prop.fget(SimpleNamespace(relation="touching"))
    monkeypatch.setenv(OF.ENV, "0")
    assert not prop.fget(SimpleNamespace(relation="under"))


def _small_sidecar(with_push: bool):
    """The under small-scale fixture with (or without) the overhang pick's push record."""
    d = json.load(open(os.path.join(os.path.dirname(T.__file__), "data", "sidecars", "inter_object_relation_satisfaction-under_small_scale.json")))
    if not with_push:
        return d
    g = d["grasps"][0]
    post = {"position": [float(v) + (0.06 if i == 0 else 0.0) for i, v in enumerate(g["grasp"]["position"])], "orientation": g["grasp"]["orientation"]}
    g["pick_overhang"] = {"contact": "push_pick", "post_push_pose": post, "overhang_m": 0.025, "push_m": 0.06, "finger_clearance_m": 0.01, "yaw_free": True}
    g["ignore_last"], g["ignore_every"] = [], []
    q = [0.70711, 0.70711, 0.0, 0.0]
    p0 = g["grasp"]["position"]
    pt = lambda dx, dz: {"position": [p0[0] + dx, p0[1], p0[2] + dz], "orientation": q}  # noqa: E731
    sup = next(iter(k for k in d["language"]["objects"] if "table" in k), None) or "breakfast_table.n.01_1"
    d["contacts"] = {"push_pick": {"arm": g["arm"], "object": g["object"], "pre": pt(-0.1, 0.1), "init": pt(-0.1, 0.0), "mid": pt(-0.07, 0.0), "goal": pt(-0.04, 0.0),
                                   "direction": [1.0, 0.0, 0.0], "distance_m": 0.06, "stance": "manipulation", "kind": "pick_overhang", "support": sup,
                                   "fingers": "open", "fingers_opening_m": 0.05, "post_push_pose": post, "ignore_every": [g["object"]], "ignore_last": [],
                                   "overhang_m": 0.025}}
    d["language"]["objects"].setdefault(sup, {"display_name": "table", "descriptor": "the table"})
    return d


def test_small_program_pushes_retreats_then_grasps():
    base = [a.name for a in expand(Sidecar.from_dict(_small_sidecar(False)).variant, Sidecar.from_dict(_small_sidecar(False)))]
    sc = Sidecar.from_dict(_small_sidecar(True))
    acts = expand(sc.variant, sc)
    names = [a.name for a in acts]
    i = base.index("grasp")
    assert names == base[:i] + ["push", "retreat"] + base[i:]  # the push and its retreat before the grasp, the rest unchanged
    push, retreat = acts[i], acts[i + 1]
    assert push.kind == "pick_overhang" and push.contact.distance_m == pytest.approx(0.06) and retreat.after is push
