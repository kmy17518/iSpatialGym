"""Attachment by contact (owner 2026-10-10): the pair collides until joined; only the task's attach rule joins it - released, in
PhysX contact with the partner, aligned modulo the part's symmetry within tight bounds - in place or after a snap onto the seat,
filtering the pair's collision in that instant; the oracle's attach place hovers above the seat, aligns laterally and in yaw, and
opens in one step."""
from __future__ import annotations

import contextlib
import math
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.oracle import execute as EX
from ispatialgym.sampling import predicates as PRED


class _Obj:
    def __init__(self, name, pos, links=None):
        self.name = name
        self.pos = th.tensor(pos, dtype=th.float32)
        self.quat = th.tensor([0.0, 0.0, 0.0, 1.0])
        self.links = links or {}
        self.moves = []

    def get_position_orientation(self):
        return self.pos.clone(), self.quat.clone()

    def set_position_orientation(self, position, orientation):
        self.pos, self.quat = th.as_tensor(position).float(), th.as_tensor(orientation).float()
        self.moves.append((self.pos.clone(), self.quat.clone()))

    def keep_still(self):
        pass


def _rule(monkeypatch, *, lateral=0.001, vertical=0.010, rot=0.01, held=False, contact=True):
    events = []
    rec = {"ok": True, "pos_m": 0.0101, "rot_rad": rot, "rot_raw_rad": 0.4, "equivalence": 3, "child_link": "M", "parent_link": "F",
           "first_free_pair": ["M", "F"], "fix": (th.tensor([0.0, 0.0, -0.010]), th.tensor([0.0, 0.0, 0.0, 1.0]))}
    monkeypatch.setattr(PRED, "attachment_alignment", lambda c, p, eq=None, with_fix=False: dict(rec))
    monkeypatch.setattr(PRED, "seat_offset", lambda c, p, cl, pl: {"lateral_m": lateral, "vertical_m": vertical})
    monkeypatch.setattr(PRED, "held_by_robot", lambda o: held)
    monkeypatch.setattr(PRED, "pair_in_contact", lambda c, p: contact)
    monkeypatch.setattr(PRED, "attach_in_place", lambda c, p, cl, pl, can_joint_break=True: events.append(("joint", c.name, p.name, cl, pl, c.pos[2].item())) or True)
    monkeypatch.setattr(PRED, "set_pair_collision_filter", lambda c, p, f: events.append(("filter", c.name, p.name, f)) or True)
    monkeypatch.setattr(PRED, "JOINS", [])
    for name in ("_RELEASED_TOUCHING", "_PENDING_FILTER", "_PENDING_UNFILTER"):
        monkeypatch.setattr(PRED, name, set())
    return _Obj("cap_0", [1.0, 2.0, 1.08]), _Obj("bottle_0", [1.0, 2.0, 0.93]), events


def test_join_needs_release_contact_and_tight_alignment(monkeypatch):
    monkeypatch.setenv("ISG_ATTACH_JOIN_MODE", "in_place")
    child, parent, events = _rule(monkeypatch)
    a = PRED.attach_if_aligned(child, parent)
    assert a["attached"] and a["joint"] == "in_place" and a["contact"] and a["trigger"] == "per_step"
    # the collision is filtered in the same instant, the joint at the current pose (nothing moved)
    assert events == [("filter", "cap_0", "bottle_0", True), ("joint", "cap_0", "bottle_0", "M", "F", pytest.approx(1.08))] and not child.moves
    assert [{k: v for k, v in j.items() if k != "sim_time"} for j in PRED.JOINS] == [
        {"child": "cap_0", "parent": "bottle_0", "trigger": "per_step", "mode": "in_place", "released": False, "contact": True,
         "lateral_m": 0.001, "vertical_m": 0.010, "rot_rad": 0.01, "pos_m": 0.0101}]
    for kw, why in (({"lateral": 0.006}, "not aligned"), ({"rot": 0.12}, "not aligned"), ({"vertical": PRED.JOIN_VERTICAL_MAX_M + 0.002}, "not aligned"),
                    ({"vertical": PRED.JOIN_VERTICAL_MIN_M - 0.002}, "not aligned"), ({"held": True}, "held by a robot"),
                    ({"contact": False}, "not in contact with the partner")):
        child, parent, events = _rule(monkeypatch, **kw)
        a = PRED.attach_if_aligned(child, parent)
        assert not a["attached"] and a["reason"] == why and not events, kw


def test_release_hook_joins_only_a_part_that_already_touches(monkeypatch):
    """At the release instant the in-hand registry still names the part (``released``). A part released above its partner is not
    joined there - the per-step check joins it once it lands; a part that already touches is joined by that step's own check
    after its physics (the filter restarts the physics, which must not happen while the step's actions are applied), still held
    in the registry and with the trigger ``release_hook``."""
    child, parent, events = _rule(monkeypatch, held=True, contact=False)
    monkeypatch.setattr(PRED, "_RELEASED_TOUCHING", set())
    scene = SimpleNamespace(object_registry=lambda key, name: {"bottle_0": parent}.get(name))
    child.scene = scene
    PRED.set_attachment_context([{"child": "cap_0", "parent": "bottle_0", "equivalences": []}])
    try:
        rec = PRED.attach_on_release(None, "right", child)
        assert not rec["attached"] and rec["reason"] == "not in contact with the partner" and rec["trigger"] == "release_hook" and not events
        assert not PRED._RELEASED_TOUCHING
        monkeypatch.setattr(PRED, "pair_in_contact", lambda c, p: True)
        rec = PRED.attach_on_release(None, "right", child)
        assert not rec["attached"] and rec["reason"].startswith("touching at the release") and not events and PRED._RELEASED_TOUCHING == {("cap_0", "bottle_0")}
        rec = PRED.attach_if_aligned(child, parent)  # the step's own check (the registry still names the part)
        assert rec["attached"] and PRED.JOINS[-1]["trigger"] == "release_hook" and PRED.JOINS[-1]["released"] and not PRED._RELEASED_TOUCHING
    finally:
        PRED.set_attachment_context([])


def test_deferred_filter_work_runs_at_the_next_check(monkeypatch):
    """A rollback's detach (inside a state load) restores the pair's collision at the next per-step check, before the rule."""
    child, parent, events = _rule(monkeypatch, contact=False)
    monkeypatch.setattr(PRED, "_PENDING_UNFILTER", {("cap_0", "bottle_0")})
    a = PRED.attach_if_aligned(child, parent)
    assert not a["attached"] and events == [("filter", "cap_0", "bottle_0", False)] and not PRED._PENDING_UNFILTER


def test_snap_moves_the_part_onto_its_seat_then_joins(monkeypatch):
    monkeypatch.setenv("ISG_ATTACH_JOIN_MODE", "snap")
    child, parent, events = _rule(monkeypatch)
    a = PRED.attach_if_aligned(child, parent)
    assert a["attached"] and a["joint"] == "snap" and a["snap"] == {"pos_m": 0.0101, "rot_rad": 0.01, "lateral_m": 0.001, "vertical_m": 0.010}
    # moved by the alignment's fix (10 mm down onto the seat) before the filter and the joint
    assert len(child.moves) == 1 and child.pos[2].item() == pytest.approx(1.07)
    assert [e[0] for e in events] == ["filter", "joint"] and events[1][-1] == pytest.approx(1.07)
    assert PRED.JOINS[-1]["snap"]["vertical_m"] == 0.010
    monkeypatch.delenv("ISG_ATTACH_JOIN_MODE")
    assert PRED.join_mode() == "snap"
    monkeypatch.setenv("ISG_ATTACH_JOIN_MODE", "glue")
    with pytest.raises(ValueError):
        PRED.join_mode()


def test_seat_axis_is_the_female_z_unless_it_points_away_from_the_partner(monkeypatch):
    """The sampler's insertion-axis rule: the female link's +z when within 45 deg of the direction from the partner's hull centroid
    to the female link, else that direction (the cork / wine bottle frames: +z along the bottle's -y)."""
    from omnigibson.object_states import AttachedTo

    monkeypatch.setattr(PRED, "_SEAT_AXIS_LOCAL", {})
    monkeypatch.setattr(PRED, "_boundary_points", lambda obj: _pts(-0.04, 0.04, -0.04, 0.04, 0.70, 0.90))  # centroid at z 0.80
    s2 = math.sqrt(0.5)

    def bottle(name, quat):
        link = SimpleNamespace(get_position_orientation=lambda: (th.tensor([0.0, 0.0, 0.93]), th.tensor(quat)))
        return SimpleNamespace(name=name, states={AttachedTo: SimpleNamespace(links={"F": link})})

    up = PRED.seat_axis(SimpleNamespace(name="cap_0"), bottle("lemonade_0", [0.0, 0.0, 0.0, 1.0]), "F")
    assert up.tolist() == pytest.approx([0.0, 0.0, 1.0], abs=1e-9)
    side = bottle("wine_0", [s2, 0.0, 0.0, s2])  # +z turned onto -y
    a = PRED.seat_axis(SimpleNamespace(name="cork_0"), side, "F")
    assert a.tolist() == pytest.approx([0.0, 0.0, 1.0], abs=1e-6)
    # fixed in the female frame: the partner tipped by 10 deg about x carries the axis along
    c, s = math.cos(math.radians(5.0)), math.sin(math.radians(5.0))
    q = PRED._qmul(th.tensor([s, 0.0, 0.0, c], dtype=th.float64), th.tensor([s2, 0.0, 0.0, s2], dtype=th.float64))
    side.states[AttachedTo].links["F"].get_position_orientation = lambda: (th.tensor([0.0, 0.0, 0.93]), q.float())
    a = PRED.seat_axis(SimpleNamespace(name="cork_0"), side, "F")
    assert a.tolist() == pytest.approx([0.0, -math.sin(math.radians(10.0)), math.cos(math.radians(10.0))], abs=1e-5)


class _Rel:
    def __init__(self):
        self.targets = set()

    def AddTarget(self, p):
        self.targets.add(p)

    def RemoveTarget(self, p):
        self.targets.discard(p)


class _Sim:
    def __init__(self, playing=False):
        self._editing_usd = False
        self.blocks = 0
        self.playing = playing
        self.calls = []

    def is_playing(self):
        return self.playing

    def dump_state(self):
        self.calls.append("dump")
        return "S"

    def stop(self):
        self.calls.append("stop")
        self.playing = False

    def play(self):
        self.calls.append("play")
        self.playing = True

    def load_state(self, state):
        self.calls.append(("load", state))

    @contextlib.contextmanager
    def editing_usd(self):
        assert not self._editing_usd, "nested editing_usd"
        assert not self.playing, "edited while PhysX plays"
        self._editing_usd = True
        self.blocks += 1
        self.calls.append("edit")
        try:
            yield
        finally:
            self._editing_usd = False


def test_pair_filter_edits_both_links_in_one_block_and_is_idempotent(monkeypatch):
    import omnigibson as og

    rels = {}
    sim = _Sim()
    monkeypatch.setattr(og, "sim", sim)

    def rel(link):
        assert sim._editing_usd, "the schema is touched inside the block only"
        return rels.setdefault(link.prim_path, _Rel())

    monkeypatch.setattr(PRED, "_filtered_pairs_rel", rel)
    monkeypatch.setattr(PRED, "_FILTERED", set())
    link = lambda p: SimpleNamespace(prim_path=p)  # noqa: E731
    child = SimpleNamespace(name="lid_0", links={"base": link("/lid/base"), "M": link("/lid/M")})
    parent = SimpleNamespace(name="teapot_0", links={"base": link("/pot/base")})
    assert PRED.set_pair_collision_filter(child, parent, True) and sim.blocks == 1
    assert rels["/lid/base"].targets == {"/pot/base"} and rels["/lid/M"].targets == {"/pot/base"} and rels["/pot/base"].targets == {"/lid/base", "/lid/M"}
    assert not PRED.set_pair_collision_filter(child, parent, True) and sim.blocks == 1
    sim._editing_usd = True  # already inside a block: edited there, no nested block
    assert PRED.set_pair_collision_filter(child, parent, False) and sim.blocks == 1
    assert all(not r.targets for r in rels.values())
    # while PhysX plays: the stock route - dump the state, stop, edit, play, load the state
    sim2 = _Sim(playing=True)
    monkeypatch.setattr(og, "sim", sim2)
    assert PRED.set_pair_collision_filter(child, parent, True)
    assert sim2.calls == ["dump", "stop", "edit", "play", ("load", "S")] and rels["/pot/base"].targets == {"/lid/base", "/lid/M"}


def test_attached_to_overrides_leave_registered_pairs_to_the_rule(monkeypatch):
    import omnigibson.object_states as OS

    calls = []

    class FakeAT:
        def __init__(self, obj, parent=None):
            self.obj, self.parent = obj, parent

        def _update(self):
            calls.append(("stock_update", self.obj.name))

        def _attach(self, other, child_link, parent_link, joint_type=None, can_joint_break=True):
            calls.append(("stock_attach", self.obj.name))

        def _detach(self):
            calls.append(("stock_detach", self.obj.name))
            self.parent = None

    monkeypatch.setattr(OS, "AttachedTo", FakeAT)
    monkeypatch.setattr(PRED, "_OVERRIDES", {})
    monkeypatch.setattr(PRED, "set_pair_collision_filter", lambda c, p, f: calls.append(("filter", c.name, p.name, f)) or True)
    monkeypatch.setattr(PRED, "attach_in_place", lambda c, p, cl, pl, can_joint_break=True: calls.append(("in_place", c.name, p.name, cl, pl)) or True)
    male, female = object(), object()
    cap, bottle, apple = (SimpleNamespace(name="cap_0", links={"M": male}), SimpleNamespace(name="bottle_0", links={"F": female}),
                          SimpleNamespace(name="apple_0", links={}))
    PRED.set_attachment_context([{"child": "cap_0", "parent": "bottle_0", "equivalences": []}])
    try:
        assert PRED._OVERRIDES  # installed by the registration
        FakeAT(cap)._update()
        FakeAT(bottle)._update()
        FakeAT(apple)._update()
        assert calls == [("stock_update", "apple_0")]
        calls.clear()
        # a state load restoring an attached checkpoint: joined in place, the filter left to the next per-step check (no physics
        # restart inside a state load)
        monkeypatch.setattr(PRED, "_FILTERED", set())
        monkeypatch.setattr(PRED, "_PENDING_FILTER", set())
        monkeypatch.setattr(PRED, "_PENDING_UNFILTER", set())
        FakeAT(cap)._attach(bottle, male, female)
        FakeAT(apple)._attach(bottle, male, female)
        assert calls == [("in_place", "cap_0", "bottle_0", "M", "F"), ("stock_attach", "apple_0")] and PRED._PENDING_FILTER == {("cap_0", "bottle_0")}
        calls.clear()
        PRED._FILTERED.add(("cap_0", "bottle_0"))
        FakeAT(cap, parent=bottle)._detach()  # a state load rolling back to before the join
        assert calls == [("stock_detach", "cap_0")] and PRED._PENDING_UNFILTER == {("cap_0", "bottle_0")} and not PRED._PENDING_FILTER
    finally:
        PRED.set_attachment_context([])


# ---------------------------------------------------------------------------------------------------------------------- oracle


def _pts(x0, x1, y0, y1, z0, z1):
    return th.tensor([[x, y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], dtype=th.float32)


def _pose(p, q=(0.0, 0.0, 0.0, 1.0)):
    return EX.Pose(position=list(p), orientation=list(q))


def test_hover_raises_the_seated_hand_until_part_and_fingertips_clear_the_partner(monkeypatch):
    """The hand 10 cm above its seated pose: the part's lowest point would be 1 cm into the partner at the seat, the fingertips 1.5
    cm; a far convex of the partner (its handle, 1.2 m high) is not under them. The hover clears the fingertips by 3 mm."""
    monkeypatch.setattr(EX, "_seat_axis", lambda state, action: th.tensor([0.0, 0.0, 1.0]))
    female = SimpleNamespace(get_position_orientation=lambda: (th.tensor([0.0, 0.0, 0.95]), th.tensor([0.0, 0.0, 0.0, 1.0])))
    part = _Obj("cap_0", [0.0, 0.0, 1.07], links={"base": SimpleNamespace(collision_boundary_points_world=_pts(-0.02, 0.02, -0.02, 0.02, 1.06, 1.08))})
    meshes = {"neck": SimpleNamespace(points_in_world_frame=_pts(-0.015, 0.015, -0.015, 0.015, 0.80, 0.97)),
              "handle": SimpleNamespace(points_in_world_frame=_pts(0.20, 0.25, -0.01, 0.01, 0.90, 1.20))}
    partner = _Obj("bottle_0", [0.0, 0.0, 0.85], links={"base": SimpleNamespace(collision_meshes=meshes)})
    finger = SimpleNamespace(collision_boundary_points_world=_pts(-0.03, 0.03, -0.005, 0.005, 1.055, 1.09))
    state = SimpleNamespace(object=lambda n: part if n == "cap_0" else partner, attachment_links=lambda c, p: (None, female),
                            eef_pose=lambda arm: (th.tensor([0.0, 0.0, 1.10]), th.tensor([0.0, 0.0, 0.0, 1.0])))
    motions = SimpleNamespace(robot=SimpleNamespace(finger_links={"right": [finger]}))
    action = SimpleNamespace(object="cap_0", on="bottle_0", arm="right")
    targets = {"place": _pose([0.0, 0.0, 1.00]), "pre_place": _pose([0.0, 0.0, 1.10]), "descent": [_pose([0.0, 0.0, 1.00])],
               "retreat": [_pose([0.0, 0.0, 1.05]), _pose([0.0, 0.0, 1.15])]}
    ctx = {}
    out = EX.attach_hover_targets(action, motions, state, targets, ctx, EX.ExecuteConfig())
    rec = ctx["attach"]["hover"]
    h = 0.97 + 0.003 - 0.955
    assert rec["applied"] and rec["height_above_seat_m"] == pytest.approx(h, abs=1e-4) and rec["lowest"] == "fingertips"
    assert rec["finger_clearance_m"] == pytest.approx(0.003, abs=1e-4) and rec["part_clearance_m"] == pytest.approx(0.008, abs=1e-4)
    assert rec["partner_top_z"] == pytest.approx(0.97) and rec["partner_convexes_under"] == 1
    assert out["place"].position[2] == pytest.approx(1.0 + h, abs=1e-5) and out["descent"] == [out["place"]]
    assert [w.position[2] for w in out["retreat"]] == pytest.approx([1.05 + h, 1.15 + h], abs=1e-5) and targets["place"].position[2] == 1.0


def test_hover_alignment_moves_the_hand_across_the_axis_and_turns_it_about_the_axis(monkeypatch):
    offs = [{"lateral": th.tensor([0.003, 0.0, 0.0]), "lateral_m": 0.003, "vertical_m": 0.013, "yaw_rad": 0.05, "rot_rad": 0.05,
             "male": th.tensor([0.0, 0.0, 1.10]), "axis": th.tensor([0.0, 0.0, 1.0])},
            {"lateral": th.tensor([0.0005, 0.0, 0.0]), "lateral_m": 0.0005, "vertical_m": 0.013, "yaw_rad": 0.004, "rot_rad": 0.004,
             "male": th.tensor([0.003, 0.0, 1.10]), "axis": th.tensor([0.0, 0.0, 1.0])}]
    events = []

    def measure(state, action):
        events.append("measure")
        return offs.pop(0) if len(offs) > 1 else offs[0]

    monkeypatch.setattr(EX, "_hover_offset", measure)
    monkeypatch.setattr(EX, "attach_correction_regime", lambda ctx, cfg: EX.AttachCorrection(soft_pos_tol_m=0.0015, max_passes=4, target_pos_m=0.002, target_rot_rad=0.02))
    legs = []

    def arm_along(arm, poses, **kw):
        legs.append(poses)
        events.append("leg")
        yield th.zeros(1)

    def settle(max_steps=None):
        events.append(("settle", max_steps))
        yield th.zeros(1)

    motions = SimpleNamespace(arm_along=arm_along, settle=settle)
    state = SimpleNamespace(eef_pose=lambda arm: (th.tensor([0.1, 0.0, 1.2]), th.tensor([0.0, 0.0, 0.0, 1.0])))
    ctx = {}
    list(EX._align_at_hover(SimpleNamespace(arm="right", object="cap_0", on="bottle_0"), motions, state, ctx, EX.ExecuteConfig()))
    assert events == [("settle", 20), "measure", "leg", ("settle", 20), "measure"]  # the arrival settles before the first measurement
    assert ctx["attach"]["alignment"]["settle_steps"] == 1
    assert len(legs) == 1
    p, q = legs[0][0]
    c, s = math.cos(0.05), math.sin(0.05)
    assert p.tolist() == pytest.approx([0.1 * c + 0.003, 0.1 * s, 1.2], abs=1e-6)  # turned about the male point, moved onto the axis, height kept
    assert q.tolist() == pytest.approx([0.0, 0.0, math.sin(0.025), math.cos(0.025)], abs=1e-6)
    rec = ctx["attach"]["alignment"]
    assert len(rec["passes"]) == 1 and rec["after"]["lateral_m"] == pytest.approx(0.0005) and rec["regime"]["max_passes"] == 4


def _touchdown(monkeypatch, *, contact_after=None, raise_after=None, steps=10):
    """A hand lowered 0.5 mm per streamed step; the pair touches from ``contact_after`` steps on (None: never); the leg raises
    after ``raise_after`` steps (None: never)."""
    seen = {"legs": [], "holds": [], "k": 0}
    eef = {"p": th.tensor([0.0, 0.0, 1.10])}

    def arm_along(arm, poses, speed=None, converge=True, soft_pos_tol=None, limit_margin=0.0, **kw):
        seen["legs"].append({"poses": poses, "speed": speed, "converge": converge})
        for i in range(steps):
            if raise_after is not None and i == raise_after:
                raise EX.SkillError("execution", "diff-IK stalled")
            eef["p"] = eef["p"] - th.tensor([0.0, 0.0, 0.0005])
            seen["k"] += 1
            yield th.zeros(1)

    def hold(n=1):
        seen["holds"].append(n)
        for _ in range(n):
            yield th.zeros(1)

    # no gripper and no command state: the touchdown never moves the fingers
    motions = SimpleNamespace(arm_along=arm_along, hold=hold)
    monkeypatch.setattr(EX, "_seat_axis", lambda state, action: th.tensor([0.0, 0.0, 1.0]))
    monkeypatch.setattr(PRED, "pair_in_contact", lambda c, p: contact_after is not None and seen["k"] >= contact_after)
    state = SimpleNamespace(object=lambda n: SimpleNamespace(name=n), eef_pose=lambda arm: (eef["p"].clone(), th.tensor([0.0, 0.0, 0.0, 1.0])))
    ctx = {"attach": {"hover": {"part_clearance_m": 0.003}}}
    return motions, state, ctx, seen


def test_touchdown_comes_down_the_seat_axis_until_the_part_touches_its_partner(monkeypatch):
    motions, state, ctx, seen = _touchdown(monkeypatch, contact_after=3)
    cfg = EX.ExecuteConfig()
    list(EX._touch_down(SimpleNamespace(arm="right", object="cap_0", on="bottle_0"), motions, state, ctx, cfg))
    rec = ctx["attach"]["touchdown"]
    leg = seen["legs"][0]
    assert leg["converge"] is False and leg["speed"] == cfg.attach_touchdown_speed_m_s
    assert leg["poses"][0][0].tolist() == pytest.approx([0.0, 0.0, 1.10 - 0.003 - cfg.attach_touchdown_extra_m])  # at most clearance + extra
    assert rec["touched"] and rec["steps"] == 3 and rec["fallback"] is None and rec["contact_at_release"]
    assert seen["holds"] == [cfg.attach_touchdown_hold_steps] and rec["lowered_m"] == pytest.approx(0.0015)


def test_a_failed_touchdown_releases_where_the_hand_is(monkeypatch):
    action = SimpleNamespace(arm="right", object="cap_0", on="bottle_0")
    motions, state, ctx, seen = _touchdown(monkeypatch, raise_after=2)
    list(EX._touch_down(action, motions, state, ctx, EX.ExecuteConfig()))
    rec = ctx["attach"]["touchdown"]
    assert not rec["touched"] and rec["steps"] == 2 and rec["fallback"].startswith("the touchdown leg failed (diff-IK stalled)") and seen["holds"] == [3]
    motions, state, ctx, seen = _touchdown(monkeypatch, steps=14)
    list(EX._touch_down(action, motions, state, ctx, EX.ExecuteConfig()))
    rec = ctx["attach"]["touchdown"]
    assert not rec["touched"] and rec["steps"] == 14 and rec["fallback"].startswith("no contact within 7.0 mm")
    motions, state, ctx, seen = _touchdown(monkeypatch, contact_after=0)
    list(EX._touch_down(action, motions, state, ctx, EX.ExecuteConfig()))
    assert ctx["attach"]["touchdown"]["touched"] and ctx["attach"]["touchdown"]["steps"] == 0 and not seen["legs"]


def test_release_into_join_opens_at_once_and_counts_the_steps_to_contact_and_joint(monkeypatch):
    seen = {"instant": None, "k": 0}

    def gripper(arm, close, instant=False):
        seen["instant"] = instant
        yield th.zeros(1)

    def hold(n=1):
        for _ in range(n):
            yield th.zeros(1)

    motions = SimpleNamespace(gripper=gripper, hold=hold, backend=SimpleNamespace(detach_object=lambda arm: None))
    k = {"n": 0}
    monkeypatch.setattr(PRED, "JOINS", [])

    def attached(child, parent):
        if k["n"] >= 4 and not PRED.JOINS:
            PRED.JOINS.append({"trigger": "per_step", "mode": "in_place", "contact": True})
        return k["n"] >= 4

    def contact(child, parent):
        k["n"] += 1
        return k["n"] >= 2

    monkeypatch.setattr(PRED, "pair_in_contact", contact)
    monkeypatch.setattr(EX, "_hover_offset", lambda state, action: {"vertical_m": 0.013 if k["n"] < 2 else 0.0102, "lateral_m": 0.0005, "rot_rad": 0.004})
    state = SimpleNamespace(object=lambda n: SimpleNamespace(name=n), held_object=lambda arm: None, attached=attached)
    ctx = {}
    list(EX._release_into_join(SimpleNamespace(arm="right", object="cap_0", on="bottle_0"), motions, state, ctx, EX.ExecuteConfig()))
    rec = ctx["attach"]["release"]
    assert seen["instant"] is True and rec["instant"] and rec["attached"]
    assert rec["steps_to_first_contact"] == 2 and rec["steps_to_join"] == 4 and rec["join"]["trigger"] == "per_step"
    assert rec["bounce_m"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------------------------------------------------- sampling / loading


@pytest.mark.parametrize("which", ["v2", "v1"])
def test_sampler_goal_check_filters_the_pair_before_seating_the_part(monkeypatch, which):
    from omnigibson.object_states import AttachedTo

    from ispatialgym.sampling.tasks import base as B

    if which == "v2":
        from ispatialgym.sampling.v2.tasks import attachment as M

        cls = M.Attachment
    else:
        from ispatialgym.sampling.tasks import attachment as M

        cls = M.AttachmentSampler
    order = []
    monkeypatch.setattr(B.TaskSampler, "apply_goal", lambda self, draft: order.append("seat"))
    monkeypatch.setattr(PRED, "set_pair_collision_filter", lambda c, p, f: order.append(("filter", f)) or True)
    monkeypatch.setattr(M, "og", SimpleNamespace(sim=SimpleNamespace(step_physics=lambda: order.append("step"))))
    male = SimpleNamespace(name="cap_0", states={AttachedTo: SimpleNamespace(set_value=lambda other, v: order.append(("attach", v)) or True)})
    female = SimpleNamespace(name="bottle_0")
    s = object.__new__(cls)
    roles = {"male": male, "female": female}
    s.t_role, s.p_role = "male", "female"
    s.obj = lambda role: roles[role]
    s.apply_goal(SimpleNamespace())
    assert order == [("filter", True), "seat", "step", ("attach", True)]


def test_loader_reports_the_attachment_pairs_without_filtering_them():
    from omnigibson.object_states import AttachedTo

    from ispatialgym.oracle import run_isg as RI

    cap = SimpleNamespace(name="cap_0", states={AttachedTo: SimpleNamespace(_disable_collision_between_child_and_parent=lambda c, p: pytest.fail("filtered at load"))})
    loader = object.__new__(RI.InstanceLoader)
    loader.physical = {"cap.n.02_1": cap, "bottle.n.01_1": SimpleNamespace(name="bottle_0")}
    out = loader._attachment_report([{"child": "cap.n.02_1", "parent": "bottle.n.01_1", "child_link": "M", "parent_link": "F"}])
    assert out == [{"child": "cap.n.02_1", "parent": "bottle.n.01_1", "child_link": "M", "parent_link": "F", "collision_filtered": False}]
