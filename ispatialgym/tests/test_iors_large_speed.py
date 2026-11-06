"""oracle-fix 2026-11-05 speed round: early checks that reject a doomed candidate before the expensive stages (CPU only)."""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch as th

IL = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_large")
from ispatialgym.sampling.v2 import program_check as PCK  # noqa: E402


class _GP:
    def __init__(self, **gen):
        self.variant, self.generation = "iors-large-test", dict(gen)

    def gen(self, key, default=None):
        return self.generation.get(key, default)


class _S:
    _b_frame_options = IL.IORSLarge._b_frame_options
    _b_margin_options = IL.IORSLarge._b_margin_options
    grasp_reach_margin = IL.IORSLarge.grasp_reach_margin

    def __init__(self, **gen):
        self.gp, self.kit = _GP(**gen), SimpleNamespace(stats={})


def _pose(z):
    return th.tensor([0.0, 0.0, float(z)]), th.tensor([0.0, 0.0, 0.0, 1.0])


def test_b_frame_options_drop_the_finger_assignments_whose_fingertips_leave_the_image(monkeypatch):
    """generation.b_fingertip_frame_check (brief15: 60 of 123 rejected attempts failed the program check's fingertip gate at grasp.pre_grasp,
    144 s each): B's row / twin are judged by that gate (pre-grasp and grasp) before the C search; off, unchanged."""
    row, twin = SimpleNamespace(name="g0"), SimpleNamespace(name="g0~r180")
    opts = [(row, _pose(1.0), _pose(1.1), False), (twin, _pose(2.0), _pose(2.1), True)]
    facts = {"chosen": "row"}
    st = object()
    s = _S()
    assert s._b_frame_options((opts, facts), st, "B") == ((opts, facts), "")
    assert s._b_margin_options((opts, facts), st, "right", "B") == ((opts, facts), "")

    s = _S(b_fingertip_frame_check=True)
    seen = []

    def gate(sampler, stance, name, items, **kw):  # the row's pre-grasp (z 1.1) leaves the image, the twin's (z 2.1) does not
        seen.append((stance, name, [k for k, _ in items]))
        return {"key": "pre_grasp", "u": 10.0, "v": 1100.0, "depth_m": 0.4} if float(items[0][1][0][0][2]) < 1.5 else None

    monkeypatch.setattr(PCK, "fingertips_in_frame", gate)
    (kept, f2), why = s._b_frame_options((opts, facts), st, "B")
    assert kept == [opts[1]] and f2["chosen"] == "twin" and "fails the fingertip frame gate" in f2["frame_flip"] and why == ""
    assert seen[0] == (st, "grasp", ["pre_grasp", "approach"]) and s.kit.stats["b_frame.rejected"] == 1
    assert facts == {"chosen": "row"}  # the caller's facts untouched

    monkeypatch.setattr(PCK, "fingertips_in_frame", lambda *a, **k: {"key": "approach", "u": 0.0, "v": 0.0, "depth_m": 0.3})
    out, why = s._b_margin_options((opts, facts), st, "right", "B")
    assert out is None and why.startswith("[B] g0: approach fingertip outside the image") and "g0~r180" in why


def test_grasp_at_pose_skips_the_grasps_its_accept_check_refuses():
    """touching#03 (oracle-fix 2026-11-05, generation.shelf_transit_per_grasp): grasp_at_pose's ``accept`` judges each grasp that passes the
    exact IK - the first it accepts is returned, a refused one is skipped for the next; all refused -> grasp_unreachable with the reasons."""
    from ispatialgym.sampling.sim_utils import Reject
    from ispatialgym.sampling.v2 import tabletop as TT
    from ispatialgym.sampling.v2.ik import IKResult

    calls = []
    s = SimpleNamespace(gp=_GP(grasp_twins=False),
                        ikc=SimpleNamespace(exact=lambda poses, st, arm, **kw: calls.append(arm) or IKResult(True, th.zeros(28), "", len(poses))))
    grasps = [SimpleNamespace(name=n, key=n, pose=(th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])), finger_q=[]) for n in ("g0", "g1", "g2")]
    manip = SimpleNamespace(name="manipulation")
    pose = (th.tensor([1.0, 0.0, 0.5]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    rec = TT.grasp_at_pose(s, manip, pose, grasps, [], 0.0, arm="left", object_name="book", accept=lambda r: "transit blocked" if r.grasp_name == "g0" else None)
    assert rec.grasp_name == "g1" and rec.arm == "left" and calls == ["left", "left"]
    assert TT.grasp_at_pose(s, manip, pose, grasps, [], 0.0, object_name="book").grasp_name == "g0"  # without accept: the first IK pass
    with pytest.raises(Reject) as e:
        TT.grasp_at_pose(s, manip, pose, grasps, [], 0.0, object_name="book", accept=lambda r: "transit blocked")
    assert e.value.reason == "grasp_unreachable" and "g2: transit blocked" in str(e.value)


def test_failure_memos_persist_in_the_sampling_cache_per_generation_block(monkeypatch, tmp_path):
    """generation.overhang_fail_memo_persist / fixed_start_memo_persist (oracle-fix 2026-11-05): a failed overhang pick / fixed-start grasp is
    stored in the sampling cache (shared by one alias's jobs) and read back by another sampler with the same scope and generation block; a
    changed generation block or scope is another key; off, nothing is read or written."""
    IS = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_small")
    monkeypatch.setenv("ISPATIALGYM_SAMPLING_CACHE", str(tmp_path))
    key = ((1.0, 2.0, 0.5, 0.0, -0.92), (("left", (1.0, 2.0, 0.5, 0.0, -0.92)),), "grasp_1", 3.927, (23.9, 13.7), 0.77, True)

    def large(**gen):
        gp = SimpleNamespace(variant="hb", generation=dict(gen))
        gp.gen = lambda k, d=None: gp.generation.get(k, d)
        return SimpleNamespace(gp=gp, ikc=SimpleNamespace(cache_scope={"scene": "house_single_floor", "rooms": ["childs_room_1"]}))

    P = IL.IORSLarge._overhang_memo_persisted
    assert P(large(overhang_fail_memo_persist=False), key, "exact IK failed") is None and not list(tmp_path.iterdir())
    s1 = large(overhang_fail_memo_persist=True, push_m=0.16, _note="x")
    assert P(s1, key) is None
    P(s1, key, "grasp exact IK failed at pose 1 of 2")
    assert P(large(overhang_fail_memo_persist=True, push_m=0.16, _note="other note"), key) == "grasp exact IK failed at pose 1 of 2"  # notes ignored
    assert P(large(overhang_fail_memo_persist=True, push_m=0.17), key) is None  # another generation block
    s2 = large(overhang_fail_memo_persist=True, push_m=0.16)
    s2.ikc.cache_scope = {"scene": "house_single_floor", "rooms": ["office_0"]}
    assert P(s2, key) is None  # another scope

    def small(**gen):
        s = large(**gen)
        s.gen = s.gp.gen
        s.shelf_pose = (th.tensor([1.0, 2.0, 0.4]), th.tensor([0.0, 0.0, 0.0, 1.0]))
        return s

    F = IS.IORSSmall._fixed_start_memo_persisted
    k2 = ((1.0, 2.0, -1.57), 0.321, 0.0, (0.1, 0.2), "left", None, True)
    F(small(fixed_start_memo_persist=True), k2, "[grasp_unreachable] transit check: blocked")
    assert F(small(fixed_start_memo_persist=True), k2) == "[grasp_unreachable] transit check: blocked"
    s3 = small(fixed_start_memo_persist=True)
    s3.shelf_pose = (th.tensor([1.0, 2.1, 0.4]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    assert F(s3, k2) is None and F(small(), k2) is None  # another start pose; off


def test_grasp_at_pose_prescreen_skips_rows_whose_row_and_twin_fail_the_batched_screen():
    """generation.fixed_start_prescreen (oracle-fix 2026-11-05; touching#03: 83 % of the attempt time in grasp_at_pose's exact IK): one batched
    screen of every row's and twin's [pre-grasp, grasp]; a row whose row and twin both fail it gets no exact IK."""
    from ispatialgym.sampling.sim_utils import Reject
    from ispatialgym.sampling.v2 import tabletop as TT
    from ispatialgym.sampling.v2.ik import IKResult

    exact, screens = [], []

    def screen(pairs, st, arm, ignore_for_last=(), fingers=None):
        screens.append((len(pairs), fingers))
        return th.tensor([False, False, False, True, True, True])  # g0 row / twin fail, g1 twin passes, g2 both pass

    s = SimpleNamespace(gp=_GP(grasp_twins=False), kit=SimpleNamespace(stats={}),
                        ikc=SimpleNamespace(exact=lambda poses, st, arm, **kw: exact.append(1) or IKResult(True, th.zeros(28), "", len(poses)), screen_pairs=screen))
    grasps = [SimpleNamespace(name=n, key=n, pose=(th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])), finger_q=[]) for n in ("g0", "g1", "g2")]
    manip, pose = SimpleNamespace(name="manipulation"), (th.tensor([1.0, 0.0, 0.5]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    s.gp.generation["grasp_twins"] = True  # rows and twins screened in pairs (pick_twin itself is not exercised: the accept refuses g1)
    import ispatialgym.sampling.v2.tabletop as T

    picks = []

    def fake_twin(s_, st, arm, g, eio, eef, pre, res, kw, **k):
        picks.append(g.name)
        return SimpleNamespace(grasp=g, eef_in_object=eio, eef=eef, pre=pre, res=res, passed=True, twin=False, facts={})

    orig = T.pick_twin
    T.pick_twin = fake_twin
    try:
        rec = TT.grasp_at_pose(s, manip, pose, grasps, [], 0.0, object_name="book", fingers=0.06, prescreen=True,
                               accept=lambda r: "transit blocked" if r.grasp_name == "g1" else None)
        assert rec.grasp_name == "g2" and picks == ["g1", "g2"] and len(exact) == 2 and screens == [(6, 0.06)]
        assert s.kit.stats["fixed_start_prescreen.dropped"] == 1
        with pytest.raises(Reject) as e:
            TT.grasp_at_pose(s, manip, pose, grasps[:1], [], 0.0, object_name="book", prescreen=True)
        assert "g0: neither the row nor its twin passes the batched screen" in str(e.value) and len(exact) == 2
    finally:
        T.pick_twin = orig


def test_dead_at_b_holds_only_when_every_drawable_pair_failed_at_b():
    """generation.c_skip_dead_b (oracle-fix 2026-11-06; under_large_scale_01: 41 of 42 C tries of a B exhausted by the cached B failures alone,
    after their push chains): a C is dead when every pair of F_live its draws can take keys into the B cache as None (failed) or has no
    aligned goal yaw; an unknown or solved pair, or no candidate pair at all, keeps it."""
    step = math.pi / 8
    g0, g1 = SimpleNamespace(name="g0"), SimpleNamespace(name="g1")
    F = [(0.0, g0), (step, g0), (step, g1)]
    dead = {(0, "g0"): None, (1, "g0"): None, (1, "g1"): None}
    assert IL.dead_at_b(F, dead, step)
    assert not IL.dead_at_b(F, {(0, "g0"): None, (1, "g0"): None}, step)  # (1, g1) not tried at B yet
    assert not IL.dead_at_b(F, {**dead, (1, "g1"): {"yaw": step}}, step)  # (1, g1) solved at B
    assert not IL.dead_at_b([], dead, step)  # nothing to draw: the C loop's own rule
    # the push-under yaw lock's buckets: only the kept yaws' pairs count
    assert IL.dead_at_b(F, {(0, "g0"): None}, step, buckets={0})
    assert not IL.dead_at_b(F, {(0, "g0"): None}, step, buckets={0, 1})
    assert not IL.dead_at_b(F, dead, step, buckets={5})  # no pair at a kept bucket
    # aligned init yaws (the touching mode): the draw's key is the aligned yaw; False (no aligned goal yaw) is a dead draw too
    assert IL.dead_at_b(F, {(0.1234, "g0"): None, (0.1234, "g1"): None}, step, end_aligned=lambda y: 0.12341)
    assert not IL.dead_at_b(F, dead, step, end_aligned=lambda y: 0.12341)
    assert IL.dead_at_b(F, {}, step, end_aligned=lambda y: False)


def test_c_dead_at_b_narrows_to_the_push_under_lock_kept_yaws():
    """_c_dead_at_b: the carry yaw offset of the C (C yaw - B yaw, 0 with carry_yaw 'world') feeds the lock's heel-first rule; with the lock
    off (or not a push) every pair counts; a lock that keeps every yaw or none leaves the judgement to the plain check / the C loop."""
    step = math.pi / 8
    g0 = SimpleNamespace(name="g0")
    F = [(0.0, g0), (math.pi, g0)]
    cache = {(0, "g0"): None}  # the yaw-0 pair failed at B, the yaw-pi pair never tried

    class _L:
        _c_dead_at_b = IL.IORSLarge._c_dead_at_b
        end_aligned_init_yaw = IL.IORSLarge.end_aligned_init_yaw

        def __init__(self, lock, push=True, **gen):
            self.gp, self.push_under, self._lock, self.rest = _GP(**gen), push, lock, th.tensor([0.0, 0.0, 0.0, 1.0])
            self.offs = []

        def push_yaw_lock(self):
            return self._lock

        def push_contact_offset(self, q, d):
            return None if self.gp.gen("no_offset") else 0.12

        def push_yaw_kept(self, q, d, off):  # keeps the world yaw near pi/2 (the heel first along +y) with a contact offset only
            self.offs.append(off)
            yaw = IL.G.yaw_of(q)
            return off > 0 and abs(math.remainder(yaw - math.pi / 2, 2 * math.pi)) < 0.3

    place = SimpleNamespace(yaw=math.pi / 2, extras={"push_dir": [0.0, 1.0]})
    manip = SimpleNamespace(yaw=0.0)
    s = _L("heel_first")
    assert s._c_dead_at_b(F, cache, place, manip, step) and s.offs == [0.12, 0.12]  # d_yaw pi/2: init yaw 0 kept only, and it failed at B
    assert not _L(None)._c_dead_at_b(F, cache, place, manip, step)  # no lock: the yaw-pi pair is live
    assert not _L("heel_first", push=False)._c_dead_at_b(F, cache, place, manip, step)
    assert not _L("heel_first", carry_yaw="world")._c_dead_at_b(F, cache, place, manip, step)  # d_yaw 0: no init yaw kept -> the C loop's rule
    s = _L("heel_first", no_offset=True)
    assert not s._c_dead_at_b(F, cache, place, manip, step) and s.offs == [-1.0, -1.0]  # no contact offset: -1 as the C loop's offs
    assert _L("heel_first")._c_dead_at_b(F, {(0, "g0"): None, (8, "g0"): None}, place, manip, step)


def test_c_skip_dead_b_runs_before_the_push_chain_and_the_f_goal_screen_and_is_opt_in():
    """The C loop of IORSLarge._couple asks _c_dead_at_b only with generation.c_skip_dead_b, at the top of a C try: before the carry-margin
    probe, the drop trunks, the chain-first push chains and the F_goal / release-tilt screens."""
    import inspect

    src = inspect.getsource(IL.IORSLarge._couple)
    i_skip = src.index("if skip_dead_b and self._c_dead_at_b(F_live, b_ik_cache, place, manip, yaw_step)")
    assert 'bool(self.gp.gen("c_skip_dead_b", False))' in src
    for later in ("if base_key in carry_failed_bases", "self._push_under_chain(", "self.release_feasible(", "self.screen_release_tilts("):
        assert src.index(later) > i_skip, later
    assert src.index("F_live = [yg for yg in F_init") < i_skip
