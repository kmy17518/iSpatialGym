"""i7-memory2 (owner 2026-11-08 2.3, lead 2026-11-08: every family with a pick-stance choice): the outcome memory's PICK hook at each
family's manipulation stance candidate ordering - the flagged (arm x trunk[, stand-off]) bin's candidates are tried last unless a Thompson
draw keeps them (never removed, never below the floor); memory off = the family's own order. Each test runs the family's real
``sample_instance`` on a stub up to its first candidate (a sentinel raised by the candidate's ``with_trunk``)."""
import types

import numpy as np
import pytest
import torch as th

from ispatialgym.sampling import alias_memory as AM


class Tried(Exception):
    def __init__(self, cand):
        super().__init__("tried")
        self.cand = cand


class Cand:
    def __init__(self, idx, tr, ti, so=0.35):
        self.idx, self.translate, self.tilt, self.stand_off, self.lateral = idx, tr, ti, so, 0.0
        self.focus = th.tensor([0.6, 0.0, 0.8])
        self.extras = {"focus_info": {"depth": 0.55}}
        self.arms = ["right"]

    def with_trunk(self, *a, **k):
        raise Tried(self)

    def world_to_base(self, xy):
        return th.tensor([0.55, 0.0])

    def local(self, xy):
        return th.tensor([0.55, 0.0])


def row(ok, tr, ti, so=0.35, arm="right", t="2026-11-07T00"):
    r = {"alias": "a", "outcome": "success" if ok else "failure", "stage": None if ok else "pick", "excluded": None, "failure_type": None if ok else "x",
         "tag": "t", "place": None, "pick": {"arm": arm, "stance": "manipulation", "stand_off": so, "translate": tr, "tilt": ti, "family": "g"},
         "ctx": {"snapshot": "trunk_a", "spec": "s", "keys": "k"}, "finished": t, "inst": "i", "jobid": "1"}
    r["place_obs"], r["pick_obs"] = AM.attribute(r)
    return r


def memory(arm="right"):
    rs = [row(True, 0.0, -0.61, arm=arm, t=f"2026-11-07T0{i}") for i in range(6)]
    rs += [row(False, 0.25, -0.53, arm=arm, t=f"2026-11-07T1{i}") for i in range(5)]  # the flagged bin
    m = AM.AliasMemory("a", rs, AM.Thresholds())
    m.weights({"snapshot": "trunk_a", "spec": "s", "keys": "k"})
    return m


def cands():
    return [Cand(0, 0.25, -0.53), Cand(1, 0.0, -0.61), Cand(2, 0.25, -0.53), Cand(3, 0.0, -0.61)]


def stub(**kw):
    s = types.SimpleNamespace(kit=types.SimpleNamespace(stats={}), gen=lambda k, d=None: d, init_stance=types.SimpleNamespace(name="init"),
                              manip_candidates=cands(), gp=types.SimpleNamespace(variant="v"))
    for k, v in kw.items():
        setattr(s, k, v)
    return s


def first_tried(fn, make, seed, mem):
    s = make()
    with pytest.raises(Tried) as e:
        fn(s, np.random.default_rng(seed))
    return e.value.cand.idx, s.kit.stats


def check_family(monkeypatch, fn, make, arm="right"):
    """Memory on: the flagged candidates (0, 2) mostly come last, sometimes first (the floor / Thompson keep); the counters count it.
    Memory off: the family's own first candidate."""
    mem = memory(arm)
    monkeypatch.setattr(AM, "memory_for", lambda sampler: None)
    own = {seed: first_tried(fn, make, seed, None)[0] for seed in range(40)}
    assert {0, 2} & set(own.values())  # without the memory the flagged candidates do come first
    monkeypatch.setattr(AM, "memory_for", lambda sampler: mem)
    firsts, stats = [], None
    for seed in range(200):
        k, stats = first_tried(fn, make, seed, mem)
        firsts.append(k)
    flagged_first = sum(1 for k in firsts if k in (0, 2))
    assert 0 < flagged_first < 100  # deferred most of the time, never blocked
    assert stats["alias_memory.pick.consulted"] >= 1 and stats["alias_memory.pick.flagged"] == 2
    assert stats["alias_memory.pick.reordered"] + stats["alias_memory.pick.explored"] == 2


def test_attachment(monkeypatch):
    from ispatialgym.sampling.v2.tasks.attachment import Attachment

    check_family(monkeypatch, Attachment.sample_instance,
                 lambda: stub(_candidate_rejects={}, _candidate_order=lambda rng, rej: [int(k) for k in rng.permutation(4)]))


def test_block_design(monkeypatch):
    from ispatialgym.sampling.v2.tasks.block_design import BlockDesign

    check_family(monkeypatch, BlockDesign.sample_instance, lambda: stub(_flips_expected=lambda: False))


def test_alignment_axial(monkeypatch):
    from ispatialgym.sampling.v2.tasks.alignment_axial import AlignmentAxial

    monkeypatch.delenv("ISPATIALGYM_CONTACT_MAPS", raising=False)
    from ispatialgym.sampling.v2 import tabletop as TT

    monkeypatch.setattr(TT, "contact_maps_on", lambda: False)
    check_family(monkeypatch, AlignmentAxial.sample_instance, lambda: stub(left_bands=None))


def test_disembedding(monkeypatch):
    from ispatialgym.sampling.v2.tasks.disembedding import Disembedding

    b = types.SimpleNamespace(one=lambda r: types.SimpleNamespace(name=r), many=lambda r: [])
    check_family(monkeypatch, Disembedding.sample_instance,
                 lambda: stub(tname="t", src_role="src", dst_role="dst", bindings=b, segment_arms=["right"]))


def test_articulation_small(monkeypatch):
    from ispatialgym.sampling.v2.tasks.articulation_small import ArticulationSmall

    check_family(monkeypatch, ArticulationSmall.sample_instance, lambda: stub(handle_arm="right"))


def test_draw_manipulation_articulation_large_and_alignment_wall(monkeypatch):
    """The uniform Phase-2 draw (articulation large, alignment wall; the hybrid follow plans from this stance): with ``memory_arm`` a flagged
    grid trunk leaves the draw's pool unless kept; the same draw without it; only flagged survivors left = drawn as before."""
    import inspect

    from ispatialgym.sampling.v2 import prompt0 as P0
    from ispatialgym.sampling.v2.tasks import alignment_wall, articulation_large

    for mod in (articulation_large, alignment_wall):
        assert "memory_arm=" in inspect.getsource(mod)

    class Grid(Cand):
        def __init__(self, *a):
            super().__init__(*a)
            self.extras = {"band": (self.tilt, self.tilt)}

        def with_trunk(self, tr, ti, name):
            return types.SimpleNamespace(translate=tr, tilt=ti, focus=self.focus, arms=None, extras={}, grid=self)

    def body_free(s, sts, **k):
        raise Tried(sts[0].grid)

    monkeypatch.setattr(P0, "body_free", body_free)
    mem = memory()

    def draw(seed, arm, pool=None):
        s = stub()
        pool = pool if pool is not None else [Grid(0, 0.25, -0.53), Grid(1, 0.0, -0.61), Grid(2, 0.25, -0.53), Grid(3, 0.0, -0.61)]
        rng = np.random.default_rng(seed)
        with pytest.raises(Tried) as e:
            P0.draw_manipulation(s, pool, rng, ["right"], memory_arm=arm)
        return e.value.cand.idx, s.kit.stats

    monkeypatch.setattr(AM, "memory_for", lambda sampler: None)
    off = [draw(seed, "right")[0] for seed in range(60)]
    assert off == [draw(seed, None)[0] for seed in range(60)]  # memory off: the same draw, bit for bit
    monkeypatch.setattr(AM, "memory_for", lambda sampler: mem)
    on = [draw(seed, "right") for seed in range(200)]
    flagged = sum(1 for k, _ in on if k in (0, 2))
    assert 0 < flagged < 60 and sum(1 for k in off if k in (0, 2)) > 15
    assert on[-1][1]["alias_memory.pick.flagged"] == 2
    only = [draw(seed, "right", pool=[Grid(0, 0.25, -0.53), Grid(2, 0.25, -0.53)])[0] for seed in range(20)]
    assert set(only) <= {0, 2}  # only flagged survivors left: still drawn (never a block)


def test_drawn_stance_keyed_by_grid_tilt():
    place, pick = AM.config_keys({"stances": {"manipulation": {"translate": 0.0, "tilt": -0.2899, "grid_tilt": -0.3, "stand_off": 0.85, "arms": ["right"]}},
                                  "grasps": [{"arm": "right", "grasp_name": "bar"}]})
    assert pick["tilt"] == -0.3 and place is None
    _, pick = AM.config_keys({"stances": {"manipulation": {"translate": 0.0, "tilt": -0.2899, "stand_off": 0.85, "arms": ["right"]}}, "grasps": [{}]})
    assert pick["tilt"] == -0.29
