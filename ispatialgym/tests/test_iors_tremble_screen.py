"""IORS small touching: the template's target stability screen (owner 2026-10-12 item 7g; touching#04: 125 settle rejections, all the
target spray_bottle/asztxi trembling): the free spots it uses, the instance settle's judgement (``settle`` + ``tremble_after`` with the
``tremble_*`` limits) on the target alone, the first still spot passing, ``target_unstable`` when every spot trembles, the target parked
again, and the gating (touching on, the other relations off, the shelf layout and a continued template skipped). A fake world stands in
for the simulator."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.sim_utils import Reject, SimWorld, SupportTop
from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.tasks import iors_small as IS


def _box(dx, dy, h):
    return th.tensor([[sx * dx / 2, sy * dy / 2, z] for sx in (-1, 1) for sy in (-1, 1) for z in (0.0, h)], dtype=th.float32)


def _top(occupied_block=True):
    """A 1.0 x 0.6 m top at z 0.8 (2 cm raster) with another object on it at x 0.40-0.48, y 0.20-0.28."""
    xs, ys = th.arange(0.0, 1.0, 0.02), th.arange(0.0, 0.6, 0.02)
    occupied = th.zeros(len(xs), len(ys), dtype=th.bool)
    if occupied_block:
        occupied[20:24, 10:14] = True
    return SupportTop("table_0", xs, ys, th.ones(len(xs), len(ys), dtype=th.bool), occupied, 0.8, 0.0)


def _footprint(hull):
    fp = G.convex_hull_2d(hull[:, :2])
    return fp - 0.5 * (fp.max(dim=0).values + fp.min(dim=0).values)


def test_the_spots_take_the_footprint_on_free_cells_middle_first_then_spread():
    top = _top()
    fp = _footprint(_box(0.10, 0.10, 0.2))
    spots = IS.tremble_screen_spots(top, fp, 2)
    assert len(spots) == 2
    for sp in spots:
        assert TT.polygons_supported(top, (fp + sp)[None], 0.02).all()
    ok = top.cells()[TT.polygons_supported(top, top.cells()[:, None, :] + fp[None], 0.02)]
    d_mid = th.linalg.norm(ok - ok.mean(dim=0), dim=1)
    assert float(th.linalg.norm(spots[0] - ok.mean(dim=0))) == pytest.approx(float(d_mid.min()))  # the admissible cell nearest their middle
    assert spots[0].tolist() == pytest.approx([0.50, 0.36], abs=1e-6)  # (0.50, 0.30) itself is too close to the other object
    assert float(th.linalg.norm(spots[1] - spots[0])) > 0.5  # the second one far away on the top
    assert IS.tremble_screen_spots(top, fp, 0) == [] and IS.tremble_screen_spots(top, _footprint(_box(1.2, 0.2, 0.2)), 2) == []


class _World:
    """The ``SimWorld`` calls the screen makes: ``settle`` / ``tremble_after`` scripted per spot (``tremble``: one error string or None per
    call), ``activate`` / ``park`` recorded, the footprint / bottom offset of ``pose_centred_on``."""

    TREMBLE_FRAMES, TREMBLE_POS_TOL, TREMBLE_ANG_TOL = SimWorld.TREMBLE_FRAMES, SimWorld.TREMBLE_POS_TOL, SimWorld.TREMBLE_ANG_TOL
    TREMBLE_DRIFT_POS_TOL, TREMBLE_DRIFT_ANG_TOL = SimWorld.TREMBLE_DRIFT_POS_TOL, SimWorld.TREMBLE_DRIFT_ANG_TOL

    def __init__(self, tremble):
        self.tremble = list(tremble)
        self.calls, self.parked, self.last_tremble = [], [], {}
        self.spawned = {"target_x_0": None, "partner_y_0": None}

    def footprint(self, hull, pose):
        return G.convex_hull_2d(G.local_points_world(hull, pose)[:, :2])

    def bottom_offset(self, hull, quat):
        return -float(G.local_points_world(hull, (th.zeros(3), G.t(quat)))[:, 2].min())

    def activate(self, obj):
        self.calls.append(("activate", obj.name))

    def settle(self, ents, steps=60, robot_still=True, still=()):
        self.calls.append(("settle", [e.name for e in ents], steps))
        return None

    def tremble_after(self, ents, **limits):
        self.calls.append(("tremble_after", [e.name for e in ents], limits))
        err = self.tremble.pop(0)
        self.last_tremble = {ents[0].name: {"step_mm": 3.3 if err else 0.0, "drift_mm": 14.9 if err else 0.0}}
        return err

    def park(self, obj, index):
        self.parked.append((obj.name, index))


class _Target:
    name = "target_x_0"

    def __init__(self):
        self.poses = []

    def set_position_orientation(self, p, q):
        self.poses.append((p, q))


def _sampler(relation="touching", tremble=(None,), gen=None, **extra):
    s = IS.IORSSmall.__new__(IS.IORSSmall)
    s.relation, s.gp, s.spec = relation, SimpleNamespace(variant="a"), SimpleNamespace(generation=dict(gen or {}))
    s.hull, s.rest = {"target": _box(0.06, 0.10, 0.25)}, {"target": C.IDENTITY_QUAT.clone()}
    s.top, s.world, s.shelf_start = _top(), _World(tremble), None
    s.template_meta, s.template_facts = {}, {}
    target = _Target()
    s.objects, s.bindings = {"target.n.01_1": target}, SimpleNamespace(one=lambda role: SimpleNamespace(name="target.n.01_1"))
    s.kit = SimpleNamespace(sync_world=lambda: s.world.calls.append(("sync_world",)))
    for k, v in extra.items():
        setattr(s, k, v)
    return s, target


def test_a_still_target_passes_at_the_first_spot_judged_like_the_instance_settle_and_is_parked_again():
    s, target = _sampler(tremble=[None], gen={"tremble_frames": 30})
    s.target_tremble_screen()
    facts = s.template_facts["tremble_screen"]
    assert facts["spots_max"] == 2 and len(facts["spots"]) == 1 and facts["spots"][0]["still"] is True
    kinds = [c[0] for c in s.world.calls]
    assert kinds == ["activate", "settle", "tremble_after", "sync_world"]
    _, ents, steps = s.world.calls[1]
    assert ents == ["target_x_0"] and steps == s.DEFAULT_SETTLE_STEPS == 90  # the target alone, the instance's settle
    limits = s.world.calls[2][2]
    assert limits == {"frames": 30, "pos_tol": 0.0005, "ang_tol": 0.0087, "drift_pos_tol": 0.001, "drift_ang_tol": 0.01}  # generation, else SimWorld's
    p, q = target.poses[0]
    assert float(p[2]) == pytest.approx(0.8 + 0.003, abs=1e-5) and th.allclose(q, C.IDENTITY_QUAT)  # rest orientation, 3 mm above the top
    assert s.world.parked == [("target_x_0", 0)]  # back at its park slot


def test_a_target_trembling_on_every_spot_rejects_the_template_and_one_still_spot_is_enough():
    s, target = _sampler(tremble=["target_x_0 trembles: worst frame 3.34 mm", "target_x_0 trembles: worst frame 2.88 mm"])
    with pytest.raises(Reject) as e:
        s.target_tremble_screen()
    assert e.value.reason == "target_unstable" and "each of 2 free spots" in str(e.value) and "2.88 mm" in str(e.value)
    facts = s.template_facts["tremble_screen"]
    assert [sp["still"] for sp in facts["spots"]] == [False, False] and facts["spots"][0]["tremble"]["step_mm"] == 3.3
    assert s.world.parked == [("target_x_0", 0)] and len(target.poses) == 2
    s2, _ = _sampler(tremble=["trembles", None])
    s2.target_tremble_screen()
    assert [sp["still"] for sp in s2.template_facts["tremble_screen"]["spots"]] == [False, True]


@pytest.mark.parametrize("relation", ["ontop", "under", "inside", "nextto"])
def test_the_screen_is_off_for_the_other_relations_unless_asked(relation):
    s, _ = _sampler(relation)
    s.target_tremble_screen()
    assert s.template_facts["tremble_screen"]["skipped"] == "off" and s.world.calls == []
    s2, _ = _sampler(relation, gen={"tremble_screen_spots": 1})
    s2.target_tremble_screen()
    assert len(s2.template_facts["tremble_screen"]["spots"]) == 1


def test_the_shelf_layout_a_continued_template_and_spots_0_skip_the_screen():
    for extra, why in (({"shelf_start": {"lean_side": 1.0}}, "shelf layout"), ({"template_meta": {"n_written_before": 3}}, "continued")):
        s, _ = _sampler(**extra)
        s.target_tremble_screen()
        assert why in s.template_facts["tremble_screen"]["skipped"] and s.world.calls == []
    s, _ = _sampler(gen={"tremble_screen_spots": 0})
    s.target_tremble_screen()
    assert s.template_facts["tremble_screen"]["skipped"] == "off" and s.world.calls == []
