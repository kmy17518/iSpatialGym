"""oracle-fix 2026-11-03: Phase 1a's opt-in shifted init stance (``generation.init_edge_laterals_m``, tabletop.shifted_init_stance)."""
import types

import torch as th

from ispatialgym.sampling.v2 import tabletop as TT
from ispatialgym.sampling.v2.stances import EdgeFrame


def _sampler(lats, free_lats):
    edge = EdgeFrame.from_edge("-u", th.tensor([0.0, 0.0, 0.77]), th.tensor([-1.0, 0.0, 0.0]), 1.8, 0.0)
    seen = []

    def collision_free(st):
        seen.append(round(float(st.lateral), 2))
        return round(float(st.lateral), 2) in free_lats

    s = types.SimpleNamespace(gen=lambda k, d=None: lats if k == "init_edge_laterals_m" else d,
                              kit=types.SimpleNamespace(trav_free=lambda xy: True),
                              ikc=types.SimpleNamespace(stance_collision_free=collision_free))
    return s, edge, seen


def test_off_without_the_key():
    s, edge, seen = _sampler(None, {0.25})
    assert TT.shifted_init_stance(s, edge, 0.45) is None and seen == []


def test_first_free_shift_within_the_edge():
    s, edge, seen = _sampler([0.25, -0.25, 0.7, -0.4], {-0.4, 0.7})
    st = TT.shifted_init_stance(s, edge, 0.45)
    assert st is not None and abs(st.lateral + 0.4) < 1e-9
    assert seen == [0.25, -0.25, -0.4]  # 0.7 lies within 0.30 m of the 1.8 m edge's end: not tried
    assert abs(float(st.base_xy[0]) + 0.45) < 1e-6  # the base 0.45 m outside the edge, against the inward normal
