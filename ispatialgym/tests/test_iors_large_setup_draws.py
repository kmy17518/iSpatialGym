"""oracle-fix 2026-11-06 (inside_large#03): generation.init_setup_draws - more draws of the template's init surface from derived streams
when the template seed's draw finds none (CPU only)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

IL = pytest.importorskip("ispatialgym.sampling.v2.tasks.iors_large")
from ispatialgym.sampling.sim_utils import Reject  # noqa: E402


class _GP:
    def __init__(self, **gen):
        self.variant, self.generation = "iors-large-test", dict(gen)

    def gen(self, key, default=None):
        return self.generation.get(key, default)


class _S:
    bind_init_surface = IL.IORSLarge.bind_init_surface

    def __init__(self, ok_on_draw=None, **gen):
        """``try_surface`` admits a candidate only on the stream whose first uniform equals that of draw ``ok_on_draw`` (None: never)."""
        self.gp, self.ctx = _GP(**gen), SimpleNamespace(base_seed=1234 * 7919)
        self.template_facts, self.init_fallback, self.bound, self.tries = {}, None, [], []
        self._ok = None if ok_on_draw is None else self._first_uniform(ok_on_draw)

    def _first_uniform(self, k):
        rng = np.random.default_rng(self.ctx.base_seed if k == 0 else [int(self.ctx.base_seed), k])
        rng.permutation(1)
        return float(rng.uniform())

    def try_surface(self, kind, cand, rng):
        u = float(rng.uniform())  # the lateral draw's stand-in: the stream decides
        self.tries.append(u)
        if self._ok is None or abs(u - self._ok) > 1e-12:
            raise Reject("no_free_edge", f"{cand.name}: +u: no_manipulation_stance")

    def bind_scene_object(self, role, cand):
        self.bound.append((role, cand.name))


def test_init_setup_draws_off_keeps_the_one_template_seed_draw():
    """Unset (today): one draw on the template seed's stream, then no_init_surface (survey1 inside_large#03: lateral -0.03, 0 pairs)."""
    s = _S(ok_on_draw=2)
    with pytest.raises(Reject) as e:
        s.bind_init_surface("support", [SimpleNamespace(name="bar_egwapq_0")])
    assert e.value.reason == "no_init_surface" and len(s.tries) == 1 and not s.bound
    assert "setup draw" not in str(e.value)


def test_init_setup_draws_retries_on_derived_streams_until_one_admits():
    """generation.init_setup_draws 8: draw 0 is today's (the template seed); draws k = 1.. use the streams [base_seed, k]; the first that
    admits binds the support and records its draw (train85 found bar_egwapq_0 +u lateral +0.54 on draw 2)."""
    s = _S(ok_on_draw=2, init_setup_draws=8)
    s.bind_init_surface("support", [SimpleNamespace(name="bar_egwapq_0")])
    assert len(s.tries) == 3 and s.tries[0] == s._first_uniform(0)
    assert s.bound == [("support", "bar_egwapq_0")]
    assert s.template_facts["init_surface"] == {"class": "support", "name": "bar_egwapq_0", "tried": 3, "setup_draw": 2}

    s = _S(ok_on_draw=0, init_setup_draws=8)  # draw 0 admits: nothing changes (no setup_draw key)
    s.bind_init_surface("support", [SimpleNamespace(name="bar_egwapq_0")])
    assert len(s.tries) == 1 and s.template_facts["init_surface"] == {"class": "support", "name": "bar_egwapq_0", "tried": 1}


def test_init_setup_draws_exhausted_rejects_with_the_last_draws_errors():
    s = _S(ok_on_draw=None, init_setup_draws=3)
    with pytest.raises(Reject) as e:
        s.bind_init_surface("support", [SimpleNamespace(name="a"), SimpleNamespace(name="b")])
    assert e.value.reason == "no_init_surface" and len(s.tries) == 8 and not s.bound
    assert "setup draw 3: " in str(e.value)
