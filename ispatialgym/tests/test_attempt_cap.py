"""i7-stancememo (owner 2026-11-08, 5x.3): the adaptive per-attempt cap ISPATIALGYM_ATTEMPT_CAP (default off): max(300 s, 5 x the
alias's median written attempt); a cut attempt is a rejection with its own reason (time_cap), never a silent loss."""
import inspect
import time

import pytest

from ispatialgym.sampling import attempt_cap as AC
from ispatialgym.sampling.v2 import core as CORE


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    for k in (AC.ENV, AC.ENV_TABLE, AC.ENV_ALIAS, AC.ENV_RAISE, AC.ENV_RAISE_MAX, "ISPATIALGYM_ALIAS_MEMORY_ALIAS"):
        monkeypatch.delenv(k, raising=False)
    AC.end()
    yield monkeypatch
    AC.end()


def test_on_by_default_and_zero_turns_it_off(clean):
    assert AC.on() and AC.cap_for("inter_object_relation_satisfaction-under_large_scale_03")[0] == 650.0  # decision #33: default ON
    assert AC.cap_for() == (None, "no alias")  # no alias key (not a fast-flow job): uncapped
    clean.setenv(AC.ENV, "0")
    assert not AC.on() and AC.cap_for("x") == (None, "off")
    AC.check("kit.ik")  # nothing armed


def test_cap_from_the_table(clean, tmp_path):
    t = tmp_path / "m.tsv"
    t.write_text("# header\nalias_key\tn_written\tmedian_written_s\tcap_s\tpool\nfast_alias\t9\t40.0\t300.0\tprimary\nslow_alias\t4\t130.0\t650.0\tprimary\n")
    clean.setenv(AC.ENV, "1")
    clean.setenv(AC.ENV_TABLE, str(t))
    assert AC.cap_for("slow_alias")[0] == 650.0  # 5 x 130
    assert AC.cap_for("fast_alias")[0] == 300.0  # the 300 s floor
    assert AC.cap_for("unknown")[0] is None  # no written attempt: uncapped
    clean.setenv("ISPATIALGYM_ALIAS_MEMORY_ALIAS", "slow_alias")  # submit_fast.py sets it for every job
    assert AC.cap_for()[0] == 650.0
    clean.setenv(AC.ENV, "42")
    assert AC.cap_for("anything") == (42.0, "fixed")


def test_the_packaged_table_has_the_ab_aliases():
    tab = AC.load_table(AC.DEFAULT_TABLE)
    assert AC.cap_of_median(tab["inter_object_relation_satisfaction-under_large_scale_03"]) == 650.0
    assert AC.cap_of_median(tab["configuration_matching-articulation_open_small_scale_02"]) == pytest.approx(341.0)


def test_check_raises_a_base_exception_once_spent_and_not_while_suspended():
    AC.begin(time.time() - 10.0, 5.0)
    with AC.suspended():
        AC.check("start draw")  # exempt
    with pytest.raises(AC.AttemptCapStop) as ei:
        try:
            AC.check("kit.ik_batch")
        except Exception:  # the search's broad handlers must not swallow it
            pytest.fail("swallowed")
    assert not isinstance(ei.value, Exception) and ei.value.where == "kit.ik_batch" and ei.value.cap_s == 5.0
    AC.end()
    AC.check("after the sample stage")  # disarmed


def test_run_instances_turns_the_stop_into_a_time_cap_rejection():
    src = inspect.getsource(CORE.SamplerV2.run_instances)
    i_begin, i_sample, i_end = src.index("AC.begin(t0, cap_s)"), src.index("draft = self.sample(rng)"), src.index("AC.end()")
    assert i_begin < i_sample < i_end
    assert 'raise Reject("time_cap"' in src and "except AC.AttemptCapStop" in src
    assert "with AC.suspended():" in inspect.getsource(CORE.SamplerV2.spa_draw)


def _sampler(cap):
    from types import SimpleNamespace

    s = CORE.SamplerV2.__new__(CORE.SamplerV2)
    s.kit = SimpleNamespace(stats={})
    s.gp = SimpleNamespace(variant="stub")
    s._attempt_cap = (cap, "table")
    return s


def test_two_consecutive_cuts_double_the_cap(clean):
    clean.setenv(AC.ENV_RAISE_MAX, "0")  # the unbounded doubling (the default bounds it at 1.5x since lead #74)
    """Lead decision #63: a cap from medians measured on older code (disembedding_10: 338 s against ~390 s drawn-start attempts) must not
    cut most attempts - two consecutive time_cap cuts double the job's cap; any other outcome resets the streak; =0 keeps the old rule."""
    s = _sampler(338.0)
    for reason in ("time_cap", "target_placement", "time_cap"):  # not consecutive
        s._cap_after_attempt(reason)
    assert s.attempt_cap()[0] == 338.0
    s._cap_after_attempt("time_cap")
    assert s.attempt_cap()[0] == 676.0 and s.kit.stats["time_cap.raised"] == 1
    for _ in range(2):
        s._cap_after_attempt("time_cap")
    assert s.attempt_cap()[0] == 1352.0
    s._cap_after_attempt(None)  # written
    s._cap_after_attempt("time_cap")
    assert s.attempt_cap()[0] == 1352.0
    clean.setenv(AC.ENV_RAISE, "0")
    s2 = _sampler(338.0)
    for _ in range(5):
        s2._cap_after_attempt("time_cap")
    assert s2.attempt_cap()[0] == 338.0 and "time_cap.raised" not in s2.kit.stats
    assert "self._cap_after_attempt(e.reason)" in inspect.getsource(CORE.SamplerV2.run_instances)


def test_the_raise_bound(clean):
    """Lead decision #74: ISPATIALGYM_ATTEMPT_CAP_RAISE_MAX bounds the raise (default off = the doubling)."""
    clean.setenv(AC.ENV_RAISE_MAX, "1.5")
    s = _sampler(650.0)
    for _ in range(2):
        s._cap_after_attempt("time_cap")
    assert s.attempt_cap()[0] == 975.0 and s.kit.stats["time_cap.raise_bounded"] == 1 and s.kit.stats["time_cap.raised"] == 1
    for _ in range(4):
        s._cap_after_attempt("time_cap")
    assert s.attempt_cap()[0] == 975.0 and s.kit.stats["time_cap.raised"] == 1 and s.kit.stats["time_cap.raise_bounded"] == 3
    clean.delenv(AC.ENV_RAISE_MAX)
    assert AC.raise_max() == 1.5  # the default since the r5 A/B (lead #74)
    clean.setenv(AC.ENV_RAISE_MAX, "0")
    assert AC.raise_max() == 0.0
    s2 = _sampler(650.0)
    for _ in range(4):
        s2._cap_after_attempt("time_cap")
    assert s2.attempt_cap()[0] == 2600.0
