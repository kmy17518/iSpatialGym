"""i7-tray 2026-11-08 (owner decision item 3, option b): the sampler skips a v2 row whose recorded in-hand turn at the close
(``validation.json`` ``achieved.off_target_rad``) exceeds the pose tolerance of the task using it (``ISPATIALGYM_GRASP_TURN_FILTER``)."""

import json
from types import SimpleNamespace

import pytest

from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2 import tabletop as TT

FORK = "disembedding-tablefork-tablefork_1_table_knife_1_tablespoon_1-breakfast_table"  # small: 0.10 rad
SCALING = "object_scaling-tablefork-x"  # micro: instance-dependent, no fixed tolerance


def _lib(tmp_path, turns):
    d = tmp_path / "tablefork__pqrtkc__s1"
    d.mkdir()
    (d / "library.json").write_text("{}")
    trials = [{"candidate": n, "verdict": "valid", "achieved": {"off_target_rad": r}} for n, r in turns.items()]
    trials.append({"candidate": "top_0", "verdict": "invalid", "achieved": {"off_target_rad": 0.9}})
    (d / "validation.json").write_text(json.dumps({"trials": trials}))
    return SimpleNamespace(path=str(d / "library.json"))


def _sampler(variant):
    return SimpleNamespace(gp=SimpleNamespace(variant=variant), kit=SimpleNamespace(stats={}))


ROWS = [SimpleNamespace(name="top_2"), SimpleNamespace(name="top_7"), SimpleNamespace(name="top_9")]


@pytest.fixture
def lib(tmp_path):
    return _lib(tmp_path, {"top_2": 0.005, "top_7": 0.233, "top_9": 0.015})


def test_tolerances_from_the_profile():
    assert TT.grasp_turn_tolerance(FORK) == pytest.approx(0.10)
    assert TT.grasp_turn_tolerance(SCALING) is None
    assert TT.grasp_turn_tolerance("configuration_matching-articulation_open_large_scale-fridge-x") is None  # no rule: unchanged
    assert TT.grasp_turn_tolerance("inter_object_relation_satisfaction-under_large_scale-sandal-x") == pytest.approx(0.15)


def test_a_row_under_the_tolerance_stays_and_one_over_it_is_skipped(lib, monkeypatch):
    monkeypatch.setenv(TT.TURN_FILTER_SWITCH, "1")
    s = _sampler(FORK)
    kept = TT.filter_by_turn(s, list(ROWS), lib, "tablefork__pqrtkc__t")
    assert [g.name for g in kept] == ["top_2", "top_9"]
    assert s.kit.stats["grasp.off_target_filtered"] == 1
    assert s.kit.stats["grasp.off_target_filtered.tablefork__pqrtkc__s1.disembedding"] == 1


def test_switch_off_measures_only(lib, monkeypatch):
    monkeypatch.setenv(TT.TURN_FILTER_SWITCH, "0")
    s = _sampler(FORK)
    assert [g.name for g in TT.filter_by_turn(s, list(ROWS), lib, "k")] == ["top_2", "top_7", "top_9"]
    assert s.kit.stats == {"grasp.off_target_would_filter": 1}


def test_a_task_without_a_pose_tolerance_is_unchanged(lib, monkeypatch):
    monkeypatch.setenv(TT.TURN_FILTER_SWITCH, "1")
    s = _sampler(SCALING)
    assert [g.name for g in TT.filter_by_turn(s, list(ROWS), lib, "k")] == ["top_2", "top_7", "top_9"]
    assert s.kit.stats == {}


def test_every_row_over_rejects(tmp_path, monkeypatch):
    monkeypatch.setenv(TT.TURN_FILTER_SWITCH, "1")
    lib = _lib(tmp_path, {"top_7": 0.233})
    with pytest.raises(Reject) as e:
        TT.filter_by_turn(_sampler(FORK), [SimpleNamespace(name="top_7")], lib, "k")
    assert e.value.reason == "grasps_off_target"


def test_unmeasured_rows_stay(tmp_path, monkeypatch):
    monkeypatch.setenv(TT.TURN_FILTER_SWITCH, "1")
    lib = _lib(tmp_path, {"top_7": 0.233})
    kept = TT.filter_by_turn(_sampler(FORK), [SimpleNamespace(name="top_7"), SimpleNamespace(name="top_3")], lib, "k")
    assert [g.name for g in kept] == ["top_3"]


def test_the_filter_is_on_by_default(lib, monkeypatch):
    """Owner decision 2026-11-08 5u.1: the switch unset = on."""
    monkeypatch.delenv(TT.TURN_FILTER_SWITCH, raising=False)
    assert TT.turn_filter_on()
    s = _sampler(FORK)
    assert [g.name for g in TT.filter_by_turn(s, list(ROWS), lib, "k")] == ["top_2", "top_9"]
