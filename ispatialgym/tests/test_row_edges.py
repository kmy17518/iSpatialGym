"""A support row that keeps only some edges of the top for the robot's stance (owner 2026-10-12 attachment#04 3a: bar_udatjt_0's long +u
edge, not its 1.14 m ends): the sheet note ``edges=`` (``ground_isg``), the customs entry and the candidate record, and Phase 1a's edge
list (``tabletop.edge_frames``); a row without the key keeps every edge."""

from __future__ import annotations

import copy
from types import SimpleNamespace as NS

import torch as th

from ispatialgym.sampling import ground_isg as GI
from ispatialgym.sampling.v2 import tabletop as TT

NOTE = "init=ontop; top_z=1.05 area=3.44 best_edge=-u run0.45=2.98 need=1.19 rooms=kitchen_0 src=B4 surface=z=1.05 ok; kitchen island"


def test_edges_note_key_is_parsed_in_order_without_duplicates():
    assert GI.parse_edges("init=ontop; top_z=1.05 edges=+u area=3.44") == ["+u"]
    assert GI.parse_edges("top_z=0.81 edges=-v,+v,-v; dining table") == ["-v", "+v"]
    assert GI.parse_edges(NOTE) == [] and GI.parse_edges("") == [] and GI.parse_edges(None) == []
    assert GI.parse_edges("best_edge=-u") == [] and GI.parse_edges("edges=+x") == []  # B4's best edge and non-edge names are not the key


def test_support_surface_entry_carries_the_edges_and_is_unchanged_without():
    row = {"scene": "house_single_floor", "name": "bar_udatjt_0", "note": NOTE}
    plain = GI.support_surface(row, "p#04~T3")
    assert plain == {"name": "bar_udatjt_0", "init": "ontop", "top_z": 1.05, "row": "p#04~T3"}
    kept = GI.support_surface(dict(row, note=NOTE.replace("top_z=1.05 ", "top_z=1.05 edges=+u ")), "p#04~T3")
    assert kept == {**plain, "edges": ["+u"]}


def _problem(surfaces):
    from ispatialgym.sampling.v2.problem import GroundedProblem, subtask_of_alias

    alias = "configuration_matching-attachment-lid-stockpot-bar"
    entry = {"iddl": "configuration_matching/attachment.iddl", "definition_id": 23,
             "roles": {"target": {"category": "lid", "sampleable": True}, "partner": {"category": "stockpot", "sampleable": True},
                       "support": {"category": "bar", "sampleable": False}, "floor": {"sampleable": False}},
             "scenes": {"house_single_floor": {"rooms": ["kitchen_0"], "bindings": {"support": [s["name"] for s in surfaces], "support_surfaces": surfaces,
                                                                                   "floor": {"room": "kitchen_0"}},
                                               "whitelist": {}, "blacklist": {}}},
             "generation": {}, "constraints": []}
    return GroundedProblem(alias, copy.deepcopy(entry), subtask=subtask_of_alias(alias))


def test_the_candidate_record_names_the_edges_only_when_the_entry_has_them():
    surfaces = [{"name": "bar_udatjt_0", "init": "ontop", "top_z": 1.05, "row": "p#04~T3", "edges": ["+u"]},
                {"name": "bar_x_0", "init": "ontop", "top_z": 0.9, "row": "p#04~T5"}]
    gp = _problem(surfaces)
    rec = gp.override_support("house_single_floor", "bar_udatjt_0", row="p#04~T3", rooms=["kitchen_0"])
    assert rec["edges"] == ["+u"] and gp.bindings("house_single_floor")["support_surfaces"] == [surfaces[0]]
    assert "edges" not in _problem(surfaces).override_support("house_single_floor", "bar_x_0", row="p#04~T5", rooms=["kitchen_0"])


def _sampler(surfaces=None):
    """A 0.20 x 0.20 m top under ``bar_udatjt_0``; ``surfaces``: the selection's support_surfaces (None: a sampler without a selection)."""
    from ispatialgym.sampling.sim_utils import SupportTop

    top = SupportTop("bar_udatjt_0", th.arange(10) * 0.02, th.arange(10) * 0.02, th.ones(10, 10, dtype=th.bool), th.zeros(10, 10, dtype=th.bool), 1.05, 0.0)
    s = NS(world=NS(obj=lambda name: NS(aabb=(th.tensor([-0.02, -0.02, 0.0]), th.tensor([0.2, 0.2, 1.05])))))
    if surfaces is not None:
        s.choice = NS(bindings={"support_surfaces": surfaces})
    return s, top


def test_edge_frames_keep_the_edges_the_support_row_names():
    every = ["+u", "+v", "-u", "-v"]
    s, top = _sampler()
    assert sorted(e.name for e, _ in TT.edge_frames(s, top)) == every and TT.row_edges(s, top) == []
    s, top = _sampler([{"name": "bar_udatjt_0", "init": "ontop", "top_z": 1.05, "row": "p#04~T3"}])
    assert sorted(e.name for e, _ in TT.edge_frames(s, top)) == every
    s, top = _sampler([{"name": "other_0", "top_z": 0.9, "edges": ["-v"]}, {"name": "bar_udatjt_0", "top_z": 1.05, "edges": ["+u"]}])
    assert TT.row_edges(s, top) == ["+u"] and [e.name for e, _ in TT.edge_frames(s, top)] == ["+u"]
    assert [e.name for e, _ in TT.edge_frames(s, top, rng=__import__("numpy").random.default_rng(3))] == ["+u"]
    # with an interior surface's open sides the row's edges narrow them further
    s, top = _sampler([{"name": "bar_udatjt_0", "top_z": 1.05, "edges": ["+u", "-v"]}])
    top.open_edges = ["-v", "+v"]
    assert [e.name for e, _ in TT.edge_frames(s, top)] == ["-v"]
