"""Large-scale IORS inside / touching / nextto, registered ahead of their modes (owner 2026-10-12): every registry knows the three
subtasks and each relation's hook rejects ``relation_not_implemented`` until its mode lands (no simulator)."""

import pytest

from ispatialgym.oracle import programs as P
from ispatialgym.oracle import sidecar as SC
from ispatialgym.sampling import generation_defaults as GD
from ispatialgym.sampling import ground_isg as GI
from ispatialgym.sampling import iddl as I
from ispatialgym.sampling import v1_bridge as VB
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2 import problem as PB
from ispatialgym.sampling.v2 import program_check as PC
from ispatialgym.sampling.v2.core import REGISTRY
from ispatialgym.sampling.v2.tasks import iors_large as IL

IORS = "inter_object_relation_satisfaction"
NEW = {"inside": f"{IORS}-inside_large_scale", "touching": f"{IORS}-touching_large_scale", "nextto": f"{IORS}-nextto_large_scale"}
ONTOP = f"{IORS}-ontop_large_scale"


def test_relation_map_and_sampler_registry():
    for rel, st in NEW.items():
        assert IL.RELATION_OF_SUBTASK[st] == rel
        assert REGISTRY[st] is IL.IORSLarge
        assert IL.MODE_HOOKS[rel] == f"_sample_{rel}" and callable(getattr(IL.IORSLarge, f"_sample_{rel}"))
    assert IL.RELATION_OF_SUBTASK[ONTOP] == "ontop" and not {"ontop", "under"} & set(IL.MODE_HOOKS)


def test_iddls_are_ontop_large_with_the_relation_as_goal():
    iddls = I.load_all()
    o = iddls[ONTOP]
    for rel, st in NEW.items():
        d = iddls[st]
        assert d.family == IORS and d.alias_template == st + "-{target}-{partner}-{support}"
        assert [rel, "?target", "?partner"] in d.goal and ["not", [rel, "target", "partner"]] in d.init
        assert d.roles == o.roles and d.measure == o.measure
        assert d.require == ([] if rel == "nextto" else o.require)  # nextto's target is pushed by the base: no grasp
        assert [a for a in d.goal if a[0] != rel] == [a for a in o.goal if a[0] != "ontop"]


def test_subtask_lookups():
    for st in NEW.values():
        alias = f"{st}-apple-sink-floor"
        assert PB.SUBTASK_FAMILY[st] == (IORS, IORS) and PB.subtask_of_alias(alias) == st
        assert SC.subtask_of_alias(alias) == st and SC.family_of_subtask(st) == IORS and st in SC.REQUIRED_BLOCKS
        assert VB.V1_KEY[st][1] == f"{IORS}/{st.split('-', 1)[1]}.iddl"
    # the small-scale subtasks still resolve to themselves
    assert PB.subtask_of_alias(f"{IORS}-inside-apple-bowl-breakfast_table") == f"{IORS}-inside"
    assert SC.subtask_of_alias(f"{IORS}-nextto-spatula-soup_ladle-breakfast_table") == f"{IORS}-nextto"


def test_program_check_stance_lists():
    ontop = PC.subtask_blocks(ONTOP)
    assert PC.subtask_blocks(NEW["inside"]) == ontop
    assert PC.subtask_blocks(NEW["touching"]) == ontop + ("contacts.push",)
    assert PC.subtask_blocks(NEW["nextto"]) == ("stances.init", "stances.manipulation", "stances.placement", "transitions", "paths", "contacts.push")
    # ``...-inside_large_scale`` also starts with ``...-inside``: the small-scale lists are unchanged
    assert PC.subtask_blocks(f"{IORS}-inside") == ("stances.init", "stances.manipulation", "transitions", "grasps")
    assert PC.subtask_blocks(f"{IORS}-touching") == ("stances.init", "stances.manipulation", "transitions", "grasps", "contacts.push")


def test_generation_defaults_are_ontop_large_copies():
    ontop = GD.defaults_for(ONTOP)
    for rel, st in NEW.items():
        g = GD.defaults_for(st)
        assert set(g) == set(ontop)
        assert {k: v for k, v in g.items() if k != "goal_drop_m"} == {k: v for k, v in ontop.items() if k != "goal_drop_m"}
        assert g["goal_drop_m"] == (0.003 if rel == "touching" else ontop["goal_drop_m"])  # touching ends on the floor, as under


def test_init_surface_chains():
    for st in NEW.values():
        assert GI.INIT_SURFACE_FALLBACK_SUBTASKS[st] == ("floor",)


def test_grounder_skips_the_picks_of_a_subtask_without_candidates_rows():
    from ispatialgym.sampling.customs import Customs

    icl = GI.ICL()
    g = GI.Grounder(I.load_all(), icl, Customs(), options=GI.Options(dry_run=True, representative_test=False))
    for st in NEW.values():
        if any(icl.has_role(st, r) for r in ("target", "partner", "support")):
            continue  # its candidates rows have landed
        assert g.combinations(st) == []
        assert {why.split(":")[0] for _, why in g.report.skipped.get(st, [])} <= {"no_candidates"}


def _mode_landed(relation: str) -> bool:
    """The relation's mode has landed: its hook returns the mode class instead of rejecting (its program and sampler are tested with it)."""
    try:
        getattr(IL.IORSLarge, IL.MODE_HOOKS[relation])(_bare(relation))
    except Reject:
        return False
    return True


def test_programs_route_each_relation_to_its_own_branch():
    for rel, st in NEW.items():
        fn, kwargs = P.program_for(st)
        assert fn is P.program_iors_large and kwargs == {"relation": rel}
        if _mode_landed(rel):
            continue
        with pytest.raises(NotImplementedError, match=f"relation_not_implemented: IORS large-scale '{rel}'"):
            fn(None, **kwargs)


def _bare(relation: str) -> IL.IORSLarge:
    """An IORSLarge without a simulator: only what the template setup reads before the relation's flow."""
    s = IL.IORSLarge.__new__(IL.IORSLarge)
    s.relation, s.template_facts, s.template_meta = relation, {}, {}
    return s


class FloorBound(Exception):
    """``bind_floor_v2`` reached: the ontop / under binding ran."""


def _floor_bound(self):
    raise FloorBound(self.relation)


@pytest.mark.parametrize("relation", sorted(NEW))
def test_sampler_hook_rejects_relation_not_implemented(relation, monkeypatch):
    if _mode_landed(relation):
        pytest.skip(f"the {relation} mode has landed")
    monkeypatch.setattr(IL.IORSLarge, "bind_floor_v2", _floor_bound)
    s = _bare(relation)
    with pytest.raises(Reject, match=f"'{relation}'") as e:
        s.bind_scene_roles()  # the first stage of the template setup
    assert e.value.reason == "relation_not_implemented"
    assert type(s) is IL.IORSLarge and s.relation_mode is None


def test_ontop_and_under_keep_their_flow(monkeypatch):
    monkeypatch.setattr(IL.IORSLarge, "bind_floor_v2", _floor_bound)
    calls = []
    monkeypatch.setattr(IL.IORSLarge, "_prepare_bindings", lambda self: calls.append(self.relation))
    for rel in ("ontop", "under"):
        s = _bare(rel)
        with pytest.raises(FloorBound):
            s.bind_scene_roles()
        s.prepare_bindings()
        assert type(s) is IL.IORSLarge and s.relation_mode is None
    assert calls == ["ontop", "under"]


def test_adopted_mode_takes_over_the_stages(monkeypatch):
    class NexttoMode:
        def bind_scene_roles(self):
            self.template_facts["bound_by"] = "nextto mode"  # e.g. a scene target bound here

        def prepare_bindings(self):
            self.template_facts["prepared_by"] = "nextto mode"

        def sample_instance(self, rng):
            return f"{self.relation} draft"

    monkeypatch.setattr(IL.IORSLarge, "_sample_nextto", lambda self: NexttoMode)
    s = _bare("nextto")
    s.bind_scene_roles()
    s.prepare_bindings()
    assert s.relation_mode is NexttoMode and isinstance(s, NexttoMode) and isinstance(s, IL.IORSLarge)
    assert s.template_facts == {"bound_by": "nextto mode", "prepared_by": "nextto mode"} and s.sample_instance(None) == "nextto draft"
    assert type(_bare("nextto")) is IL.IORSLarge  # the class itself is untouched


def test_mode_without_overrides_runs_the_ontop_under_flow_with_its_helpers(monkeypatch):
    class InsideMode:
        def prepare_partner(self):
            return "inside partner geometry"

    monkeypatch.setattr(IL.IORSLarge, "bind_floor_v2", _floor_bound)
    calls = []
    monkeypatch.setattr(IL.IORSLarge, "_sample_inside", lambda self: InsideMode)
    monkeypatch.setattr(IL.IORSLarge, "_prepare_bindings", lambda self: calls.append(self.prepare_partner()))
    s = _bare("inside")
    with pytest.raises(FloorBound):
        s.bind_scene_roles()  # adopted, then IORSLarge's own binding
    s.prepare_bindings()
    assert calls == ["inside partner geometry"] and s.relation_mode is InsideMode and "template_facts" in s.template_meta
