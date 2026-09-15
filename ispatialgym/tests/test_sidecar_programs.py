"""Oracle sidecar schema (``isg-oracle-v1``), action declarations and per-subtask programs — no simulator.

One hand-written sidecar per subtask lives under ``data/sidecars/<subtask>.json``. The expected action sequences below
are the rows of ``oracle_primitives_design.md`` §6.2 with each fixture's conditionals resolved (which stance pairs have a
path and / or a transition recorded, which blocks flip / need a yaw-align / carry a row or column brace, ...).

``targets()`` resolves poses through ``ispatialgym.robot.conventions`` (TODO F5, written in parallel); those tests check
keys and shapes only and are skipped with a message when the module is not importable.
"""

import copy
import glob
import json
import os

import pytest

from ispatialgym.oracle import actions as A
from ispatialgym.oracle.programs import PROGRAMS, expand
from ispatialgym.oracle.sidecar import (
    ARMS,
    REQUIRED_BLOCKS,
    SUBTASKS,
    Pose,
    Sidecar,
    SidecarError,
    alias_templates,
    family_of_subtask,
    sha256_of_file,
    subtask_of_alias,
)

DATA = os.path.join(os.path.dirname(__file__), "data", "sidecars")

try:
    import ispatialgym.robot.conventions  # noqa: F401

    HAVE_CONVENTIONS = True
except ImportError:
    HAVE_CONVENTIONS = False

needs_conventions = pytest.mark.skipif(
    not HAVE_CONVENTIONS,
    reason="ispatialgym.robot.conventions (TODO F5) is not importable: targets() key/shape tests skipped",
)

# ----------------------------------------------------------------------------------------------------------------
# Expected action-name sequences per subtask (oracle doc §6.2, conditionals resolved from the fixture)
# ----------------------------------------------------------------------------------------------------------------

_TABLE_IN = ["transition"]  # tabletop init -> manipulation: same edge / lateral, only a transition is recorded
_WALL_IN = ["navigate", "transition"]  # Prompt-0 stances: a path and a transition
_PICK = ["grasp", "lift"]
_PLACE = ["place", "retreat"]
_BRACE_PUSH = ["brace", "push", "retreat", "retreat"]  # brace (held) -> push -> retreat push arm -> release + retreat brace arm

EXPECTED_SEQUENCE = {
    "camera_relocalization-standard": ["navigate", "set_torso"],
    "camera_relocalization-obstructed": ["navigate", "set_torso"],
    "object_scaling": ["reorient"],
    "mental_rotation": ["reorient"],
    "disembedding": _TABLE_IN + _PICK + ["navigate"] + _PLACE,
    "configuration_matching-block_design": _TABLE_IN
    + ["grasp", "lift", "reorient"] + _PLACE  # anchor: yaw-align only, its own init position, nothing to push against
    + ["grasp", "lift", "reorient"] + _PLACE + _BRACE_PUSH  # top-right: flips; pushed left against the anchor (row brace)
    + ["grasp", "lift"] + _PLACE + _BRACE_PUSH  # bottom-left: no yaw, no flip; pushed up against the anchor (column brace)
    + ["grasp", "lift", "reorient"] + _PLACE + _BRACE_PUSH + _BRACE_PUSH,  # bottom-right: flips; row push, then column push
    "configuration_matching-articulation_open_large_scale": _WALL_IN + ["grasp", "base_follow", "retreat"],
    "configuration_matching-articulation_close_large_scale": _WALL_IN + ["grasp", "base_follow", "retreat"],
    "configuration_matching-articulation_open_small_scale": _TABLE_IN + ["brace", "grasp", "base_follow", "retreat", "retreat"],
    "configuration_matching-articulation_close_small_scale": _TABLE_IN + ["brace", "grasp", "base_follow", "retreat", "retreat"],
    "configuration_matching-attachment": _TABLE_IN + _PICK + ["reorient", "brace", "place", "retreat", "retreat"],
    "alignment-world_vertical": _WALL_IN + _BRACE_PUSH,
    "alignment-world_horizontal": _WALL_IN + _BRACE_PUSH,
    "alignment-axial": _TABLE_IN + _BRACE_PUSH,
    "inter_object_relation_satisfaction-ontop_large_scale": ["transition"] + _PICK + ["navigate", "transition"] + _PLACE,
    "inter_object_relation_satisfaction-under_large_scale": ["transition"] + _PICK + ["navigate"] + _PLACE,  # B and C share the floor-band trunk
    "inter_object_relation_satisfaction-ontop_small_scale": _TABLE_IN + _PICK + _PLACE,
    "inter_object_relation_satisfaction-under_small_scale": _TABLE_IN + _PICK + _PLACE,
    "inter_object_relation_satisfaction-inside": _TABLE_IN + _PICK + _PLACE,
    "inter_object_relation_satisfaction-nextto": _TABLE_IN + _PICK + _PLACE,
    "inter_object_relation_satisfaction-touching": _TABLE_IN + _PICK + _PLACE + _BRACE_PUSH,
}

# (name, arms, objects) of every action for the subtasks whose arm / object assignment the spec fixes.
EXPECTED_ARMS_OBJECTS = {
    "camera_relocalization-obstructed": [("navigate", (), ()), ("set_torso", ("left", "right"), ())],
    "camera_relocalization-standard": [("navigate", (), ()), ("set_torso", (), ())],
    "object_scaling": [("reorient", ("right",), ("beaker.n.02_1",))],
    "mental_rotation": [("reorient", ("right",), ("mug.n.04_1",))],
    "disembedding": [
        ("transition", (), ()),
        ("grasp", ("right",), ("bottle_of_beer.n.01_1",)),
        ("lift", ("right",), ("bottle_of_beer.n.01_1",)),
        ("navigate", (), ("bottle_of_beer.n.01_1",)),
        ("place", ("right",), ("bottle_of_beer.n.01_1", "tray.n.01_2")),
        ("retreat", ("right",), ("bottle_of_beer.n.01_1",)),
    ],
    "configuration_matching-articulation_open_small_scale": [
        ("transition", (), ()),
        ("brace", ("left",), ("toolbox.n.01_1",)),
        ("grasp", ("right",), ("toolbox.n.01_1",)),
        ("base_follow", ("right",), ("toolbox.n.01_1",)),
        ("retreat", ("right",), ("toolbox.n.01_1",)),
        ("retreat", ("left",), ("toolbox.n.01_1",)),
    ],
    "configuration_matching-attachment": [
        ("transition", (), ()),
        ("grasp", ("right",), ("cap.n.02_1",)),
        ("lift", ("right",), ("cap.n.02_1",)),
        ("reorient", ("right",), ("cap.n.02_1",)),
        ("brace", ("left",), ("lemonade_bottle.n.01_1",)),
        ("place", ("right",), ("cap.n.02_1", "lemonade_bottle.n.01_1")),
        ("retreat", ("right",), ("cap.n.02_1",)),
        ("retreat", ("left",), ("lemonade_bottle.n.01_1",)),
    ],
    "alignment-axial": [
        ("transition", (), ()),
        ("brace", ("left",), ("paperback_book.n.01_1",)),
        ("push", ("right",), ("paperback_book.n.01_1",)),
        ("retreat", ("right",), ("paperback_book.n.01_1",)),
        ("retreat", ("left",), ("paperback_book.n.01_1",)),
    ],
    "inter_object_relation_satisfaction-touching": [
        ("transition", (), ()),
        ("grasp", ("right",), ("can_of_soda.n.01_1",)),
        ("lift", ("right",), ("can_of_soda.n.01_1",)),
        ("place", ("right",), ("can_of_soda.n.01_1", "mug.n.04_1")),
        ("retreat", ("right",), ("can_of_soda.n.01_1",)),
        ("brace", ("left",), ("mug.n.04_1",)),
        ("push", ("right",), ("can_of_soda.n.01_1",)),
        ("retreat", ("right",), ("can_of_soda.n.01_1",)),
        ("retreat", ("left",), ("mug.n.04_1",)),
    ],
}


def fixture_path(subtask):
    return os.path.join(DATA, f"{subtask}.json")


def load(subtask) -> Sidecar:
    return Sidecar.load(fixture_path(subtask))


def raw(subtask) -> dict:
    with open(fixture_path(subtask)) as f:
        return json.load(f)


class StubState:
    """Object poses for ``targets()``: the tests check keys and shapes, not numbers."""

    def object_pose(self, name):
        import torch as th

        return th.tensor([7.0, 3.1, 0.85]), th.tensor([0.0, 0.0, 0.0, 1.0])


# ----------------------------------------------------------------------------------------------------------------
# Registry and fixtures
# ----------------------------------------------------------------------------------------------------------------


def test_fixture_set_covers_every_subtask():
    files = {os.path.basename(p)[: -len(".json")] for p in glob.glob(os.path.join(DATA, "*.json"))}
    assert files == set(SUBTASKS)
    assert len(SUBTASKS) == 21


def test_subtask_registry_is_consistent():
    assert set(alias_templates()) == set(SUBTASKS) == set(PROGRAMS) == set(REQUIRED_BLOCKS)
    assert set(EXPECTED_SEQUENCE) == set(SUBTASKS)
    # 14 rows of §6.2 -> 13 program functions (camera relocalization standard / obstructed share one with a flag)
    assert len({fn for fn, _ in PROGRAMS.values()}) == 13


@pytest.mark.parametrize("subtask", SUBTASKS)
def test_sidecar_loads_and_validates(subtask):
    sc = load(subtask)
    assert sc.schema == "isg-oracle-v1"
    assert sc.subtask == subtask
    assert sc.task_family == family_of_subtask(subtask)
    assert sc.problems() == []
    assert sc.validate() is sc
    for block in REQUIRED_BLOCKS[subtask]:
        assert sc.has_block(block), block


@pytest.mark.parametrize("subtask", SUBTASKS)
def test_sidecar_round_trips(subtask, tmp_path):
    sc = load(subtask)
    d = sc.to_dict()
    again = Sidecar.from_dict(json.loads(json.dumps(d)))
    assert again.to_dict() == d
    assert again == sc
    out = tmp_path / "x.oracle.json"
    sc.save(str(out))
    assert Sidecar.load(str(out)) == sc
    # unknown keys round-trip through ``extra`` (samplers may record diagnostics)
    d2 = copy.deepcopy(raw(subtask))
    d2["sampler_note"] = {"phase": 3}
    d2["language"]["objects"] = {k: dict(v, pixels=123) for k, v in d2["language"]["objects"].items()}
    sc2 = Sidecar.from_dict(d2)
    assert sc2.extra["sampler_note"] == {"phase": 3}
    assert sc2.to_dict()["sampler_note"] == {"phase": 3}
    assert all(v["pixels"] == 123 for v in sc2.to_dict()["language"]["objects"].values())


def test_subtask_of_alias():
    for subtask in SUBTASKS:
        assert subtask_of_alias(load(subtask).alias) == subtask
        _family, template = alias_templates()[subtask]
        assert subtask_of_alias(template) == subtask
    assert subtask_of_alias("configuration_matching-block_design-block_4-breakfast_table") == "configuration_matching-block_design"
    assert subtask_of_alias("disembedding-mug-bottle_of_beer_3_bottle_of_sake_1-breakfast_table") == "disembedding"
    for bad in ("", "camera_relocalization", "camera_relocalization-standard-extra", "alignment-diagonal-x-y", "disembedding-Mug-x-y", "nope"):
        with pytest.raises(SidecarError):
            subtask_of_alias(bad)


def test_verify_tro(tmp_path):
    tro = tmp_path / "inst-tro_state.json"
    tro.write_text('{"metadata": {}}')
    d = raw("object_scaling")
    d["tro_sha256"] = sha256_of_file(str(tro))
    sc = Sidecar.from_dict(d)
    assert sc.verify_tro(str(tro)) == d["tro_sha256"]
    tro.write_text('{"metadata": {"changed": 1}}')
    with pytest.raises(SidecarError, match="tro_sha256 mismatch"):
        sc.verify_tro(str(tro))


# ----------------------------------------------------------------------------------------------------------------
# Programs
# ----------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("subtask", SUBTASKS)
def test_expand_returns_the_expected_sequence(subtask):
    sc = load(subtask)
    actions = expand(sc.alias, sc)
    assert [a.name for a in actions] == EXPECTED_SEQUENCE[subtask]
    # the template alias resolves to the same program
    assert [a.name for a in expand(alias_templates()[subtask][1], sc)] == EXPECTED_SEQUENCE[subtask]


@pytest.mark.parametrize("subtask", SUBTASKS)
def test_actions_follow_the_contract(subtask):
    sc = load(subtask)
    actions = expand(sc.alias, sc)
    assert actions
    for action in actions:
        assert isinstance(action, A.Action)
        assert type(action) in A.VOCABULARY
        assert action.name == type(action).__dataclass_fields__["name"].default
        assert action.TARGET_KEYS
        assert set(action.arms) <= set(ARMS)
        assert set(action.objects) <= set(sc.language.objects), action.summary()
        if action.name in ("push", "brace"):
            assert action.qa_exempt == frozenset({A.QA_FAILED_GRASP})
        else:
            assert action.qa_exempt == frozenset()
        if action.name in ("grasp", "lift", "reorient", "place", "push", "brace", "base_follow", "retreat"):
            assert len(action.arms) == 1 and len(action.objects) >= 1
        if action.name in ("navigate", "transition"):
            assert action.arms == ()
        # declared, implemented by F2
        with pytest.raises(NotImplementedError):
            next(iter(action.execute(None, None)))
        with pytest.raises(NotImplementedError):
            action.postcondition(None)


@pytest.mark.parametrize("subtask", sorted(EXPECTED_ARMS_OBJECTS))
def test_arms_and_objects_per_action(subtask):
    sc = load(subtask)
    got = [(a.name, a.arms, a.objects) for a in expand(sc.alias, sc)]
    assert got == EXPECTED_ARMS_OBJECTS[subtask]


def test_block_design_program_details():
    sc = load("configuration_matching-block_design")
    actions = expand(sc.alias, sc)
    order = [b.instance for b in sorted(sc.blocks, key=lambda b: b.order)]
    grasps = [a for a in actions if a.name == "grasp"]
    assert [a.object for a in grasps] == order
    assert all(a.kind == "cube" and a.arm == "right" for a in grasps)
    assert all(a.arms == ("right",) for a in actions if a.name == "push")
    assert all(a.arms == ("left",) for a in actions if a.name == "brace")
    reoriented = [a.object for a in actions if a.name == "reorient"]
    assert reoriented == [b.instance for b in sorted(sc.blocks, key=lambda b: b.order) if b.reorients]
    # brace/push pairing: the brace holds an already placed block, the push moves the block just placed
    for brace, push in zip((a for a in actions if a.name == "brace"), (a for a in actions if a.name == "push")):
        assert brace.contact.object in order[: order.index(push.contact.object)]
    # a block without braces contributes no push
    d = raw("configuration_matching-block_design")
    for b in d["blocks"]:
        for k in ("brace_row", "push_row", "brace_col", "push_col"):
            b.pop(k, None)
    sc2 = Sidecar.from_dict(d)
    assert [a.name for a in expand(sc2.alias, sc2) if a.name in ("brace", "push")] == []
    # an unpaired brace is rejected
    d = raw("configuration_matching-block_design")
    d["blocks"][1].pop("push_row")
    assert any("push_row" in p for p in Sidecar.from_dict(d).problems())


def test_articulation_program_details():
    sc = load("configuration_matching-articulation_open_large_scale")
    follow = [a for a in expand(sc.alias, sc) if a.name == "base_follow"][0]
    assert follow.articulation.follow_mode == "base"
    assert [f for f, _ in follow.articulation.followed_fractions()] == [0.25, 0.5, 0.75]  # up to passive_from
    assert follow.direction == "open"
    grasp = [a for a in expand(sc.alias, sc) if a.name == "grasp"][0]
    assert grasp.kind == "handle" and grasp.articulation is sc.articulation
    d = raw("configuration_matching-articulation_open_large_scale")
    d["articulation"]["passive_from"] = None
    sc2 = Sidecar.from_dict(d)
    assert [f for f, _ in sc2.articulation.followed_fractions()] == [0.25, 0.5, 0.75, 1.0]
    sc_close = load("configuration_matching-articulation_close_large_scale")
    follow = [a for a in expand(sc_close.alias, sc_close) if a.name == "base_follow"][0]
    assert follow.articulation.follow_mode == "arm" and follow.direction == "close"
    # small-scale must move the arm; the brace arm is the other arm on the same object
    d = raw("configuration_matching-articulation_open_small_scale")
    d["articulation"]["follow_mode"] = "base"
    d["articulation"]["base_follow"] = {k: [7.1, 2.45, 1.5708] for k in d["articulation"]["waypoints"] if k != "0.0"}
    with pytest.raises(SidecarError, match="follow_mode"):
        expand(d["alias"], Sidecar.from_dict(d))
    d = raw("configuration_matching-articulation_open_small_scale")
    d["contacts"]["brace"]["arm"] = "right"
    with pytest.raises(SidecarError, match="brace"):
        expand(d["alias"], Sidecar.from_dict(d))


def test_camera_relocalization_program_details():
    std, obs = load("camera_relocalization-standard"), load("camera_relocalization-obstructed")
    nav_s, torso_s = expand(std.alias, std)
    nav_o, torso_o = expand(obs.alias, obs)
    assert nav_s.obstacles == () and nav_o.obstacles == ("hamper.n.02_1",)
    assert nav_o.objects == ()  # the obstacle is never an intended contact
    assert torso_s.goal.arms == "same" and torso_s.arms == ()
    assert torso_o.goal.arms == "untucked" and torso_o.arms == ("left", "right")
    d = raw("camera_relocalization-standard")
    d["paths"] = []
    with pytest.raises(SidecarError, match="paths"):
        expand(d["alias"], Sidecar.from_dict(d))


def test_travel_rule_between_stances():
    # a trunk change without a recorded transition is an error (spec §3.8) ...
    d = raw("inter_object_relation_satisfaction-ontop_small_scale")
    d["transitions"] = []
    with pytest.raises(SidecarError, match="transition"):
        expand(d["alias"], Sidecar.from_dict(d))
    # ... equal trunks and base poses need nothing ...
    d = raw("inter_object_relation_satisfaction-ontop_small_scale")
    d["transitions"] = []
    d["stances"]["manipulation"].update({k: d["stances"]["init"][k] for k in ("base_pose2d", "translate", "tilt")})
    names = [a.name for a in expand(d["alias"], Sidecar.from_dict(d))]
    assert names == ["grasp", "lift", "place", "retreat"]
    # ... a path alone is fine when only the base moves; the carried object must match
    d = raw("disembedding")
    d["paths"][0]["carried"] = None
    with pytest.raises(SidecarError, match="carried"):
        expand(d["alias"], Sidecar.from_dict(d))
    # alias must belong to the sidecar
    sc = load("disembedding")
    with pytest.raises(SidecarError, match="alias"):
        expand("object_scaling-mug", sc)


def test_iors_place_kinds_and_relations():
    kinds = {}
    for rel in ("ontop_small_scale", "under_small_scale", "inside", "nextto", "touching", "ontop_large_scale", "under_large_scale"):
        sc = load(f"inter_object_relation_satisfaction-{rel}")
        place = [a for a in expand(sc.alias, sc) if a.name == "place"][0]
        kinds[rel] = (place.kind, place.relation)
    assert kinds["inside"] == ("drop", "into")
    assert kinds["nextto"] == ("support", "next to") and kinds["touching"] == ("support", "next to")
    assert kinds["under_large_scale"] == ("support", "under")
    assert kinds["ontop_small_scale"] == kinds["under_small_scale"] == kinds["ontop_large_scale"] == ("support", "on")
    att = load("configuration_matching-attachment")
    assert [a.kind for a in expand(att.alias, att) if a.name == "place"] == ["attach"]


# ----------------------------------------------------------------------------------------------------------------
# Annotations
# ----------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("subtask", SUBTASKS)
def test_annotate_returns_two_texts_with_display_names(subtask):
    sc = load(subtask)
    for action in expand(sc.alias, sc):
        fine, numbers = action.annotate(sc)
        assert isinstance(fine, str) and fine.strip()
        assert isinstance(numbers, str) and numbers.strip()
        assert any(ch.isdigit() for ch in numbers), numbers
        for obj in action.objects:
            entry = sc.language.objects[obj]
            phrase = entry.descriptor or f"the {entry.display_name}"
            assert phrase in fine and phrase in numbers, (action.summary(), fine)
        if isinstance(action, A.Navigate) and action.obstacles:
            assert all(sc.language.objects[o].descriptor in fine for o in action.obstacles)
        assert all(obj not in fine for obj in action.objects)  # bddl ids never leak into the text


def test_annotation_numbers():
    sc = load("mental_rotation")
    _, numbers = expand(sc.alias, sc)[0].annotate(sc)
    assert "77°" in numbers
    sc = load("object_scaling")
    _, numbers = expand(sc.alias, sc)[0].annotate(sc)
    assert "15 cm" in numbers
    sc = load("configuration_matching-articulation_open_large_scale")
    _, numbers = [a for a in expand(sc.alias, sc) if a.name == "base_follow"][0].annotate(sc)
    assert "77°" in numbers and "75 %" in numbers and "driving the base" in numbers
    sc = load("inter_object_relation_satisfaction-touching")
    push = [a for a in expand(sc.alias, sc) if a.name == "push"][0]
    fine, numbers = push.annotate(sc)
    assert "to the left" in fine and "1.8 cm" in numbers  # partner on the robot's left


def test_direction_words():
    yaw = 1.5708  # facing +y: forward = +y, left = -x
    assert A.direction_word([0, 1, 0], yaw) == "forward"
    assert A.direction_word([-1, 0, 0], yaw) == "to the left"
    assert A.direction_word([1, 0, 0], yaw) == "to the right"
    assert A.direction_word([0, 0, 1], yaw) == "up"
    assert A.direction_word([0.1, 0.0, -0.9], None) == "down"


# ----------------------------------------------------------------------------------------------------------------
# targets(): keys and shapes (needs the conventions module)
# ----------------------------------------------------------------------------------------------------------------


def _check_value(key, value, where):
    if value is None:
        assert key in ("base_poses", "recorded", "carried"), where
        return
    if isinstance(value, Pose):
        assert len(value.position) == 3 and len(value.orientation) == 4, where
    elif isinstance(value, list) and value and isinstance(value[0], Pose):
        for p in value:
            assert len(p.position) == 3 and len(p.orientation) == 4, where
    elif key in ("waypoints", "base_poses"):
        assert value and all(len(w) == 3 for w in value), where
    elif key in ("goal", "base", "drive_back", "drive_in", "pivot", "direction"):
        assert len(value) == 3, where
    elif key == "trunk":
        assert len(value) == 2, where
    elif key in ("fractions", "q_values"):
        assert value and all(isinstance(v, float) for v in value), where
    elif key in ("posture", "arms", "mode", "carried"):
        assert isinstance(value, str) and value, where
    elif key == "distance_m":
        assert value > 0, where
    else:
        raise AssertionError(f"unexpected target key {key!r} at {where}")


@needs_conventions
@pytest.mark.parametrize("subtask", SUBTASKS)
def test_targets_keys_and_shapes(subtask):
    sc = load(subtask)
    state = StubState()
    for i, action in enumerate(expand(sc.alias, sc)):
        targets = action.targets(sc, state)
        where = f"{subtask}[{i}] {action.name}"
        assert set(targets) == set(action.TARGET_KEYS), where
        for key, value in targets.items():
            _check_value(key, value, f"{where}.{key}")
        end = action.end_pose(sc, state)
        assert end is None or isinstance(end, Pose)


@needs_conventions
def test_targets_chain_through_end_poses():
    sc = load("inter_object_relation_satisfaction-ontop_small_scale")
    state = StubState()
    actions = expand(sc.alias, sc)
    grasp, lift = actions[1], actions[2]
    g = grasp.targets(sc, state)
    assert g["approach"][-1].position == pytest.approx(g["grasp"].position, abs=1e-5)
    lift_wps = lift.targets(sc, state)["lift"]  # start excluded, goal last (conventions waypoint rule)
    assert lift_wps[-1].position[:2] == pytest.approx(g["grasp"].position[:2], abs=1e-5)
    assert lift_wps[-1].position[2] - g["grasp"].position[2] == pytest.approx(lift.carry_height, abs=1e-5)
    assert g["grasp"].position[2] < lift_wps[0].position[2] < lift_wps[-1].position[2]
    assert lift.end_pose(sc, state).position == pytest.approx(lift_wps[-1].position, abs=1e-5)
    place, retreat = actions[3], actions[4]
    p = place.targets(sc, state)
    assert p["descent"][-1].position == pytest.approx(p["place"].position, abs=1e-5)
    assert p["retreat"][-1].position == pytest.approx(place.end_pose(sc, state).position, abs=1e-5)
    r = retreat.targets(sc, state)["retreat"]
    assert r[-1].position == pytest.approx(retreat.end_pose(sc, state).position, abs=1e-5)
    # the retreat leaves from where the place ended and moves RETREAT_LITTLE further along -a
    start = place.end_pose(sc, state)
    assert sum((a - b) ** 2 for a, b in zip(r[-1].position, start.position)) ** 0.5 == pytest.approx(A.RETREAT_LITTLE, abs=1e-5)


# ----------------------------------------------------------------------------------------------------------------
# Validation rejects
# ----------------------------------------------------------------------------------------------------------------


def _problems(subtask, mutate):
    d = raw(subtask)
    mutate(d)
    return Sidecar.from_dict(d).problems()


def test_from_dict_rejects_structural_errors():
    d = raw("disembedding")
    d["schema"] = "isg-oracle-v0"
    with pytest.raises(SidecarError, match="schema"):
        Sidecar.from_dict(d)
    d = raw("disembedding")
    d["grasps"][0]["grasp"]["orientation"] = [0.5, 0.5, 0.5, 0.9]
    with pytest.raises(SidecarError, match="unit quaternion"):
        Sidecar.from_dict(d)
    d = raw("disembedding")
    del d["stances"]["init"]["translate"]
    with pytest.raises(SidecarError, match="translate"):
        Sidecar.from_dict(d)
    d = raw("disembedding")
    d["paths"][0]["waypoints"][0] = [1.0, 2.0]
    with pytest.raises(SidecarError, match="waypoints"):
        Sidecar.from_dict(d)
    d = raw("disembedding")
    del d["language"]
    with pytest.raises(SidecarError, match="language"):
        Sidecar.from_dict(d)


@pytest.mark.parametrize(
    "subtask,mutate,match",
    [
        ("disembedding", lambda d: d.update(alias="alignment-diagonal-x-y"), "matches no iddl"),
        ("disembedding", lambda d: d.update(task_family="alignment"), "task_family"),
        ("disembedding", lambda d: d.update(tro_sha256="abc"), "tro_sha256"),
        ("disembedding", lambda d: d["stances"].pop("placement"), "required block 'stances.placement'"),
        ("disembedding", lambda d: d.update(grasps=[]), "required block 'grasps'"),
        ("disembedding", lambda d: d["stances"]["init"].update(translate=1.3), "translate"),
        ("disembedding", lambda d: d["transitions"][0].update(order=["trunk", "drive-in"]), "order"),
        ("disembedding", lambda d: d["paths"][0].update(to="nowhere"), "nowhere"),
        ("disembedding", lambda d: d["paths"][0].update(mode="teleport"), "mode"),
        ("disembedding", lambda d: d["grasps"][0].update(arm="middle"), "arm"),
        ("disembedding", lambda d: d["grasps"][0].pop("place_on"), "place_on"),
        ("disembedding", lambda d: d["language"]["objects"].pop("tray.n.01_2"), "language.objects has no entry"),
        ("camera_relocalization-standard", lambda d: d.update(path_bucket_m=None), "path_bucket_m"),
        ("camera_relocalization-standard", lambda d: d["goal"].update(arms="tucked"), "goal.arms"),
        ("alignment-axial", lambda d: d["contacts"]["brace"].update(arm="right"), "same arm"),
        ("alignment-axial", lambda d: d["contacts"]["push"].update(direction=[0.5, 0.5, 0.0]), "unit vector"),
        ("alignment-axial", lambda d: d["contacts"]["push"].pop("mid"), "straight push"),
        ("inter_object_relation_satisfaction-touching", lambda d: d["contacts"]["push"]["goal"]["position"].__setitem__(0, 9.0), "straight push"),
        ("configuration_matching-block_design", lambda d: d["blocks"][1].update(order=0), "orders"),
        ("configuration_matching-block_design", lambda d: d["blocks"][1].update(flip_sign=0), "flip_axis"),
        ("configuration_matching-block_design", lambda d: d["blocks"][0].update(face_pair="w"), "face_pair"),
        ("configuration_matching-articulation_open_large_scale", lambda d: d["articulation"].pop("base_follow"), "base_follow"),
        ("configuration_matching-articulation_open_large_scale", lambda d: d["articulation"].update(passive_from=0.6), "passive_from"),
        ("configuration_matching-articulation_open_large_scale", lambda d: d["articulation"]["handle_reach"].update({"0.5": False}), "unreachable"),
        ("configuration_matching-articulation_open_large_scale", lambda d: d["articulation"]["waypoints"].pop("0.0"), "fraction 0.0"),
        ("configuration_matching-articulation_open_large_scale", lambda d: d["articulation"].update(joint_type="ball"), "joint_type"),
        ("configuration_matching-articulation_close_large_scale", lambda d: d["articulation"].update(base_follow={"0.25": [1.0, 2.0, 0.0]}), "follow_mode is 'arm'"),
    ],
)
def test_validation_rejects(subtask, mutate, match):
    problems = _problems(subtask, mutate)
    assert problems, match
    assert any(match in p for p in problems), problems
    with pytest.raises(SidecarError):
        Sidecar.from_dict((lambda d: (mutate(d), d)[1])(raw(subtask))).validate()
