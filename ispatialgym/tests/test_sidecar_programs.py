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
import math
import os

import pytest

from ispatialgym.oracle import actions as A
from ispatialgym.oracle.programs import PROGRAMS, expand
from ispatialgym.robot import base_path as BP
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
VARIANTS = os.path.join(os.path.dirname(__file__), "data", "sidecar_variants")  # variant fixtures (push-only axial, push-under, ...)

try:
    import torch as th
    from ispatialgym.robot import conventions as C

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
    # blocks: ``yaw_deg`` is the wrist yaw of the cube grasp (not an in-hand rotation); the fixture records no
    # ``yaw_align_deg`` (the table-alignment yaw ψ the sampler writes next to it), so only the flipping blocks reorient
    "configuration_matching-block_design": _TABLE_IN
    + ["grasp", "lift"] + _PLACE  # anchor: no flip, its own init position, nothing to push against
    + ["grasp", "lift", "reorient"] + _PLACE + _BRACE_PUSH  # top-right: flips; pushed left against the anchor (row brace)
    + ["grasp", "lift"] + _PLACE + _BRACE_PUSH  # bottom-left: no flip; pushed up against the anchor (column brace)
    + ["grasp", "lift", "reorient"] + _PLACE + _BRACE_PUSH + _BRACE_PUSH,  # bottom-right: flips; row push, then column push
    "configuration_matching-articulation_open_large_scale": _WALL_IN + ["grasp", "base_follow", "retreat"],
    "configuration_matching-articulation_close_large_scale": _WALL_IN + ["grasp", "base_follow", "retreat"],
    "configuration_matching-articulation_open_small_scale": _TABLE_IN + ["brace", "grasp", "base_follow", "retreat", "retreat"],
    "configuration_matching-articulation_close_small_scale": _TABLE_IN + ["brace", "grasp", "base_follow", "retreat", "retreat"],
    "configuration_matching-attachment": _TABLE_IN + _PICK + ["reorient", "brace", "place", "retreat", "retreat"],
    "alignment-world_vertical": _WALL_IN + _BRACE_PUSH,
    "alignment-axial": _TABLE_IN + _BRACE_PUSH,
    "inter_object_relation_satisfaction-ontop_large_scale": ["transition"] + _PICK + ["navigate", "transition"] + _PLACE,
    "inter_object_relation_satisfaction-inside_large_scale": ["transition"] + _PICK + ["navigate", "transition"] + _PLACE,  # the place is a drop
    "inter_object_relation_satisfaction-under_large_scale": ["transition"] + _PICK + ["navigate"] + _PLACE,  # B and C share the floor-band trunk
    # the drop from stance C, the deep push crouch entered with empty hands (trunk-only transition), the floor push against the face
    "inter_object_relation_satisfaction-touching_large_scale": ["transition"] + _PICK + ["navigate"] + _PLACE + ["transition", "push", "retreat"],
    "inter_object_relation_satisfaction-nextto_large_scale": _WALL_IN + ["base_push", "retreat"],  # no grasp: the base pushes the floor-standing target
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


#: registered 2026-10-12 ahead of their modes: a subtask among them gets its fixture and expected sequence with its mode
MODES_PENDING = ("inter_object_relation_satisfaction-inside_large_scale", "inter_object_relation_satisfaction-touching_large_scale",
                 "inter_object_relation_satisfaction-nextto_large_scale")
FIXTURE_SUBTASKS = tuple(s for s in SUBTASKS if os.path.exists(fixture_path(s)))


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
    assert files == set(FIXTURE_SUBTASKS)
    assert set(SUBTASKS) - files <= set(MODES_PENDING)
    assert len(SUBTASKS) == 23


def test_subtask_registry_is_consistent():
    assert set(alias_templates()) == set(SUBTASKS) == set(PROGRAMS) == set(REQUIRED_BLOCKS)
    assert set(EXPECTED_SEQUENCE) == set(FIXTURE_SUBTASKS)
    # 14 rows of §6.2 -> 13 program functions (camera relocalization standard / obstructed share one with a flag)
    assert len({fn for fn, _ in PROGRAMS.values()}) == 13


@pytest.mark.parametrize("subtask", FIXTURE_SUBTASKS)
def test_sidecar_loads_and_validates(subtask):
    sc = load(subtask)
    assert sc.schema == "isg-oracle-v1"
    assert sc.subtask == subtask
    assert sc.task_family == family_of_subtask(subtask)
    assert sc.problems() == []
    assert sc.validate() is sc
    for block in REQUIRED_BLOCKS[subtask]:
        assert sc.has_block(block), block


@pytest.mark.parametrize("subtask", FIXTURE_SUBTASKS)
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
        if subtask in FIXTURE_SUBTASKS:
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


@pytest.mark.parametrize("subtask", FIXTURE_SUBTASKS)
def test_expand_returns_the_expected_sequence(subtask):
    sc = load(subtask)
    actions = expand(sc.alias, sc)
    assert [a.name for a in actions] == EXPECTED_SEQUENCE[subtask]
    # the template alias resolves to the same program
    assert [a.name for a in expand(alias_templates()[subtask][1], sc)] == EXPECTED_SEQUENCE[subtask]


@pytest.mark.parametrize("subtask", FIXTURE_SUBTASKS)
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
        if action.name in ("push", "brace", "base_push"):
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
    assert reoriented == [b.instance for b in sorted(sc.blocks, key=lambda b: b.order) if A.block_reorients(b)]
    assert reoriented == [b.instance for b in sorted(sc.blocks, key=lambda b: b.order) if b.flips]  # no yaw_align_deg recorded
    assert all(a.mode == "flip" and a.block.flips for a in actions if a.name == "reorient")
    # a non-zero table-alignment yaw ψ (``yaw_align_deg``) makes a block reorient even without a flip; the wrist yaw does not
    d = raw("configuration_matching-block_design")
    assert d["blocks"][0]["flip_sign"] == 0 and abs(d["blocks"][0]["yaw_deg"]) > 1.0
    d["blocks"][0]["yaw_align_deg"] = 12.0
    sc_psi = Sidecar.from_dict(d)
    assert A.yaw_align_deg(sc_psi.blocks[0]) == 12.0 and A.block_reorients(sc_psi.blocks[0])
    assert [a.name for a in expand(sc_psi.alias, sc_psi)][:7] == _TABLE_IN + ["grasp", "lift", "reorient", "place", "retreat", "grasp"]
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


def test_block_design_flip_on_the_table_sub_chain():
    """Owner 2026-09-16 17:10 item 1: a block with a recorded ``table_flip`` sub-chain is flipped on the table —
    grasp → lift → place(intermediate) → retreat → grasp(the other pitch class) → lift → place(slot) → retreat — with
    no in-hand reorient; the re-grasp is a cube grasp on the recorded face pair / tilt / yaw with its own carry height;
    blocks without the record keep the in-hand flip. A record missing its re-grasp is a sidecar error."""
    d = raw("configuration_matching-block_design")
    flipping = [b for b in d["blocks"] if b["flip_sign"] != 0]
    assert flipping, "the fixture needs a flipping block"
    b = flipping[0]
    b["table_flip"] = {
        "cell": [0.55, -0.25],
        "place": {"position": [0.55, -0.25, 0.86], "orientation": [0.0, 0.7071, 0.0, 0.7071]},
        "pre_place": {"position": [0.55, -0.25, 0.96], "orientation": [0.0, 0.7071, 0.0, 0.7071]},
        "regrasp": {"face_pair": "y" if b["face_pair"] != "y" else "x", "tilt_pick_deg": 0.0, "yaw_deg": 5.0, "carry_height": 0.12,
                    "pick": {"position": [0.55, -0.25, 0.85], "orientation": [0.0, 0.7071, 0.0, 0.7071]}},
    }
    sc = Sidecar.from_dict(d)
    actions = expand(sc.alias, sc)
    first = next(i for i, a in enumerate(actions) if a.name == "grasp" and a.objects[0] == b["instance"])
    names = []
    for a in actions[first:]:
        if a.name in ("brace", "push") or (a.name == "grasp" and a.objects[0] != b["instance"]):
            break
        names.append(a.name)
    assert names == ["grasp", "lift", "place", "retreat", "grasp", "lift", "place", "retreat"]
    grasps = [a for a in actions if a.name == "grasp" and a.object == b["instance"]]
    assert grasps[0].block.face_pair == b["face_pair"] and grasps[1].block.face_pair == b["table_flip"]["regrasp"]["face_pair"]
    assert grasps[1].block.flip_sign == 0 and grasps[1].block.yaw_deg == 5.0 and grasps[1].block.carry_height == 0.12
    places = [a for a in actions if a.name == "place" and a.object == b["instance"]]
    assert places[0].place.position == [0.55, -0.25, 0.86] and places[0].pre_place is not None and "re-grasp" in places[0].relation
    assert places[1].place.position == sc.blocks[[x["instance"] for x in d["blocks"]].index(b["instance"])].place.position
    lifts = [a for a in actions if a.name == "lift" and a.object == b["instance"]]
    assert lifts[1].carry_height == 0.12 and lifts[1].after is grasps[1]
    # the other flipping blocks (without the record) still reorient in hand
    others = [x["instance"] for x in flipping[1:]]
    assert [a.object for a in actions if a.name == "reorient"] == [o for o in others if o in [x["instance"] for x in d["blocks"] if x["flip_sign"] != 0]]
    # a sub-chain without its re-grasp is rejected
    d2 = raw("configuration_matching-block_design")
    [x for x in d2["blocks"] if x["flip_sign"] != 0][0]["table_flip"] = {"place": b["table_flip"]["place"]}
    with pytest.raises(SidecarError):
        expand(sc.alias, Sidecar.from_dict(d2))


def test_block_design_flip_in_place_then_move_and_per_block_stances():
    """Owner 2026-09-19: (1) a ``table_flip`` record with ``in_hand_flip`` turns the block in hand BEFORE the set-down on its
    own cell — grasp → lift → reorient → place(own cell) → retreat → grasp(re-grasp, its own tip clearance) → lift →
    place(slot) → retreat; (2) a block whose record names another ``stance`` is preceded by the recorded travel from the
    stance the robot stands at, with the working arm of that stance; (3) the recorded ``pre_place`` is used by the place."""
    d = raw("configuration_matching-block_design")
    flipping = [b for b in d["blocks"] if b["flip_sign"] != 0]
    b = flipping[0]
    b["tip_clearance_m"] = 0.007
    b["table_flip"] = {
        "cell": [0.55, -0.25], "in_hand_flip": True,
        "place": {"position": [0.55, -0.25, 0.86], "orientation": [0.0, 0.7071, 0.0, 0.7071]},
        "pre_place": {"position": [0.55, -0.25, 0.96], "orientation": [0.0, 0.7071, 0.0, 0.7071]},
        "block_pose_after": {"position": [0.55, -0.25, 0.83], "orientation": [0.0, 0.0, 0.0, 1.0]},
        "regrasp": {"face_pair": "y" if b["face_pair"] != "y" else "x", "tilt_pick_deg": 30.0, "yaw_deg": 5.0, "carry_height": 0.12, "tip_clearance_m": 0.001,
                    "pick": {"position": [0.55, -0.25, 0.85], "orientation": [0.0, 0.7071, 0.0, 0.7071]}},
    }
    b["pre_place"] = {"position": [0.6, -0.2, 0.95], "orientation": [0.0, 0.7071, 0.0, 0.7071]}
    # the last block works from a second stance 0.3 m along the edge, reached by a recorded path + transition
    last = sorted(d["blocks"], key=lambda x: x["order"])[-1]
    manip = d["stances"]["manipulation"]
    st2 = dict(manip)
    st2["base_pose2d"] = [manip["base_pose2d"][0], manip["base_pose2d"][1] + 0.3, manip["base_pose2d"][2]]
    st2["arms"] = ["right"]
    d["stances"]["manipulation_2"] = st2
    last["stance"] = "manipulation_2"
    d.setdefault("paths", []).append({"from": "manipulation", "to": "manipulation_2", "mode": "local", "waypoints": [list(manip["base_pose2d"]), list(st2["base_pose2d"])],
                                      "length_m": 0.3, "posture": "untucked", "carried": None})
    d.setdefault("transitions", []).append({"from": "manipulation", "to": "manipulation_2", "stand_off": float(manip.get("stand_off", 0.35)), "order": ["drive-back", "trunk", "drive-in"]})
    sc = Sidecar.from_dict(d)
    actions = expand(sc.alias, sc)
    first = next(i for i, a in enumerate(actions) if a.name == "grasp" and a.objects[0] == b["instance"])
    names = []
    for a in actions[first:]:
        if a.name in ("brace", "push") or (a.name == "grasp" and a.objects[0] != b["instance"]):
            break
        names.append(a.name)
    assert names == ["grasp", "lift", "reorient", "place", "retreat", "grasp", "lift", "place", "retreat"]
    grasps = [a for a in actions if a.name == "grasp" and a.object == b["instance"]]
    assert grasps[0].block.extra["tip_clearance_m"] == 0.007 and grasps[1].block.extra["tip_clearance_m"] == 0.001
    assert grasps[1].block.flip_sign == 0 and grasps[1].block.tilt_pick_deg == 30.0
    reorient = [a for a in actions if a.name == "reorient" and a.object == b["instance"]][0]
    assert reorient.mode == "flip" and reorient.block is grasps[0].block
    places = [a for a in actions if a.name == "place" and a.object == b["instance"]]
    assert places[0].place.position == [0.55, -0.25, 0.86] and "own cell" in places[0].relation
    assert places[1].pre_place is not None and places[1].pre_place.position == [0.6, -0.2, 0.95]
    # the travel to the second stance sits right before the last block's grasp
    i_last = next(i for i, a in enumerate(actions) if a.name == "grasp" and a.objects[0] == last["instance"])
    travel = [a.name for a in actions[:i_last] if a.name in ("navigate", "transition")]
    assert travel[-2:] == ["navigate", "transition"] and actions[i_last - 1].name == "transition" and actions[i_last - 1].transition.to == "manipulation_2"
    assert actions[i_last].arm == "right"


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
    # the long drives of this family are the shared face_travel schedule, also for records without a ``style``
    assert nav_s.style == BP.FACE_TRAVEL and nav_s.targets(std, None)["style"] == BP.FACE_TRAVEL
    assert nav_o.targets(obs, None)["style"] == BP.FACE_TRAVEL
    # every other family's program pins no style: the record decides (owner 2026-09-26: the families that opt in with
    # generation.base_path_style record their uncarried drives as ``face_travel`` - IORS large init -> B, the large-articulation
    # and wall-alignment approaches - and every carried leg as recorded); a record without a style is driven as recorded
    iors = load("inter_object_relation_satisfaction-ontop_large_scale")
    navs = [a for a in expand(iors.alias, iors) if a.name == "navigate"]
    assert navs and all(a.style is None for a in navs)
    assert all(a.targets(iors, None)["style"] == (BP.FACE_TRAVEL if (a.path.extra or {}).get("style") == BP.FACE_TRAVEL else BP.RECORDED) for a in navs)
    navs[0].path.extra["style"] = BP.FACE_TRAVEL  # an init -> B record written by a face_travel family
    assert navs[0].targets(iors, None)["style"] == BP.FACE_TRAVEL
    carried = [a for a in navs if (a.path.extra or {}).get("carried") or getattr(a.path, "carried", None)]
    for a in carried:  # a carried leg recorded as "recorded" stays recorded
        a.path.extra["style"] = BP.RECORDED
        assert a.targets(iors, None)["style"] == BP.RECORDED
    with pytest.raises(ValueError, match="style"):
        A.Navigate(path=nav_s.path, style="sideways")
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


def test_disembedding_grasp_may_touch_the_tray_objects():
    """Owner 2026-10-12 (A2b): ``grasps[].may_touch`` names the distractors / parts the grasp may touch and the tray they must stay on;
    the program's Grasp carries them as its ``touches`` (QA whitelist), the record round-trips, and a sidecar without it is unchanged."""
    d = raw("disembedding")
    d["grasps"][0]["may_touch"] = {"objects": ["box_of_cereal.n.01_1", "box_of_cereal.n.01_2"], "stay_on": "tray.n.01_1"}
    sc = Sidecar.from_dict(d)
    assert sc.problems() == []
    assert sc.grasps[0].may_touch == (("box_of_cereal.n.01_1", "box_of_cereal.n.01_2"), "tray.n.01_1")
    assert sc.grasps[0].to_dict()["may_touch"] == d["grasps"][0]["may_touch"]
    grasp = [a for a in expand(sc.alias, sc) if a.name == "grasp"][0]
    assert grasp.touches == ("box_of_cereal.n.01_1", "box_of_cereal.n.01_2") and grasp.objects == ("bottle_of_beer.n.01_1",)
    assert sc.grasps[0].may_touch_pre_grasp == ()
    plain = load("disembedding")
    assert plain.grasps[0].may_touch == ((), None) and [a for a in expand(plain.alias, plain) if a.name == "grasp"][0].touches == ()
    # the puzzle's fixed-layout parts: the pre-grasp may overlap them too
    d["grasps"][0]["may_touch"]["pre_grasp"] = ["box_of_cereal.n.01_2"]
    sc = Sidecar.from_dict(d)
    assert sc.problems() == [] and sc.grasps[0].may_touch_pre_grasp == ("box_of_cereal.n.01_2",)
    assert sc.grasps[0].to_dict()["may_touch"]["pre_grasp"] == ["box_of_cereal.n.01_2"]


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


@pytest.mark.parametrize("subtask", FIXTURE_SUBTASKS)
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


def test_brace_annotation_says_open_for_a_wide_stop():
    """Owner 2026-10-14 ~04:25 ("fix it"): an open-finger brace (the record's ``fingers_opening_m``: touching's wide stop
    ``brace_wide_h-``) was annotated "with the closed left hand"; a closed brace keeps its text."""
    import copy

    sc = load("inter_object_relation_satisfaction-touching")
    brace = [a for a in expand(sc.alias, sc) if a.name == "brace"][0]
    assert not A.brace_is_open(brace.contact)
    fine, numbers = brace.annotate(sc)
    assert fine.endswith("with the closed left hand") and numbers.startswith(fine + " at x=")
    wide = copy.deepcopy(brace)
    wide.contact.extra["fingers_opening_m"] = 0.0315
    assert A.brace_is_open(wide.contact)
    fine_w, numbers_w = wide.annotate(sc)
    assert fine_w == fine.replace("with the closed left hand", "with the open left hand")
    assert numbers_w == numbers.replace("with the closed left hand", "with the open left hand")
    for v in (None, 0.0, "x"):
        wide.contact.extra["fingers_opening_m"] = v
        assert not A.brace_is_open(wide.contact) and wide.annotate(sc) == (fine, numbers)


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
        # follow_path / follow_segments: hybrid base-follow only; arms_q: only when the record carries it; second: a two-hand base push only
        assert key in ("base_poses", "carried", "follow_path", "follow_segments", "arms_q", "second"), where
        return
    if key == "second":
        assert set(value) == {"arm", "pre", "approach", "contact"} and value["arm"] in ("left", "right"), where
        for k in ("pre", "approach", "contact"):
            _check_value(k, value[k], f"{where}.{k}")
        return
    if key == "arms_q":
        # navigate / transition: the arm joints the record was swept with (owner 2026-09-24), arm -> 7 values
        assert isinstance(value, dict) and value and all(a in ("left", "right") and len(v) == 7 for a, v in value.items()), where
        return
    if key == "follow_path":
        assert isinstance(value, list) and len(value) >= 2 and all(len(p["base_pose2d"]) == 3 and len(p["arm_q"]) == 7 for p in value), where
        return
    if isinstance(value, Pose):
        assert len(value.position) == 3 and len(value.orientation) == 4, where
    elif isinstance(value, list) and value and isinstance(value[0], Pose):
        for p in value:
            assert len(p.position) == 3 and len(p.orientation) == 4, where
    elif key in ("waypoints", "base_poses", "base_path"):
        assert value and all(len(w) == 3 for w in value), where
    elif key in ("goal", "base", "drive_back", "drive_in", "pivot", "direction"):
        assert len(value) == 3, where
    elif key == "trunk":
        assert len(value) == 2, where
    elif key in ("fractions", "q_values"):
        assert value and all(isinstance(v, float) for v in value), where
    elif key in ("posture", "arms", "mode", "carried"):
        assert isinstance(value, str) and value, where
    elif key == "style":
        assert value in BP.STYLES, where  # navigate: the recorded polyline or the shared face_travel schedule
    elif key == "axis_leg":
        assert isinstance(value, bool), where  # navigate: the heading kept, forward / backward / lateral moves only (owner 2026-10-13)
    elif key == "distance_m":
        assert value > 0, where
    elif key in ("carry", "carry_via"):
        assert isinstance(value, list), where  # empty without a run-time eef pose (the stub state) / without a recorded waypoint
    else:
        raise AssertionError(f"unexpected target key {key!r} at {where}")


@needs_conventions
@pytest.mark.parametrize("subtask", FIXTURE_SUBTASKS)
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
# targets(): the F5 call shapes (signatures, units, fields, waypoint counts) the actions resolve through
# ----------------------------------------------------------------------------------------------------------------


def _close(a, b, tol=1e-5):
    return float((C.as_tensor(a) - C.as_tensor(b)).abs().max()) < tol


def _same_quat(a, b, tol=1e-5):
    a, b = C.as_tensor(a), C.as_tensor(b)
    return min(float((a - b).abs().max()), float((a + b).abs().max())) < tol


def _same_pose(pose: Pose, other, tol=1e-5):
    p, q = pose.tensors()
    return _close(p, other[0], tol) and _same_quat(q, other[1], tol)


def _pose_dict(p, q):
    return {"position": [float(v) for v in p], "orientation": [float(v) for v in q]}


@needs_conventions
def test_transition_targets_are_the_transition_poses_fields():
    sc = load("disembedding")
    tr = [a for a in expand(sc.alias, sc) if a.name == "transition"][0]
    tp = C.transition_poses(tr.stance_from.to_dict(), tr.stance_to.to_dict(), tr.transition.stand_off)
    assert isinstance(tp, C.TransitionPoses)
    t = tr.targets(sc, StubState())
    assert t["drive_back"] == pytest.approx(list(tp.drive_back))
    assert t["drive_in"] == pytest.approx(list(tp.drive_in))
    assert t["drive_in"] == pytest.approx(list(tr.stance_to.base_pose2d))
    assert t["trunk"] == pytest.approx(tuple(tp.trunk))
    assert t["trunk"] == pytest.approx(tr.stance_to.trunk)
    assert isinstance(t["trunk"][0], float) and isinstance(t["trunk"][1], float)
    # drive-back = the arriving stance backed off by s_T - stand_off along its own -x, yaw kept (spec §3.8)
    back = tr.transition.stand_off - tr.stance_to.stand_off
    x, y, yaw = tr.stance_to.base_pose2d
    assert t["drive_back"] == pytest.approx([x - back * math.cos(yaw), y - back * math.sin(yaw), yaw], abs=1e-6)
    assert tp.start == pytest.approx(tuple(tr.stance_from.base_pose2d))
    assert tp.trunk_from == pytest.approx(tr.stance_from.trunk)


@needs_conventions
def test_library_grasp_is_derived_from_the_settled_pose_without_the_recorded_yaw():
    sc = load("inter_object_relation_satisfaction-ontop_small_scale")
    grasp = [a for a in expand(sc.alias, sc) if a.name == "grasp"][0]
    g = grasp.grasp
    assert grasp.kind == "library" and abs(g.yaw) > 0.1
    # the settled object stands at the yaw the sampler drew (spec §4.6: R_z(yaw) on the library's rest orientation)
    obj = (th.tensor([7.0, 3.1, 0.85]), C.qmul(C.quat_from_yaw(g.yaw), C.rest_quat(g.grasp_key)))

    class Settled:
        def object_pose(self, name):
            return obj

    got = grasp.grasp_pose(sc, Settled())
    # == conventions.grasp_pose(obj_pose, grasp_key, grasp_name) with yaw omitted == obj_pose ∘ library entry
    assert _same_pose(got, C.grasp_pose(obj, g.grasp_key, g.grasp_name))
    assert _same_pose(got, C.compose(obj, C.grasp_entry(g.grasp_key, g.grasp_name).pose))
    # a float yaw makes F5 *replace* the orientation by the nominal R_z(yaw) ∘ rest_quat: the recorded yaw reproduces
    # the (un-drifted) settled object here, yaw 0 does not - so neither is what the oracle passes
    assert _same_pose(got, C.grasp_pose(obj, g.grasp_key, g.grasp_name, yaw=g.yaw))
    assert not _same_pose(got, C.grasp_pose(obj, g.grasp_key, g.grasp_name, yaw=0.0))
    # a drifted settled pose moves the grasp with it (the oracle tracks the object in front of the robot)
    drifted = (obj[0] + th.tensor([0.01, -0.02, 0.0]), C.qmul(C.quat_from_yaw(0.05), obj[1]))

    class Drifted:
        def object_pose(self, name):
            return drifted

    assert _same_pose(grasp.grasp_pose(sc, Drifted()), C.compose(drifted, C.grasp_entry(g.grasp_key, g.grasp_name).pose))
    t = grasp.targets(sc, Settled())
    assert _same_pose(t["pre_grasp"], C.pre_pose(got.tensors(), A.PRE_DIST)) and len(t["approach"]) == A.N_WAYPOINTS


@needs_conventions
def test_cube_grasp_uses_the_wrist_yaw_in_radians_and_the_pick_tilt_in_degrees():
    sc = load("configuration_matching-block_design")
    state = StubState()
    grasps = [a for a in expand(sc.alias, sc) if a.name == "grasp"]
    assert grasps and all(a.kind == "cube" for a in grasps)
    for grasp in grasps:
        b = grasp.block
        centre = A.block_centre_pose(state, b).tensors()  # root pose ∘ (centre_offset, I)
        cube = C.cube_grasp(centre, b.face_pair, b.tilt_pick_deg, math.radians(b.yaw_deg))
        assert _same_pose(grasp.grasp_pose(sc, state), cube.eef)
        assert cube.tilt_deg == b.tilt_pick_deg and cube.face_pair == b.face_pair and cube.downward
        # F5 snaps the wrist yaw to the nearer admissible value (eef +x ⊥ the pinch axis)
        assert abs(C.wrap_angle(cube.yaw - math.radians(b.yaw_deg))) <= math.pi / 2 + 1e-6
        face_axis = {"x": [1.0, 0.0, 0.0], "y": [0.0, 1.0, 0.0], "z": [0.0, 0.0, 1.0]}[b.face_pair]
        assert _close(C.closing_axis(cube.eef[1]).abs(), C.quat_rotate(state.object_pose(b.instance)[1], th.tensor(face_axis)).abs())  # pads close across the face pair
        _, numbers = grasp.annotate(sc)
        assert f"wrist yaw {b.yaw_deg:.0f}°" in numbers and f"pick tilt {b.tilt_pick_deg:.0f}°" in numbers


@needs_conventions
def test_block_centre_offset_moves_the_cube_grasp_and_the_flip_pivot():
    """``blocks[].centre_offset`` (the cube centre in the root frame, written by the block-design sampler): the cube
    grasp is built about ``root ∘ (centre_offset, I)`` and the in-hand flip pivots there, not at the root link."""
    sc = load("configuration_matching-block_design")
    state = StubState()
    actions = expand(sc.alias, sc)
    grasp = [a for a in actions if a.name == "grasp"][0]
    reorient = [a for a in actions if a.name == "reorient" and a.block is not None][0]
    b = grasp.block
    root = state.object_pose(b.instance)
    off = [0.01, -0.02, 0.03]
    plain = grasp.grasp_pose(sc, state)
    plain_pivot = reorient.targets(sc, state)["pivot"]
    saved = dict(b.extra), dict(reorient.block.extra)
    off0 = th.tensor(b.extra.get("centre_offset", [0.0, 0.0, 0.0]))
    b.extra["centre_offset"] = off
    reorient.block.extra["centre_offset"] = list(off)
    centre = A.block_centre_pose(state, b)
    assert _close(th.tensor(centre.position), root[0] + C.quat_rotate(root[1], th.tensor(off))) and _close(th.tensor(centre.orientation), root[1])
    shift = C.quat_rotate(root[1], th.tensor(off) - off0)
    shifted = grasp.grasp_pose(sc, state)
    assert _close(th.tensor(shifted.position) - th.tensor(plain.position), shift, tol=1e-5)
    assert _close(th.tensor(shifted.orientation), th.tensor(plain.orientation))
    pivot = reorient.targets(sc, state)["pivot"]
    assert _close(th.tensor(pivot) - th.tensor(plain_pivot), shift, tol=1e-5)
    b.extra.clear(), b.extra.update(saved[0])
    reorient.block.extra.clear(), reorient.block.extra.update(saved[1])
    assert _same_pose(grasp.grasp_pose(sc, state), plain.tensors())
    # every block record carries yaw_align_deg and centre_offset now (the sampler writes them on every record)
    assert all("yaw_align_deg" in blk.extra and "centre_offset" in blk.extra for blk in sc.blocks)


@needs_conventions
def test_flip_reorient_calls_flip_waypoints_with_the_f5_signature():
    state = StubState()
    d = raw("configuration_matching-block_design")
    flipper = next(b for b in d["blocks"] if b["flip_sign"] != 0)
    flipper["yaw_align_deg"] = -31.0  # the table-alignment yaw ψ, written next to the wrist yaw
    sc = Sidecar.from_dict(d)
    reor = next(a for a in expand(sc.alias, sc) if a.name == "reorient" and a.object == flipper["instance"])
    assert reor.mode == "flip" and reor.after.name == "lift" and reor.waypoints is None
    t = reor.targets(sc, state)
    start = reor.after.end_pose(sc, state).tensors()
    pivot = th.tensor(t["pivot"])
    # flip_waypoints(eef, pivot, yaw=ψ rad, flip_axis, flip_sign, n_yaw=2, n_flip=3): the yaw stage then the flip stage
    want = C.flip_waypoints(start, pivot, yaw=math.radians(-31.0), flip_axis=th.tensor(flipper["flip_axis"]), flip_sign=flipper["flip_sign"],
                            n_yaw=A.N_YAW_ALIGN, n_flip=A.N_FLIP)
    assert len(t["waypoints"]) == len(t["derived"]) == len(want) == A.N_YAW_ALIGN + A.N_FLIP == 5
    assert t["waypoints"] == t["derived"] and t["end"] == t["derived"][-1]  # nothing recorded: the regeneration is tracked
    for got, exp in zip(t["derived"], want):
        assert _same_pose(got, exp)
    q_yaw = C.qmul(C.quat_from_yaw(math.radians(-31.0)), start[1])  # +ψ about world z through the pivot ...
    assert _same_quat(t["derived"][A.N_YAW_ALIGN - 1].tensors()[1], q_yaw)
    q_end = C.qmul(C.quat_from_axis_angle(th.tensor(flipper["flip_axis"]), flipper["flip_sign"] * math.pi / 2), q_yaw)  # ... then ±90° about the flip axis
    assert _same_quat(t["end"].tensors()[1], q_end)
    r0 = float(th.linalg.norm(start[0] - pivot))
    assert all(abs(float(th.linalg.norm(w.tensors()[0] - pivot)) - r0) < 1e-5 for w in t["derived"])  # the block turns in place
    fine, numbers = reor.annotate(sc)
    assert "yaw it -31°" in numbers and "flip it 90°" in numbers and "yawing and flipping" in fine
    # ψ = 0: the yaw stage is skipped -> the 3 flip poses only
    flipper["yaw_align_deg"] = 0.0
    sc0 = Sidecar.from_dict(d)
    reor0 = next(a for a in expand(sc0.alias, sc0) if a.name == "reorient" and a.object == flipper["instance"])
    t0 = reor0.targets(sc0, state)
    assert len(t0["derived"]) == A.N_FLIP == 3 and "yaw it" not in reor0.annotate(sc0)[1]
    want0 = C.flip_waypoints(start, pivot, yaw=0.0, flip_axis=th.tensor(flipper["flip_axis"]), flip_sign=flipper["flip_sign"],
                             n_yaw=A.N_YAW_ALIGN, n_flip=A.N_FLIP)
    assert len(want0) == 3
    for got, exp in zip(t0["derived"], want0):
        assert _same_pose(got, exp)
    assert _same_quat(t0["end"].tensors()[1], C.qmul(C.quat_from_axis_angle(th.tensor(flipper["flip_axis"]), flipper["flip_sign"] * math.pi / 2), start[1]))
    # a yaw-align without a flip: orbit_waypoints(eef, pivot, quat_from_yaw(ψ), n=2)
    anchor = d["blocks"][0]
    assert anchor["flip_sign"] == 0 and anchor.get("flip_axis") is None
    anchor["yaw_align_deg"] = 12.0
    sc1 = Sidecar.from_dict(d)
    reor1 = next(a for a in expand(sc1.alias, sc1) if a.name == "reorient" and a.object == anchor["instance"])
    t1 = reor1.targets(sc1, state)
    s1 = reor1.after.end_pose(sc1, state).tensors()
    assert len(t1["derived"]) == A.N_YAW_ALIGN == 2
    for got, exp in zip(t1["derived"], C.orbit_waypoints(s1, th.tensor(t1["pivot"]), C.quat_from_yaw(math.radians(12.0)), n=2)):
        assert _same_pose(got, exp)
    assert _same_quat(t1["end"].tensors()[1], C.qmul(C.quat_from_yaw(math.radians(12.0)), s1[1]))
    # recorded block waypoints (``blocks[].waypoints``) are tracked verbatim; the regeneration stays alongside as ``derived``
    recorded = [w.to_dict() for w in reversed(t["derived"])]
    flipper["yaw_align_deg"] = -31.0
    flipper["waypoints"] = recorded
    sc2 = Sidecar.from_dict(d)
    reor2 = next(a for a in expand(sc2.alias, sc2) if a.name == "reorient" and a.object == flipper["instance"])
    t2 = reor2.targets(sc2, state)
    assert [w.to_dict() for w in t2["waypoints"]] == recorded and t2["end"].to_dict() == recorded[-1]
    assert reor2.end_pose(sc2, state).to_dict() == recorded[-1]
    assert [w.to_dict() for w in t2["derived"]] == [w.to_dict() for w in t["derived"]]


@needs_conventions
@pytest.mark.parametrize("subtask,mode", [("object_scaling", "line"), ("mental_rotation", "orbit")])
def test_held_object_reorient_tracks_the_recorded_three_intermediate_waypoints(subtask, mode):
    sc = load(subtask)
    state = StubState()
    (reor,) = expand(sc.alias, sc)
    g = sc.grasps[0]
    assert reor.mode == mode and reor.n == A.N_INHAND == 4 and reor.start_pose == g.grasp
    t = reor.targets(sc, state)
    # the recorded key poses (3 intermediate + the goal) are what the oracle tracks; the end is the recorded goal
    assert t["waypoints"] == g.waypoints and len(t["waypoints"]) == 4
    assert t["end"] == g.waypoints[-1] and reor.end_pose(sc, state) == g.waypoints[-1]
    assert len(t["derived"]) == 4  # regenerated with the same count: line_waypoints(p0, p1, 4) / orbit_waypoints(..., n=4)
    assert t["pivot"] == pytest.approx(state.object_pose(g.object)[0].tolist())  # the object root at the start
    # a sidecar whose key poses come from the sampler's own construction (held_object.line_key_poses /
    # orbit_key_poses with 3 intermediate poses): the regeneration reproduces them pose by pose
    start, end = g.grasp.tensors(), g.waypoints[-1].tensors()
    pivot = state.object_pose(g.object)[0]
    if mode == "line":
        sampler = C.line_waypoints(start, end, 3 + 1)
    else:
        sampler = C.orbit_waypoints(start, pivot, C.qmul(end[1], C.qinv(start[1])), 3 + 1, frame="world")
    d = raw(subtask)
    d["grasps"][0]["waypoints"] = [_pose_dict(p, q) for p, q in sampler]
    sc2 = Sidecar.from_dict(d)
    (reor2,) = expand(sc2.alias, sc2)
    t2 = reor2.targets(sc2, state)
    assert len(t2["waypoints"]) == len(t2["derived"]) == 4
    for got, exp in zip(t2["derived"], t2["waypoints"]):
        assert _same_pose(got, exp.tensors())
    # the intermediate poses sit at the fractions 1/4, 2/4, 3/4 of the motion, the last one is the goal
    total = C.quat_angle(start[1], end[1])
    for k, w in enumerate(t2["waypoints"], start=1):
        p, q = w.tensors()
        if mode == "line":
            assert _close(p, start[0] + (end[0] - start[0]) * k / 4) and _same_quat(q, start[1])
        else:
            assert C.quat_angle(start[1], q) == pytest.approx(total * k / 4, abs=1e-4)
            assert float(th.linalg.norm(p - pivot)) == pytest.approx(float(th.linalg.norm(start[0] - pivot)), abs=1e-5)
    if mode == "line":
        assert _same_pose(t2["waypoints"][-1], end)
    else:  # the goal orientation is the recorded one; the eef turned rigidly with the object about its root
        assert _same_quat(t2["waypoints"][-1].tensors()[1], end[1])
        assert _close(t2["waypoints"][-1].tensors()[0], pivot + C.quat_rotate(C.qmul(end[1], C.qinv(start[1])), start[0] - pivot))
    # regeneration without recorded poses is refused for the held-object modes (the sidecar must carry them)
    with pytest.raises(ValueError, match="recorded waypoints"):
        A.Reorient(mode=mode, arm=g.arm, object=g.object, start_pose=g.grasp)


@needs_conventions
def test_attachment_reorient_is_the_eef_line_with_slerped_orientation():
    sc = load("configuration_matching-attachment")
    state = StubState()
    reor = next(a for a in expand(sc.alias, sc) if a.name == "reorient")
    assert reor.mode == "slerp" and reor.after.name == "lift"
    t = reor.targets(sc, state)
    g = sc.grasps[0]
    assert t["waypoints"] == g.waypoints and len(t["derived"]) == A.N_INHAND
    start = reor.after.end_pose(sc, state).tensors()
    for got, exp in zip(t["derived"], C.line_waypoints(start, g.waypoints[-1].tensors(), A.N_INHAND)):
        assert _same_pose(got, exp)


@needs_conventions
def test_handle_records_reproduce_handle_waypoints_of_the_sampler():
    # the sampler's inputs - the handle annotation (object frame) and the joint info - and its F5 call ...
    handle = {"contact": [0.35, -0.02, 0.95], "pinch_axis": [0.0, 0.0, 1.0], "approach_dir": [-1.0, 0.0, 0.0]}
    jinfo = {"type": "RevoluteJoint", "axis_dir_obj": [0.0, 0.0, 1.0], "hinge_point_obj": [0.4, -0.45, 0.0], "lower": 0.0}
    d = raw("configuration_matching-articulation_open_large_scale")
    art = d["articulation"]
    q_init, q_goal = art["q_init"], art["q_goal"]
    fractions = [0.0, 0.25, 0.5, 0.75, 1.0]
    q_values = [q_init + k * (q_goal - q_init) for k in fractions]
    obj_pose = (th.tensor([4.9, -1.45, 0.0]), C.quat_from_yaw(math.pi))
    wps = C.handle_waypoints(handle, jinfo, q_values, obj_pose)
    assert len(wps) == 5 and all(isinstance(w, C.HandleWaypoint) for w in wps)
    # ... whose *outputs* the sidecar carries: contact / approach / pinch per fraction key (HandleWaypoint.to_sidecar)
    art["waypoints"] = {str(k): w.to_sidecar() for k, w in zip(fractions, wps)}
    assert set(art["waypoints"]) == {"0.0", "0.25", "0.5", "0.75", "1.0"}
    assert set(art["waypoints"]["0.5"]) == {"contact", "approach", "pinch"}
    art["handle_reach"] = {str(k): True for k in fractions[1:]}
    art["passive_from"] = None
    stance = d["stances"]["manipulation"]
    base = C.pose2d_to_pose(*stance["base_pose2d"], z=C.NOMINAL_BASE_Z)
    T = C.eef_base_transform(wps[0].eef, base)
    art["base_follow"] = {str(k): [float(v) for v in b] for k, b in zip(fractions[1:], C.base_follow_poses([w.eef for w in wps[1:]], T))}
    sc = Sidecar.from_dict(d)
    assert sc.problems() == []
    actions = expand(sc.alias, sc)
    grasp = next(a for a in actions if a.name == "grasp")
    follow = next(a for a in actions if a.name == "base_follow")
    state = StubState()
    # Grasp[handle]: handle_eef of h(q_init) == HandleWaypoint.eef (origin = contact - 0.012·a, a = approach, c = pinch)
    got = grasp.grasp_pose(sc, state)
    assert _same_pose(got, wps[0].eef)
    p, q = got.tensors()
    assert _close(p, th.tensor(art["waypoints"]["0.0"]["contact"]) - C.HANDLE_EEF_OFFSET * C.approach_axis(q))
    assert _close(C.approach_axis(q), wps[0].approach) and _close(C.closing_axis(q), wps[0].pinch)
    gt = grasp.targets(sc, state)
    assert _same_pose(gt["pre_grasp"], wps[0].pre_grasp) and _same_pose(gt["approach"][-1], wps[0].eef)
    # BaseFollow: every followed h(q_k) as handle_eef of its record; b_k = h(q_k) ∘ T_eef→base(q*) as base_follow_poses
    t = follow.targets(sc, state)
    assert t["mode"] == "base" and t["fractions"] == fractions[1:]
    assert t["q_values"] == pytest.approx(q_values[1:])
    for got_k, w in zip(t["handle_poses"], wps[1:]):
        assert _same_pose(got_k, w.eef)
    assert len(t["base_poses"]) == 4
    for got_k, exp in zip(t["base_poses"], art["base_follow"].values()):
        assert got_k == pytest.approx(exp, abs=1e-4)
    assert follow.end_pose(sc, state) == t["handle_poses"][-1]
    # the handle must not be re-derived from the annotation: the records already moved with the joint
    assert not _close(th.tensor(art["waypoints"]["1.0"]["contact"]), th.tensor(art["waypoints"]["0.0"]["contact"]), 1e-3)


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


def test_under_large_push_under_chain():
    """Owner decision 2026-09-16 item 7: ``grasps[0].place_kind == "drop"`` + ``contacts.push`` of kind ``push_under``
    turn the under-large chain into … → place(drop on the floor) → retreat → push (along the floor, partner forbidden)
    → retreat; without the push record the sidecar is invalid."""
    with open(os.path.join(VARIANTS, "inter_object_relation_satisfaction-under_large_scale__push_under.json")) as f:
        d = json.load(f)
    sc = Sidecar.from_dict(d)
    assert sc.problems() == [] and sc.contacts.push.kind == "push_under"
    actions = expand(sc.alias, sc)
    assert [a.name for a in actions] == ["transition", "grasp", "lift", "navigate", "place", "retreat", "push", "retreat"]
    place, ret1, push, ret2 = actions[4:]
    assert place.kind == "drop" and place.on == "floor.n.01_1" and place.relation == "under" and place.avoid == ("breakfast_table.n.01_1",)
    assert ret1.after is place and push.kind == "push_under" and push.touches == ("floor.n.01_1",) and push.avoid == ("breakfast_table.n.01_1",)
    assert push.arms == ("right",) and push.objects == ("briefcase.n.01_1",) and ret2.after is push
    assert "along the floor under the dining table" in push.annotate(sc)[0] and "let it drop" in place.annotate(sc)[0]
    t = push.targets(sc, StubState()) if HAVE_CONVENTIONS else None
    if t is not None:
        assert len(t["waypoints"]) == 3 and t["distance_m"] == 0.25
    # the plain under-large fixture (no drop) keeps the support place
    plain = load("inter_object_relation_satisfaction-under_large_scale")
    assert [a.name for a in expand(plain.alias, plain)] == EXPECTED_SEQUENCE["inter_object_relation_satisfaction-under_large_scale"]
    # a drop without the push record, or with a plain push, is rejected
    d2 = copy.deepcopy(d)
    d2.pop("contacts")
    assert any("push_under" in p for p in Sidecar.from_dict(d2).problems())
    d3 = copy.deepcopy(d)
    d3["contacts"]["push"].pop("kind")
    assert any("push_under" in p for p in Sidecar.from_dict(d3).problems())
    d4 = copy.deepcopy(d)
    d4["grasps"][0].pop("place_kind")
    sc4 = Sidecar.from_dict(d4)
    assert sc4.problems() == [] and [a.name for a in expand(sc4.alias, sc4)] == ["transition", "grasp", "lift", "navigate", "place", "retreat"]


def test_block_design_braces_are_optional():
    """Owner decision 2026-09-16 11:08: row / column braces (with their paired pushes) are recorded only when a joint
    two-arm placement exists; a brace-less block record is valid and its program is pick → [reorient] → place →
    retreat with no brace / push tail; a half-recorded axis is still rejected."""
    with open(os.path.join(VARIANTS, "configuration_matching-block_design__brace_less.json")) as f:
        d = json.load(f)
    assert all(k not in b for b in d["blocks"] for k in ("brace_row", "push_row", "brace_col", "push_col"))
    sc = Sidecar.from_dict(d)
    assert sc.problems() == [] and sc.validate() is sc
    assert "contacts.brace" not in REQUIRED_BLOCKS["configuration_matching-block_design"]
    names = [a.name for a in expand(sc.alias, sc)]
    assert "brace" not in names and "push" not in names
    per_block = ["grasp", "lift", "place", "retreat"]
    assert names == _TABLE_IN + per_block + ["grasp", "lift", "reorient", "place", "retreat"] + per_block + ["grasp", "lift", "reorient", "place", "retreat"]
    # the fully recorded fixture keeps its tails; a brace without its push (or the reverse) on one axis is rejected
    assert [a.name for a in expand(load("configuration_matching-block_design").alias, load("configuration_matching-block_design"))] == EXPECTED_SEQUENCE["configuration_matching-block_design"]
    d2 = raw("configuration_matching-block_design")
    d2["blocks"][1].pop("brace_row")
    assert any("must be recorded together" in p for p in Sidecar.from_dict(d2).problems())


def test_axial_is_push_only_without_a_brace_record():
    """Owner decision 2026-09-16 08:10: ``contacts.brace`` is optional for axial; without it the program is
    transition → push → retreat (the oracle's push closure is the stop), with it the braced chain as before."""
    assert "contacts.brace" not in REQUIRED_BLOCKS["alignment-axial"] and "contacts.push" in REQUIRED_BLOCKS["alignment-axial"]
    with open(os.path.join(VARIANTS, "alignment-axial__push_only.json")) as f:
        d = json.load(f)
    assert "brace" not in d["contacts"]
    sc = Sidecar.from_dict(d)
    assert sc.problems() == [] and sc.validate() is sc and not sc.has_block("contacts.brace")
    actions = expand(sc.alias, sc)
    assert [a.name for a in actions] == ["transition", "push", "retreat"]
    assert [(a.name, a.arms, a.objects) for a in actions[1:]] == [
        ("push", ("right",), ("paperback_book.n.01_1",)),
        ("retreat", ("right",), ("paperback_book.n.01_1",)),
    ]
    assert actions[2].after is actions[1] and actions[2].release is False
    # the same sidecar with the brace record keeps the braced chain
    braced = load("alignment-axial")
    assert [a.name for a in expand(braced.alias, braced)] == _TABLE_IN + _BRACE_PUSH
    # a left-arm push-only record is still rejected (spec §6.10: the right gripper pushes)
    d2 = copy.deepcopy(d)
    d2["contacts"]["push"]["arm"] = "left"
    problems = Sidecar.from_dict(d2).problems()
    assert any("right gripper pushes" in p for p in problems), problems


def test_touching_is_push_only_without_a_brace_record():
    """Owner decision 2026-09-16 17:10 (the axial rule for touching; core ``6de8803`` records ``contacts.brace`` only
    when the joint placement exists): without it the program is the nextto chain (transition → grasp → lift → place at
    the nextto position → retreat) followed by push → retreat, the push closure being the stop; with the brace record
    the braced chain stays; the 17:54 / 18:30 / 18:53 napkins runs failed here with ``contacts.brace: required``."""
    st = "inter_object_relation_satisfaction-touching"
    assert "contacts.brace" not in REQUIRED_BLOCKS[st] and {"grasps", "contacts.push"} <= set(REQUIRED_BLOCKS[st])
    with open(os.path.join(VARIANTS, f"{st}__push_only.json")) as f:
        d = json.load(f)
    assert "brace" not in d["contacts"] and d["summary"]["brace"]["present"] is False
    sc = Sidecar.from_dict(d)
    assert sc.problems() == [] and sc.validate() is sc and not sc.has_block("contacts.brace")
    actions = expand(sc.alias, sc)
    assert [a.name for a in actions] == _TABLE_IN + _PICK + _PLACE + ["push", "retreat"]
    assert [(a.name, a.arms, a.objects) for a in actions[-2:]] == [
        ("push", ("right",), ("can_of_soda.n.01_1",)),
        ("retreat", ("right",), ("can_of_soda.n.01_1",)),
    ]
    assert actions[-1].after is actions[-2] and actions[-1].release is False
    assert all("brace" not in a.name for a in actions)
    # the annotation still reads without the partner's brace entry being used
    for a in actions:
        fine, numbers = a.annotate(sc)
        assert fine and numbers
    # the braced fixture keeps the braced chain
    braced = load(st)
    assert [a.name for a in expand(braced.alias, braced)] == EXPECTED_SEQUENCE[st]
    # a left-arm push-only record is still rejected (spec §6.11: the right gripper pushes the target)
    d2 = copy.deepcopy(d)
    d2["contacts"]["push"]["arm"] = "left"
    problems = Sidecar.from_dict(d2).problems()
    assert any("right gripper pushes" in p for p in problems), problems
    # a push-only record whose push acts on the partner instead of the placed target is rejected
    d3 = copy.deepcopy(d)
    d3["contacts"]["push"]["object"] = "mug.n.04_1"
    problems = Sidecar.from_dict(d3).problems()
    assert any("placed target" in p for p in problems), problems


@pytest.mark.parametrize("st", ["configuration_matching-articulation_open_small_scale", "configuration_matching-articulation_close_small_scale"])
def test_small_articulation_is_handle_only_without_a_brace_record(st):
    """Owner decision D-008 = A (2026-09-16 22:02): the sampler records ``contacts.brace`` only when a joint two-arm cell
    exists (tabletop ``generation.brace_optional``: ``contacts`` absent, ``summary.brace: none``, manipulation arms =
    [handle_arm], ``articulation.brace_kind: null``, ``articulation.handle_arm``); the program is then transition →
    grasp(handle) → articulate → release → retreat; the braced fixture keeps the braced chain."""
    direction = "open" if "open" in st else "close"
    assert "contacts.brace" not in REQUIRED_BLOCKS[st] and "articulation" in REQUIRED_BLOCKS[st]
    with open(os.path.join(VARIANTS, f"{st}__handle_only.json")) as f:
        d = json.load(f)
    assert "contacts" not in d and d["summary"]["brace"] == "none" and d["stances"]["manipulation"]["arms"] == ["right"] and d["articulation"]["brace_kind"] is None
    sc = Sidecar.from_dict(d)
    assert sc.problems() == [] and sc.validate() is sc and not sc.has_block("contacts.brace")
    actions = expand(sc.alias, sc)
    assert [a.name for a in actions] == _TABLE_IN + ["grasp", "base_follow", "retreat"]
    assert [(a.name, a.arms, a.objects) for a in actions[1:]] == [
        ("grasp", ("right",), ("toolbox.n.01_1",)),
        ("base_follow", ("right",), ("toolbox.n.01_1",)),
        ("retreat", ("right",), ("toolbox.n.01_1",)),
    ]
    assert actions[1].kind == "handle" and actions[2].direction == direction and actions[3].release is True and actions[3].after is actions[2]
    for a in actions:
        fine, numbers = a.annotate(sc)
        assert fine and numbers
    # the handle arm comes from articulation.handle_arm: a left-handed record drives the grasp with the left arm
    d2 = copy.deepcopy(d)
    d2["articulation"]["handle_arm"] = "left"
    d2["stances"]["manipulation"]["arms"] = ["left"]
    assert [a.arms for a in expand(sc.alias, Sidecar.from_dict(d2))[1:]] == [("left",)] * 3
    d3 = copy.deepcopy(d)
    d3["articulation"]["handle_arm"] = "middle"
    assert any("unknown arm 'middle'" in p for p in Sidecar.from_dict(d3).problems())
    # the braced fixture keeps brace -> grasp -> articulate -> retreat both
    braced = load(st)
    assert [a.name for a in expand(braced.alias, braced)] == EXPECTED_SEQUENCE[st]
    # a brace record on the handle arm, or on another object, is still rejected
    d4 = raw(st)
    d4["contacts"]["brace"]["arm"] = "right"
    assert any("must use 'left'" in p for p in Sidecar.from_dict(d4).problems())
    d5 = raw(st)
    d5["contacts"]["brace"]["object"] = "nothing.n.01_1"
    assert any("base link" in p or "language.objects has no entry" in p for p in Sidecar.from_dict(d5).problems())


def test_small_articulation_braced_stance_lists_both_working_arms():
    """The sampler records the braced manipulation stance with both working arms (``arms: [handle_arm, brace_arm]``) and
    the handle arm in ``articulation.handle_arm``; the program takes the handle arm from there (a two-arm stance without the
    record is resolved as the arm that is not the brace arm)."""
    st = "configuration_matching-articulation_open_small_scale"
    with open(os.path.join(VARIANTS, f"{st}__two_arm_stance.json")) as f:
        d = json.load(f)
    assert d["stances"]["manipulation"]["arms"] == ["right", "left"] and d["articulation"]["handle_arm"] == "right"
    sc = Sidecar.from_dict(d)
    assert sc.problems() == [] and [a.name for a in expand(sc.alias, sc)] == EXPECTED_SEQUENCE[st]
    assert [a.arms for a in expand(sc.alias, sc)[1:]] == [("left",), ("right",), ("right",), ("right",), ("left",)]
    d2 = copy.deepcopy(d)
    d2["articulation"].pop("handle_arm")
    assert [a.arms for a in expand(sc.alias, Sidecar.from_dict(d2))[1:]] == [("left",), ("right",), ("right",), ("right",), ("left",)]
    d3 = copy.deepcopy(d2)
    d3.pop("contacts")
    assert any("cannot tell the handle arm" in p for p in Sidecar.from_dict(d3).problems())


def test_room_door_close_pushes_the_outer_face():
    """Owner decision B (2026-09-20 20:19): a room door (``articulation.door_class`` ``room_door``) is closed by pushing its
    outer face — navigate, transition, push_close with the working arm, no grasp; the hinge is the centre of the circle
    through the recorded handle contacts and the closed leaf's frame is the waypoint at the lower limit (k = 1 for a
    close). Any other class keeps the inverse of the open (grasp, follow, retreat), as does a room door's open."""
    from ispatialgym.oracle.programs import program_articulation_large

    st = "configuration_matching-articulation_close_large_scale"
    with open(os.path.join(VARIANTS, f"{st}__room_door.json")) as f:
        d = json.load(f)
    sc = Sidecar.from_dict(d)
    assert sc.problems() == []
    actions = expand(sc.alias, sc)
    assert [a.name for a in actions] == _WALL_IN + ["push_close"]
    push = actions[-1]
    assert isinstance(push, A.PushClose) and type(push) in A.VOCABULARY
    assert (push.arm, push.arms, push.objects) == ("right", ("left", "right"), ("door.n.01_1",))
    assert push.qa_exempt == frozenset({A.QA_FAILED_GRASP}) and set(push.targets(sc, None)) == set(push.TARGET_KEYS) == {"leaf"}
    art = d["articulation"]
    for w in art["waypoints"].values():  # a vertical hinge: every recorded handle contact on one circle about it
        assert math.hypot(w["contact"][0] - push.hinge_xy[0], w["contact"][1] - push.hinge_xy[1]) == pytest.approx(push.handle_radius_m, abs=2e-3)
    closed = art["waypoints"]["1.0"]
    assert push.handle_xy_closed == pytest.approx(tuple(closed["contact"][:2])) and push.handle_z == pytest.approx(closed["contact"][2])
    assert push.approach_closed == pytest.approx(tuple(closed["approach"]))
    assert (push.q_from, push.q_lower) == (pytest.approx(art["q_init"]), pytest.approx(art["q_goal"]))
    fine, numbers = push.annotate(sc)
    assert fine == "close the door by pushing its outer face with the closed hand" and numbers.startswith(fine)
    for cls in (None, "cabinet"):
        d2 = copy.deepcopy(d)
        d2["articulation"]["door_class"] = cls
        assert [a.name for a in expand(sc.alias, Sidecar.from_dict(d2))] == _WALL_IN + ["grasp", "base_follow", "retreat"]
    assert [a.name for a in program_articulation_large(sc, direction="open")] == _WALL_IN + ["grasp", "base_follow", "retreat"]
    d3 = copy.deepcopy(d)
    d3["articulation"]["waypoints"] = {k: d3["articulation"]["waypoints"][k] for k in ("0.0", "1.0")}
    d3["articulation"]["handle_reach"] = {"1.0": True}
    with pytest.raises(SidecarError, match=">= 3 recorded handle waypoints"):
        expand(sc.alias, Sidecar.from_dict(d3))


def _cabinet_push_finish(k=0.75):
    """The close large-scale fixture's leaf as a cabinet door followed by the arm, with a push finish released at fraction ``k``
    (``articulation.follow_to`` + the sampler's ``push_finish`` record; the contact a unit pose on the leaf)."""
    with open(os.path.join(VARIANTS, "configuration_matching-articulation_close_large_scale__room_door.json")) as f:
        d = json.load(f)
    art = d["articulation"]
    for key in ("base_follow", "base_follow_path"):
        art.pop(key, None)
    q_from = art["q_init"] + k * (art["q_goal"] - art["q_init"])
    w = art["waypoints"]["0.75"]
    art.update({"door_class": "cabinet", "follow_mode": "arm", "follow_to": k,
                "push_finish": {"arm": "right", "stance": "manipulation", "base_pose2d": list(d["stances"]["manipulation"]["base_pose2d"]), "k_release": k,
                                "q_from": q_from, "q_to": art["q_goal"], "q_lower": art["q_goal"], "hinge_world": [-6.8613, 5.2119, 1.0], "swing_sign": 1.0,
                                "contact": {"position": list(w["contact"]), "orientation": [0.0, 0.0, 0.0, 1.0]}, "normal": [-float(v) for v in w["approach"]],
                                "pre_m": 0.10, "arm_q_pre": [0.0] * 7}})
    return d


def test_cabinet_push_finish_program():
    """Owner 2026-10-12 ~04:20 (close_large#03): a cabinet door whose closing the arm follows only part of the way
    (``articulation.push_finish`` + ``follow_to``) is grasped, followed up to the release, released and retreated, then pushed the rest
    of the way shut by ``push_close`` with the planned chunk (no stance search, no base drag), judged on the instance's joint tolerance;
    the push carries no eef pose the program check would solve. The room door's push-close, a plain cabinet close and an open are
    unchanged; a malformed record is a sidecar problem."""
    from ispatialgym.oracle.programs import program_articulation_large
    from ispatialgym.sampling.v2.program_check import iter_pose_targets

    st = "configuration_matching-articulation_close_large_scale"
    d = _cabinet_push_finish()
    sc = Sidecar.from_dict(d)
    assert sc.problems() == []
    actions = expand(sc.alias, sc)
    assert [a.name for a in actions] == _WALL_IN + ["grasp", "base_follow", "retreat", "push_close"]
    grasp, follow, retreat, push = actions[2:]
    art = d["articulation"]
    assert [key for _, key in follow.articulation.followed_fractions()] == ["0.25", "0.5", "0.75"] and retreat.after is follow and retreat.release is True
    assert isinstance(push, A.PushClose) and (push.arm, push.arms, push.objects) == ("right", ("left", "right"), ("door.n.01_1",))
    assert push.plan == [art["push_finish"]] and push.drag_m == 0.0 and push.q_from == pytest.approx(art["push_finish"]["q_from"])
    assert (push.q_lower, push.tol_span_rad) == (pytest.approx(art["q_goal"]), pytest.approx(abs(art["q_goal"] - art["q_init"])))
    targets = push.targets(sc, None)
    assert set(targets) == {"leaf"} and targets["leaf"]["planned_chunks"] == 1 and list(iter_pose_targets(targets)) == []
    fine, numbers = push.annotate(sc)
    assert fine == "push the door the rest of the way shut with the closed right hand on its outer face" and numbers.startswith(fine)
    assert "letting go there for the push that takes it on" in follow.annotate(sc)[1]
    # the room door, a plain cabinet close and the open: as before
    with open(os.path.join(VARIANTS, f"{st}__room_door.json")) as f:
        room = Sidecar.from_dict(json.load(f))
    assert [a.name for a in expand(room.alias, room)] == _WALL_IN + ["push_close"] and expand(room.alias, room)[-1].plan is None
    plain = copy.deepcopy(d)
    for key in ("push_finish", "follow_to"):
        plain["articulation"].pop(key)
    assert [a.name for a in expand(sc.alias, Sidecar.from_dict(plain))] == _WALL_IN + ["grasp", "base_follow", "retreat"]
    assert [a.name for a in program_articulation_large(sc, direction="open")] == _WALL_IN + ["grasp", "base_follow", "retreat"]
    # the release as its own waypoint (the sampler records the exact reach fraction)
    d2 = _cabinet_push_finish(0.8547)
    d2["articulation"]["waypoints"]["0.8547"] = dict(d2["articulation"]["waypoints"]["0.75"])
    d2["articulation"]["handle_reach"].update({"0.8547": True, "1.0": False})
    sc2 = Sidecar.from_dict(d2)
    assert sc2.problems() == [] and [k for _, k in expand(sc2.alias, sc2)[3].articulation.followed_fractions()] == ["0.25", "0.5", "0.75", "0.8547"]
    # malformed records
    for mutate, match in ((lambda a: a.pop("follow_to"), "follow_to must be recorded"),
                          (lambda a: a["push_finish"].update(k_release=0.5), "the push starts where the hand lets go"),
                          (lambda a: a["push_finish"].update(swing_sign=0.0), "swing_sign"),
                          (lambda a: a["push_finish"].pop("contact"), "missing ['contact']"),
                          (lambda a: a["push_finish"].update(arm="middle"), "unknown arm"),
                          (lambda a: a.update(push_finish="planned"), "expected a dict")):
        bad = _cabinet_push_finish()
        mutate(bad["articulation"])
        problems = Sidecar.from_dict(bad).problems()
        assert any(match in p for p in problems), (match, problems)


def test_room_door_push_close_program_is_unchanged():
    """The room door's push-close (no ``close_by_push``) is what it was before the cabinet's room-door style close (values of 388fe4a2):
    the leaf geometry from the handle waypoints, every ``PushClose`` field, the targets and the annotation; the cabinet leaf options
    stay at their room-door defaults."""
    from ispatialgym.oracle.programs import door_leaf_geometry

    with open(os.path.join(VARIANTS, "configuration_matching-articulation_close_large_scale__room_door.json")) as f:
        sc = Sidecar.from_dict(json.load(f))
    assert door_leaf_geometry(sc.articulation) == {"hinge_xy": (-6.861306703398571, 5.211934738096777), "handle_radius_m": 0.7964860434373733,
                                                   "handle_xy_closed": (-6.827795, 4.416154), "approach_closed": (-0.999872, -0.015978, -1e-05), "handle_z": 1.036606}
    push = expand(sc.alias, sc)[-1]
    assert {k: getattr(push, k) for k in ("arm", "object", "joint", "hinge_xy", "handle_xy_closed", "approach_closed", "handle_radius_m", "handle_z", "q_from", "q_lower",
                                          "leaf_link", "push_inset_m", "push_height_drop_m", "hook_gap_m", "push_pre_m", "chunk_deg", "push_n", "closed_tol_rad",
                                          "max_chunks", "drag_m", "push_arms", "plan", "plan_repeats", "tol_span_rad")} == {
        "arm": "right", "object": "door.n.01_1", "joint": "j_leaf", "hinge_xy": (-6.861306703398571, 5.211934738096777), "handle_xy_closed": (-6.827795, 4.416154),
        "approach_closed": (-0.999872, -0.015978, -1e-05), "handle_radius_m": 0.7964860434373733, "handle_z": 1.036606, "q_from": 0.858809, "q_lower": 0.0,
        "leaf_link": "leaf", "push_inset_m": 0.15, "push_height_drop_m": 0.05, "hook_gap_m": 0.035, "push_pre_m": 0.1, "chunk_deg": 20.0, "push_n": 5,
        "closed_tol_rad": 0.05, "max_chunks": 10, "drag_m": 0.2, "push_arms": ("right", "left"), "plan": None, "plan_repeats": 2, "tol_span_rad": None}
    assert (push.leaf_width_m, push.face_offset_m, push.handle_standoff_m, push.q_closed) == (None, None, None, 0.0)
    assert push.targets(sc, None) == {"leaf": {"hinge_xy": [-6.861306703398571, 5.211934738096777], "handle_radius_m": 0.7964860434373733, "q_from": 0.858809, "q_to": 0.0}}
    assert push.annotate(sc)[1] == "close the door by pushing its outer face with the closed hand: joint j_leaf from 0.86 rad to 0.00 rad (49°), pushed in arcs of 20° about the hinge"


def _cabinet_close_by_push():
    """The close large-scale fixture's leaf as a cabinet door closed room-door style: ``articulation.close_by_push`` and the sampler's
    ``push_close`` record (knobs + the leaf; the fixture's hinge and its +1 swing), the recorded hybrid follow kept as the gate."""
    with open(os.path.join(VARIANTS, "configuration_matching-articulation_close_large_scale__room_door.json")) as f:
        d = json.load(f)
    d["articulation"].update({"door_class": "cabinet", "close_by_push": True,
                              "push_close": {"push_inset_m": 0.15, "push_height_drop_m": 0.05, "chunk_deg": 45.0, "drag_m": 0.08, "contact_clearance_m": 0.0307,
                                             "leaf_width_m": 0.80, "face_offset_m": 0.02, "handle_standoff_m": 0.055, "leaf_z": [0.0, 2.2], "swing_sign": 1.0,
                                             "hinge_world": [-6.8613, 5.2119, 1.0]}})
    return d


def test_cabinet_close_by_push_takes_the_room_door_branch():
    """Owner 2026-10-12 (close_large#03 c2): a cabinet close recorded with ``articulation.close_by_push`` is pushed shut room-door style -
    travel to the manipulation stance, ``push_close`` with no grasp, the leaf geometry from the handle waypoints (as the room door's)
    and the recorded leaf / knobs as ``PushClose`` options (``q_closed`` = the lower limit); no eef pose for the program check.
    ``false`` is the grasp program as before; a malformed record, a push finish alongside or an open is a sidecar problem."""
    from ispatialgym.oracle.programs import PUSH_CLOSE_OPTIONS, program_articulation_large
    from ispatialgym.sampling.v2.program_check import iter_pose_targets

    st = "configuration_matching-articulation_close_large_scale"
    d = _cabinet_close_by_push()
    sc = Sidecar.from_dict(d)
    assert sc.problems() == []
    actions = expand(sc.alias, sc)
    assert [a.name for a in actions] == _WALL_IN + ["push_close"]
    push, art = actions[-1], d["articulation"]
    assert isinstance(push, A.PushClose) and push.plan is None and (push.arm, push.arms) == ("right", ("left", "right"))
    assert {k: getattr(push, k) for k in PUSH_CLOSE_OPTIONS} == {k: art["push_close"][k] for k in PUSH_CLOSE_OPTIONS}
    assert (push.q_from, push.q_lower, push.q_closed) == (pytest.approx(art["q_init"]), pytest.approx(art["q_goal"]), pytest.approx(min(art["q_init"], art["q_goal"])))
    with open(os.path.join(VARIANTS, f"{st}__room_door.json")) as f:
        room = Sidecar.from_dict(json.load(f))
    door = expand(room.alias, room)[-1]
    assert [getattr(push, k) for k in ("hinge_xy", "handle_radius_m", "handle_xy_closed", "approach_closed", "handle_z")] == \
        [getattr(door, k) for k in ("hinge_xy", "handle_radius_m", "handle_xy_closed", "approach_closed", "handle_z")]
    assert list(iter_pose_targets(push.targets(sc, None))) == [] and push.annotate(sc)[1].endswith("pushed in arcs of 45° about the hinge")
    plain = copy.deepcopy(d)
    plain["articulation"]["close_by_push"] = False
    assert [a.name for a in expand(sc.alias, Sidecar.from_dict(plain))] == _WALL_IN + ["grasp", "base_follow", "retreat"]
    with pytest.raises(SidecarError, match="a close pushes the leaf shut"):
        program_articulation_large(sc, direction="open")
    for mutate, match in ((lambda a: a.update(close_by_push="yes"), "true or false"),
                          (lambda a: a.pop("push_close"), "expected a dict"),
                          (lambda a: a["push_close"].pop("leaf_width_m"), "missing ['leaf_width_m']"),
                          (lambda a: a["push_close"].update(chunk_deg="wide"), "must be numbers"),
                          (lambda a: a["push_close"].update(swing_sign="minus"), "['swing_sign'] must be numbers"),
                          (lambda a: a["push_close"].update(hinge_world=[-6.86, "x", 1.0]), "['hinge_world (3)'] must be numbers"),
                          (lambda a: a["push_close"].update(push_inset_m=0.9), "out of range"),
                          (lambda a: a["push_close"].update(handle_standoff_m=0.9), "out of range"),
                          (lambda a: a["push_close"].update(contact_clearance_m=0.12), "out of range"),
                          (lambda a: a["push_close"].update(swing_sign=-1.0), "the handle waypoints swing the leaf with +1"),
                          (lambda a: a["push_close"].update(hinge_world=[-6.80, 5.2119, 1.0]), "is not the hinge of the handle waypoints"),
                          (lambda a: a.update(push_finish={"arm": "right"}, follow_to=0.75), "exclude each other"),
                          (lambda a: a.update(joint_type="prismatic"), "a door leaf pushed shut is revolute")):
        bad = _cabinet_close_by_push()
        mutate(bad["articulation"])
        problems = Sidecar.from_dict(bad).problems()
        assert any(match in p for p in problems), (match, problems)


def test_attachment_second_base_pose_is_travelled_while_holding():
    """Owner decision 2026-09-16 17:10 item 2 (spec §6.8; tabletop ``58afae2``, core ``bd10c72``): when the attach pose
    and the brace are not both reachable from the grasp stance the sidecar records an ``attach`` stance reached with the
    male part in hand (``paths[] carried`` + ``transitions[] holding``, ``grasps[0].attach_stance`` / ``lift``). The
    program drives that move after the lift (navigate + transition, the held object in ``objects``), then reorients from
    the carried lift, braces, attaches and retreats both; the 19:06 run rejected every attempt in the program check."""
    st = "configuration_matching-attachment"
    with open(os.path.join(VARIANTS, f"{st}__second_base_pose.json")) as f:
        d = json.load(f)
    sc = Sidecar.from_dict(d)
    assert sc.problems() == [] and sc.validate() is sc
    assert sc.transition("manipulation", "attach").holding == "cap.n.02_1" and sc.path("manipulation", "attach").carried == "cap.n.02_1"
    actions = expand(sc.alias, sc)
    assert [a.name for a in actions] == _TABLE_IN + _PICK + ["navigate", "transition", "reorient", "brace", "place", "retreat", "retreat"]
    nav, tr, reo, br = actions[3], actions[4], actions[5], actions[6]
    assert nav.path.to == "attach" and nav.objects == ("cap.n.02_1",)
    assert tr.transition.to == "attach" and tr.carried == "cap.n.02_1" and tr.objects == ("cap.n.02_1",)
    assert tr.stance_to is sc.stance("attach") and tr.stance_from is sc.stance("manipulation")
    fine, numbers = tr.annotate(sc)
    assert "with the bottle cap" in fine and "with the bottle cap" in numbers
    # the reorient starts from the sampler's carried lift (grasps[0].lift at the grasp orientation), not from the unshifted lift
    assert reo.mode == "slerp" and reo.after is None and reo.start_pose is not None
    assert reo.start_pose.position == d["grasps"][0]["lift"] and reo.start_pose.orientation == d["grasps"][0]["grasp"]["orientation"]
    assert reo.waypoints[-1].position == d["grasps"][0]["pre_place"]["position"]
    # the brace / place annotate from the attach stance (its yaw); both arms retreat, the brace arm releases
    assert br.arms == ("left",) and br.objects == ("lemonade_bottle.n.01_1",)
    assert actions[-2].arm == "right" and actions[-1].arm == "left" and actions[-1].release is True
    if HAVE_CONVENTIONS:
        tg = reo.targets(sc, StubState())
        assert len(tg["waypoints"]) == len(d["grasps"][0]["waypoints"]) and len(tg["derived"]) == reo.n
        assert _close(tg["derived"][-1].position, tg["waypoints"][-1].position)
        first = tg["derived"][0].position
        # the derived line starts at the carried lift: its first point lies between the carried lift and the pre-attach
        for i in range(3):
            lo, hi = sorted([d["grasps"][0]["lift"][i], d["grasps"][0]["pre_place"]["position"][i]])
            assert lo - 1e-6 <= first[i] <= hi + 1e-6
    # without the sampler's lift record the carried lift is the recorded grasp + carry_height moved with the base
    d2 = copy.deepcopy(d)
    d2["grasps"][0].pop("lift")
    reo2 = [a for a in expand(sc.alias, Sidecar.from_dict(d2)) if a.name == "reorient"][0]
    assert _close(reo2.start_pose.position, d["grasps"][0]["lift"], tol=1e-6)
    # a transition swept with another object in hand, or with the object the program does not carry, is rejected
    d3 = copy.deepcopy(d)
    d3["transitions"][1]["holding"] = "lemonade_bottle.n.01_1"
    problems = Sidecar.from_dict(d3).problems()
    assert any("recorded holding='lemonade_bottle.n.01_1'" in p for p in problems), problems
    d4 = copy.deepcopy(d)
    d4["transitions"][0]["holding"] = "cap.n.02_1"  # init -> manipulation: nothing is in the hand yet
    problems = Sidecar.from_dict(d4).problems()
    assert any("transitions 'init' -> 'manipulation'" in p and "carries None" in p for p in problems), problems
    # an attach stance that is neither recorded nor reachable is rejected
    d5 = copy.deepcopy(d)
    d5["grasps"][0]["attach_stance"] = "placement"
    assert any("not recorded" in p for p in Sidecar.from_dict(d5).problems())
    d6 = copy.deepcopy(d)
    d6["paths"], d6["transitions"] = [], d6["transitions"][:1]
    assert any("no transition is recorded" in p or "neither a recorded path nor a transition" in p for p in Sidecar.from_dict(d6).problems())
    d7 = copy.deepcopy(d6)
    d7["stances"]["attach"]["translate"], d7["stances"]["attach"]["tilt"] = d["stances"]["manipulation"]["translate"], d["stances"]["manipulation"]["tilt"]
    assert any("neither a path nor a transition" in p for p in Sidecar.from_dict(d7).problems())
    # the base fixture (attach from the grasp stance) keeps its sequence
    base = load(st)
    assert [a.name for a in expand(base.alias, base)] == EXPECTED_SEQUENCE[st]


def test_attachment_without_a_brace_record_is_a_one_arm_attach():
    """Owner decision 2026-09-16 17:10 item 2: only when the joint placement has no cell after the second base pose is
    the attachment brace-less (``summary.brace: none``); the program is then grasp → lift → reorient → attach → retreat
    and ``contacts.brace`` is optional (the 19:06 run failed ``check_blocks`` on it)."""
    st = "configuration_matching-attachment"
    assert "contacts.brace" not in REQUIRED_BLOCKS[st] and "grasps" in REQUIRED_BLOCKS[st]
    with open(os.path.join(VARIANTS, f"{st}__brace_less.json")) as f:
        d = json.load(f)
    assert "contacts" not in d and d["summary"]["brace"] == "none"
    sc = Sidecar.from_dict(d)
    assert sc.problems() == [] and sc.validate() is sc and not sc.has_block("contacts.brace")
    actions = expand(sc.alias, sc)
    assert [a.name for a in actions] == _TABLE_IN + _PICK + ["reorient", "place", "retreat"]
    assert actions[-2].kind == "attach" and actions[-2].on == "lemonade_bottle.n.01_1" and actions[-1].arm == "right"
    assert all(a.arms == ("right",) for a in actions[1:])
    # a brace on the wrong arm / object is still rejected when recorded
    d2 = copy.deepcopy(raw(st))
    d2["contacts"]["brace"]["arm"] = "right"
    assert any("must use the other" in p or "same arm" in p for p in Sidecar.from_dict(d2).problems())
    d3 = copy.deepcopy(raw(st))
    d3["contacts"]["brace"]["object"] = "cap.n.02_1"
    assert any("attachment partner" in p for p in Sidecar.from_dict(d3).problems())



@pytest.mark.parametrize(
    "subtask,mutate,match",
    [
        ("disembedding", lambda d: d.update(alias="alignment-diagonal-x-y"), "matches no iddl"),
        ("disembedding", lambda d: d.update(task_family="alignment"), "task_family"),
        ("disembedding", lambda d: d.update(tro_sha256="abc"), "tro_sha256"),
        ("disembedding", lambda d: d["stances"].pop("placement"), "required block 'stances.placement'"),
        ("disembedding", lambda d: d.update(grasps=[]), "required block 'grasps'"),
        ("disembedding", lambda d: d["stances"]["init"].update(translate=2.3), "translate"),  # the JoyLo range is [0, 2] (ground half, W1 2026-09-20)
        ("disembedding", lambda d: d["transitions"][0].update(order=["trunk", "drive-in"]), "order"),
        ("disembedding", lambda d: d["paths"][0].update(to="nowhere"), "nowhere"),
        ("disembedding", lambda d: d["paths"][0].update(mode="teleport"), "mode"),
        ("disembedding", lambda d: d["grasps"][0].update(arm="middle"), "arm"),
        ("disembedding", lambda d: d["grasps"][0].pop("place_on"), "place_on"),
        ("disembedding", lambda d: d["language"]["objects"].pop("tray.n.01_2"), "language.objects has no entry"),
        ("disembedding", lambda d: d["grasps"][0].update(may_touch={"objects": ["box_of_cereal.n.01_1"]}), "may_touch"),
        ("disembedding", lambda d: d["grasps"][0].update(may_touch={"objects": ["bottle_of_beer.n.01_1"], "stay_on": "tray.n.01_1"}), "may_touch"),
        ("disembedding", lambda d: d["grasps"][0].update(may_touch={"objects": ["box_of_cereal.n.01_1"], "stay_on": "tray.n.01_1", "pre_grasp": ["tray.n.01_2"]}),
         "may_touch.pre_grasp"),
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
