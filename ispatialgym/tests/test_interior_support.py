"""Objects on an INTERIOR support surface (owner 2026-10-11: a shelf inside a bookcase / wardrobe / hall tree, the sampler's
``tabletop.support_surface`` recorded in template_meta ``template_facts.support_surface``): the oracle keeps the support as an
obstacle except its board pieces under the object (``execute.interior_split``, the reach bands' ``--surface-z`` rule) - the grasp
transit plans around the whole support, the approach / lift / descent / retreat finishes and the Retreat action leave out only the
board pieces. Without that record (every tabletop instance) each leg's collision world is the one it was."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import torch as th

from ispatialgym.oracle import execute as EX
from ispatialgym.oracle.actions import Place
from ispatialgym.oracle.motions import ApproachFinish, links_kw
from ispatialgym.oracle.programs import expand
from ispatialgym.tests.test_v2_oracle import iors_world, make_executor, success_eval, _place_goal

SHELF = "bookcase_x_0"


class _Mesh:
    def __init__(self, path, lo, hi):
        self.prim_path = path
        self.aabb = (th.tensor(lo, dtype=th.float32), th.tensor(hi, dtype=th.float32))


class _Shelf:
    """A bookcase compartment around the IORS test sidecar's grasp (7.20, 3.12) and place (7.36, 3.13) whose board's top is at
    ``z``: the board in three pieces (``board_mid`` under the grasp, ``board_mid`` + ``board_right`` under the place, ``board_left``
    under neither), a piece of the same height 1.3 m away, the board above, the side walls and the back panel - one link."""

    def __init__(self, z: float, name: str = SHELF):
        self.name, self.category = name, "bookcase"
        root = f"/World/scene_0/{name}/base_link/collisions"
        y0, y1 = 2.95, 3.35
        boxes = {
            "board_left": ([6.70, y0, z - 0.02], [6.90, y1, z]),
            "board_mid": ([6.90, y0, z - 0.02], [7.36, y1, z]),
            "board_right": ([7.36, y0, z - 0.02], [7.70, y1, z]),
            "board_far": ([8.70, y0, z - 0.02], [9.00, y1, z]),
            "board_above": ([6.70, y0, z + 0.33], [7.70, y1, z + 0.35]),
            "side_left": ([6.68, y0, 0.0], [6.70, y1, 1.8]),
            "side_right": ([7.70, y0, 0.0], [7.72, y1, 1.8]),
            "back": ([6.68, y1, 0.0], [7.72, y1 + 0.02, 1.8]),
        }
        self.meshes = {k: _Mesh(f"{root}/{k}", lo, hi) for k, (lo, hi) in boxes.items()}
        self.links = {"base_link": SimpleNamespace(collision_meshes=dict(self.meshes))}

    @property
    def aabb(self):
        los = th.stack([m.aabb[0] for m in self.meshes.values()])
        his = th.stack([m.aabb[1] for m in self.meshes.values()])
        return los.min(dim=0).values, his.max(dim=0).values

    def triples(self, *names):
        return [(self, "base_link", n) for n in names]


def _boxed(obj):
    """The fake object with a live world AABB (its position +- half its extent), as a scene object has."""
    obj.__class__ = type(f"Boxed{type(obj).__name__}", (type(obj),), {"aabb": property(lambda self: (self.pos - 0.5 * self.aabb_extent, self.pos + 0.5 * self.aabb_extent))})
    return obj


def _record(z: float, support: str = SHELF, surface: str = "interior"):
    return {"support": support, "spec": {"name": support, "init": "inside", "top_z": round(z, 2)}, "surface": surface, "z_world": z,
            "z_above_floor": z, "open_edges": ["-v"], "why": "init=inside"}


def _names(objs):
    return [getattr(o, "name", str(o)) for o in objs or ()]


def _links(links):
    return [f"{getattr(e[0], 'name', e[0])}:{':'.join(str(v) for v in e[1:])}" for e in links or ()]


# ------------------------------------------------------------------------------------------------ the record and the board


def test_the_interior_record_comes_from_template_meta_and_only_for_an_interior_surface():
    rec = _record(0.79)
    assert EX.interior_support_surface({"template_facts": {"support_surface": rec}}) is rec
    assert EX.interior_support_surface({"template_facts": {"support_surface": _record(0.79, surface="top")}}) is None
    assert EX.interior_support_surface({"template_facts": {"support_surface": {**rec, "z_world": None}}}) is None
    assert EX.interior_support_surface({"template_facts": {"support_surface": {**rec, "support": None}}}) is None
    assert EX.interior_support_surface({"template_facts": {}}) is None and EX.interior_support_surface({}) is None and EX.interior_support_surface(None) is None


def test_board_pieces_are_the_thin_pieces_at_the_board_height_under_the_box():
    shelf, cfg = _Shelf(0.793), EX.ExecuteConfig()
    got = EX.board_pieces(shelf, 0.793, (7.06, 3.02), (7.34, 3.22), cfg)
    assert got == shelf.triples("board_mid")  # not the board above, the side walls, the back panel or the pieces beside the box
    assert EX.board_pieces(shelf, 0.793, (7.22, 3.03), (7.50, 3.23), cfg) == shelf.triples("board_mid", "board_right")
    assert EX.board_pieces(shelf, 0.793 + 0.35, (7.06, 3.02), (7.34, 3.22), cfg) == shelf.triples("board_above")
    assert EX.board_pieces(shelf, 0.60, (7.06, 3.02), (7.34, 3.22), cfg) == []


def _split_state(shelf, obj_lo, obj_hi, eef=None):
    objs = {SHELF: shelf, "book.n.01_1": SimpleNamespace(name="book_0", aabb=(th.tensor(obj_lo), th.tensor(obj_hi)))}
    return SimpleNamespace(object=lambda name: objs[name], eef_pose=lambda arm: eef)


def test_interior_split_takes_the_support_out_and_keeps_only_the_board_under_the_object():
    shelf, floor = _Shelf(0.793), SimpleNamespace(name="floors_x_0")
    ctx = {"support_surface": _record(0.793)}
    st = _split_state(shelf, [7.16, 3.08, 0.793], [7.24, 3.16, 0.873])
    split = EX.interior_split(st, ctx, "book.n.01_1", [shelf, floor])
    assert split.support is shelf and split.objects == [floor] and split.links == shelf.triples("board_mid")
    assert split.note["support_was_ignored"] and split.note["board_pieces"] == ["base_link/board_mid"] and split.note["support"] == SHELF
    # a support no contact report found (the object asleep): the board under the object still leaves the contact legs' world
    split = EX.interior_split(st, ctx, "book.n.01_1", [floor])
    assert split.objects == [floor] and split.links == shelf.triples("board_mid") and not split.note["support_was_ignored"]
    # the place: the held object's footprint moved with the hand to the place pose (2 cm up, 16 cm along x)
    held = _split_state(shelf, [7.16, 3.08, 0.80], [7.24, 3.16, 0.88], eef=(th.tensor([7.2, 3.12, 0.863]), th.tensor([0.0, 0.0, 0.0, 1.0])))
    place = (th.tensor([7.36, 3.13, 0.885]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    split = EX.interior_split(held, ctx, "book.n.01_1", [shelf], place=place, arm="right")
    assert split.objects == [] and split.links == shelf.triples("board_mid", "board_right")
    assert abs(split.note["object_bottom_z"] - 0.822) < 1e-4


def test_interior_split_stays_out_of_every_other_situation():
    shelf, floor = _Shelf(0.793), SimpleNamespace(name="floors_x_0")
    st = _split_state(shelf, [7.16, 3.08, 0.793], [7.24, 3.16, 0.873])

    def no_query(*a, **k):
        raise AssertionError("no simulator query without the interior record")

    blind = SimpleNamespace(object=no_query, eef_pose=no_query)
    assert EX.interior_split(blind, {}, "book.n.01_1", [shelf, floor]) is None  # every tabletop instance
    assert EX.interior_split(blind, {"support_surface": _record(0.793, surface="top")}, "book.n.01_1", [shelf]) is None
    # the object is not on the recorded board: on the support's top (1.12) or one shelf up
    assert EX.interior_split(st, {"support_surface": _record(1.12)}, "book.n.01_1", [shelf, floor]) is None
    assert EX.interior_split(st, {"support_surface": _record(0.793 - 0.35)}, "book.n.01_1", [shelf, floor]) is None
    # the support neither ignored nor under the object: nothing to change (the record's support stands elsewhere)
    far_shelf = _Shelf(0.793)
    far = _split_state(far_shelf, [9.5, 3.08, 0.793], [9.6, 3.16, 0.873])
    assert EX.interior_split(far, {"support_surface": _record(0.793)}, "book.n.01_1", [floor]) is None
    # the support ignored but no board piece under the object (the band computer's warning case): the whole support stays
    split = EX.interior_split(far, {"support_surface": _record(0.793)}, "book.n.01_1", [far_shelf, floor])
    assert split.objects == [floor] and split.links == [] and "no_board_piece" in split.note


def test_links_keyword_only_when_there_are_links():
    assert links_kw(()) == {} and links_kw([]) == {} and links_kw(None) == {}
    shelf = _Shelf(0.793)
    assert links_kw(shelf.triples("board_mid")) == {"ignore_links": shelf.triples("board_mid")}
    assert ApproachFinish(pose=None).ignore_links == ()
    assert EX._plan_ignored([shelf], [(shelf, "base_link"), (shelf, "base_link", "board_mid")]) == [SHELF, f"{SHELF}:base_link", f"{SHELF}:base_link:board_mid"]


# ------------------------------------------------------------------------------------------------ the executor, end to end


def _world(tmp_path, record, relation="next to"):
    """The IORS ontop test world (apple grasped at z 0.863, set down 16 cm to the side at z 0.885) with the apple and the plate
    standing in a bookcase compartment whose board's top is at the apple's bottom; the place is a next-to place (its ignore set
    is the partner's support, as for nextto / touching). ``record``: None (a tabletop instance), or the record's board height."""
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    for o in env.objects.values():
        _boxed(o)
    z = float(env.objects["apple.n.01_1"].aabb[0][2])
    shelf, floor = _Shelf(z), SimpleNamespace(name="floors_x_0", category="floors")
    ex, motions, backend, _ = make_executor(env, sc, tmp_path)
    real_object = ex.state.object
    ex.state.object = lambda name: shelf if name == SHELF else real_object(name)
    ex.state.contact_supports = lambda name, margin=0.10, depth=2: [shelf, floor]
    ex.support_surface = None if record is None else EX.interior_support_surface({"template_facts": {"support_surface": _record(z + record)}})
    trace = {"world": [], "along": [], "to": [], "steps": []}
    real_update = backend.update_world

    def update_world(ignore_objects=(), ignore_links=()):
        trace["world"].append((_names(ignore_objects), _links(ignore_links)))
        return real_update(ignore_objects=ignore_objects)

    real_along, real_to = motions.arm_along, motions.arm_to

    def arm_along(arm, waypoints, *a, **kw):
        f = kw.get("finish")
        trace["along"].append({"finish": None if f is None else (_names(f.ignore_objects), _links(f.ignore_links)), "ignore": _names(kw.get("ignore_objects")),
                               "links": _links(kw.get("ignore_links")), "check": bool(kw.get("check_collisions"))})
        return real_along(arm, waypoints, *a, **kw)

    def arm_to(arms, *a, **kw):
        trace["to"].append((_names(kw.get("ignore_objects")), _links(kw.get("ignore_links"))))
        return real_to(arms, *a, **kw)

    backend.update_world, motions.arm_along, motions.arm_to = update_world, arm_along, arm_to
    env.after_step = lambda a: trace["steps"].append(a.clone())
    actions = [replace(a, relation=relation) if isinstance(a, Place) else a for a in expand(sc.variant, sc)]
    return sc, env, ex, motions, backend, shelf, actions, trace


def test_interior_instance_keeps_the_support_but_the_board_under_the_object(tmp_path):
    sc, env, ex, motions, backend, shelf, actions, trace = _world(tmp_path, record=0.0)
    names = [a.name for a in actions]
    result = ex.run_program(actions)
    assert result["success"], result["verdict"]
    assert all(SHELF not in objs for objs, _ in trace["world"]), trace["world"]  # never left out whole, in any leg
    assert all(SHELF not in objs for objs, _ in trace["to"]) and all(SHELF not in (rec["finish"] or ([], []))[0] + rec["ignore"] for rec in trace["along"])
    finishes = [rec["finish"] for rec in trace["along"] if rec["finish"] is not None]
    grasp_pieces, place_pieces = [f"{SHELF}:base_link:board_mid"], [f"{SHELF}:base_link:board_mid", f"{SHELF}:base_link:board_right"]
    # grasp approach, lift, carry, descent, place retreat: the board pieces under the object only on the legs at the contact
    assert (["apple_omzprq_0", "floors_x_0"], grasp_pieces) in finishes
    assert (["floors_x_0"], grasp_pieces) in finishes  # the lift
    assert (["floors_x_0"], []) in finishes  # the carry to the pre-place: the whole support stays
    assert (["floors_x_0"], place_pieces) in finishes  # the descent
    assert (["floors_x_0", "apple_omzprq_0"], place_pieces) in finishes  # the place's own retreat
    retreat = [rec for rec in trace["along"] if rec["check"]]  # the Retreat action's collision-checked leg
    assert retreat and retreat[-1]["ignore"] == ["apple_omzprq_0", "floors_x_0"] and retreat[-1]["links"] == place_pieces
    notes = {names[i]: rec.notes.get("interior_support") for i, rec in enumerate(ex.action_records)}
    assert notes["grasp"]["board_pieces"] == ["base_link/board_mid"] and notes["grasp"]["transit"] == "support present"
    assert notes["lift"]["board_pieces"] == ["base_link/board_mid"]
    assert notes["place"]["board_pieces"] == ["base_link/board_mid", "base_link/board_right"] and notes["place"]["carry"] == "support present"
    assert notes["retreat"]["support"] == SHELF and notes["transition"] is None
    grasp_rec = ex.action_records[names.index("grasp")]
    assert grasp_rec.notes["plan_ignored"] == ["floors_x_0", f"{SHELF}:base_link:board_mid"]


def test_a_stalled_interior_approach_is_finished_with_only_the_board_pieces_out(tmp_path):
    """The finishing ``arm_to`` of a stalled approach plans in the world without the board pieces under the object - the board
    above and the side walls are still there - and a tabletop finish passes no link-level ignore at all (the call it always made)."""
    for record, want in ((0.0, [f"{SHELF}:base_link:board_mid"]), (None, [])):
        sc, env, ex, motions, backend, shelf, actions, trace = _world(tmp_path / str(record), record=record)
        inner = motions.arm_along

        def stall_the_approach(arm, waypoints, *a, **kw):
            f = kw.get("finish")
            if f is not None and "apple_omzprq_0" in _names(f.ignore_objects):  # the approach's finish leaves the grasped apple out
                n = len(motions._densify(motions.eef_pose(arm), waypoints))
                backend.stall_ik_after = (n - 1, 0.005)  # the straight approach stalls 5 mm off its last waypoint
            return inner(arm, waypoints, *a, **kw)

        motions.arm_along = stall_the_approach
        calls = []
        real_to = motions.arm_to
        motions.arm_to = lambda arms, *a, **kw: (calls.append(kw), real_to(arms, *a, **kw))[1]
        for i, action in enumerate(actions[:actions.index(next(a for a in actions if a.name == "grasp")) + 1]):
            assert ex.run_action(i, action) is None
        finish_calls = [kw for kw in calls if kw.get("max_excursion") is not None]
        assert finish_calls, calls
        assert _links(finish_calls[-1].get("ignore_links")) == want and ("ignore_links" in finish_calls[-1]) == bool(want)
        assert SHELF not in _names(finish_calls[-1]["ignore_objects"]) if record is not None else SHELF in _names(finish_calls[-1]["ignore_objects"])


def test_without_the_interior_situation_every_leg_is_unchanged(tmp_path):
    """The default branch: a tabletop instance (no record), a record of another surface (the support's top) and a record whose board
    is not where the objects rest run the very same motions - the same world updates, finishes, arm_to calls and the same 23-D
    actions at every step - with the support left out whole as before and no link-level ignore anywhere."""
    runs = {}
    for label, record in (("tabletop", None), ("other_height", 0.35)):
        sc, env, ex, motions, backend, shelf, actions, trace = _world(tmp_path / label, record=record)
        result = ex.run_program(actions)
        assert result["success"], result["verdict"]
        assert all("interior_support" not in rec.notes for rec in ex.action_records)
        runs[label] = (trace, [rec.notes.get("plan_ignored") for rec in ex.action_records])
    base, other = runs["tabletop"], runs["other_height"]
    assert base[0]["world"] == other[0]["world"] and base[0]["along"] == other[0]["along"] and base[0]["to"] == other[0]["to"] and base[1] == other[1]
    assert len(base[0]["steps"]) == len(other[0]["steps"]) and all(th.equal(a, b) for a, b in zip(base[0]["steps"], other[0]["steps"]))
    assert all(not links for _, links in base[0]["world"]) and all(not rec["links"] and not (rec["finish"] or ([], []))[1] for rec in base[0]["along"])
    finishes = [rec["finish"][0] for rec in base[0]["along"] if rec["finish"] is not None]
    assert ["apple_omzprq_0", SHELF, "floors_x_0"] in finishes  # the grasp approach still leaves the whole support out
    # a record of the support's top never reaches an action
    assert EX.interior_support_surface({"template_facts": {"support_surface": _record(0.79, surface="top")}}) is None
