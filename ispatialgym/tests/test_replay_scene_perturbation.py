"""CPU tests of the replay-time scene perturbation (ispatialgym.multienv.replay_scene_perturbation)."""
import json

import numpy as np

from ispatialgym.multienv import replay_scene_perturbation as RSP


def box(lo, hi):
    return np.asarray(lo, float), np.asarray(hi, float)


def test_touch_and_components():
    boxes = {"table": box([0, 0, 0], [1, 1, 0.8]), "cup": box([0.4, 0.4, 0.8], [0.5, 0.5, 0.9]),
             "chair": box([3, 3, 0], [3.5, 3.5, 1]), "lamp": box([5, 5, 0], [5.2, 5.2, 1.5])}
    assert RSP.aabbs_touch(boxes["table"], boxes["cup"], 0.0)
    assert not RSP.aabbs_touch(boxes["table"], boxes["chair"], 0.01)
    assert RSP.components(list(boxes), boxes, 0.01) == [["chair"], ["cup", "table"], ["lamp"]]


def test_protected_member_protects_unit_and_unknown_box():
    boxes = {"table": box([0, 0, 0], [1, 1, 0.8]), "cup": box([0.4, 0.4, 0.8], [0.5, 0.5, 0.9]),
             "chair": box([3, 3, 0], [3.5, 3.5, 1]), "ghost": None}
    units, prot = RSP.removable_units(["table", "cup", "chair", "ghost"], boxes, {"cup": "moved"}, 0.01)
    assert units == [["chair"]]
    assert prot["table"] == "unit:cup" and prot["cup"] == "moved" and prot["ghost"] == "aabb unknown"


def test_draw_deterministic_and_independent_per_index():
    units = [[f"o{i}"] for i in range(12)]
    a = RSP.draw(units, RSP.episode_key(0, "demo", 0), 1.0, 0.5)
    b = RSP.draw(units, RSP.episode_key(0, "demo", 0), 1.0, 0.5)
    assert a == b and a["perturbed"]
    draws = {tuple(RSP.draw(units, RSP.episode_key(0, "demo", r), 1.0, 0.5)["removed"]) for r in range(8)}
    assert len(draws) > 4  # replay indices draw different removals
    assert RSP.draw(units, 1, 0.0, 1.0) == {"perturbed": False, "units_removed": [], "removed": []}
    full = RSP.draw(units, 1, 1.0, 1.0)
    assert full["removed"] == sorted(n for u in units for n in u)


def test_draw_rate():
    units = [["a"], ["b"]]
    n = sum(RSP.draw(units, RSP.episode_key(3, "d", r), 0.5, 0.5)["perturbed"] for r in range(2000))
    assert 900 < n < 1100


def test_moved_objects():
    f0 = {"a": {"root_link": {"pos": np.zeros(3)}}, "b": {"root_link": {"pos": np.ones(3)}}, "c": {"joint": [0.0, 0.0]}}
    f1 = {"a": {"root_link": {"pos": np.zeros(3) + 5e-5}}, "b": {"root_link": {"pos": np.ones(3) + 0.01}}, "c": {"joint": [0.0, 0.3]}}
    assert RSP.moved_objects([f0, f1], 1e-4) == {"b", "c"}
    assert RSP.moved_objects([f0], 1e-4) == set()


def test_task_objects_and_exempt():
    meta = {"tracked_objects": {"x.n.01_1": "target_x"}, "protected_objects": {"y.n.01_1": "dest_y"}, "task_family": "disembedding"}
    tro = {"floor.n.01_1": {"scene_object_name": "floors_a"}, "z.n.01_1": {"scene_object_name": "distractor_z"}}
    t = RSP.task_objects(meta, tro)
    assert set(t) == {"target_x", "dest_y", "floors_a", "distractor_z"}
    assert not RSP.exempt(meta)
    cam = {"task_family": "camera_relocalization", "variant": "camera_relocalization-standard"}
    assert not RSP.exempt(cam) and RSP.is_camera_relocalization(cam) and RSP.exempt(cam, ("camera_relocalization",))


def test_structural_and_frames():
    assert RSP.is_structural("walls") and RSP.is_structural("elevator_door") and not RSP.is_structural("desk")
    assert RSP.sample_frames(5, 40) == [0, 1, 2, 3, 4]
    fr = RSP.sample_frames(1000, 4)
    assert fr[0] == 0 and fr[-1] == 999 and len(fr) == 4


def test_records_roundtrip(tmp_path):
    rec = {"tag": "REPLAY_SCENE_PERT", "schema": RSP.SCHEMA, "demo_name": "d1", "replay_index": 2, "removed": ["a", "b"]}
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps({"tag": "REPLAY_DEMO"}) + "\n" + json.dumps(rec) + "\n")
    assert RSP.load_from_records(str(p)) == {("d1", 2): ["a", "b"]}
    p2 = tmp_path / "r.log"
    p2.write_text("noise\n".replace("noise\n", "") + "REPLAY_SCENE_PERT " + json.dumps({k: v for k, v in rec.items() if k != "tag"}) + "\n")
    assert RSP.load_from_records(str(p2)) == {("d1", 2): ["a", "b"]}


def test_structural_matches_sampling_policy():
    from scripts.campaign_20261110.training_scene_policy import STRUCTURAL

    assert RSP.STRUCTURAL == STRUCTURAL


class _Prim:
    def __init__(self, typ="Xform", children=()):
        self.typ, self.children = typ, list(children)

    def GetTypeName(self):
        return self.typ

    def GetChildren(self):
        return self.children


class _Obj:
    def __init__(self, scene, name, category, lo, hi, light=False):
        self.scene, self.name, self.category = scene, name, category
        self.base = box(lo, hi)
        self.prim = _Prim(children=[_Prim("SphereLight")] if light else [])
        self.visible = True

    @property
    def aabb(self):
        off = self.scene.offsets.get(self.name, np.zeros(3))
        return self.base[0] + off, self.base[1] + off


class _Link:
    def __init__(self, scene):
        self.scene = scene

    @property
    def collision_boundary_points_world(self):
        c = self.scene.hand
        return np.stack([c - 0.05, c + 0.05])


class _Robot:
    name = "robot_r1"

    def __init__(self, scene):
        self.links = {"hand": _Link(scene)}


class _Scene:
    """Frames: 0 cup on table, hand far; 1 hand at shelf; 2 cup moved (on table)."""

    def __init__(self):
        self.offsets, self.hand = {}, np.array([9.0, 9.0, 1.0])
        self.objs = [_Obj(self, "table", "breakfast_table", [0, 0, 0], [1, 1, 0.8]),
                     _Obj(self, "target_cup", "cup", [0.4, 0.4, 0.8], [0.5, 0.5, 0.9]),
                     _Obj(self, "vase", "vase", [0.8, 0.8, 0.8], [0.9, 0.9, 1.0]),
                     _Obj(self, "shelf", "shelf", [3, 0, 0], [3.5, 1, 2]),
                     _Obj(self, "chair", "chair", [5, 5, 0], [5.5, 5.5, 1]),
                     _Obj(self, "sofa", "sofa", [7, 0, 0], [8, 1, 1]),
                     _Obj(self, "pillow", "pillow", [7.2, 0.2, 1.0], [7.5, 0.5, 1.2]),
                     _Obj(self, "lamp", "floor_lamp", [9, 0, 0], [9.2, 0.2, 1.5], light=True),
                     _Obj(self, "walls_a", "walls", [0, -1, 0], [10, -0.9, 3]),
                     _Obj(self, "parked_chair", "chair", [200, 200, 0], [200.5, 200.5, 1])]
        self.robots = [_Robot(self)]
        self.objects = self.objs + [type("R", (), {"name": "robot_r1"})()]

    def object_registry(self, key, name):
        return next((o for o in self.objs if o.name == name), None)

    def _frame(self, i):
        i = int(i)
        return {"cup": np.array([0.0, 0.0, 0.0]) if i < 2 else np.array([0.1, 0.0, 0.0]),
                "hand": np.array([3.2, 0.5, 1.0]) if i == 1 else np.array([9.0, 9.0, 1.0])}

    def deserialize(self, flat):
        f = self._frame(flat)
        reg = {o.name: {"pos": (f["cup"] if o.name == "target_cup" else np.zeros(3))} for o in self.objs}
        reg["robot_r1"] = {"pos": f["hand"]}
        return {"registry": {"object_registry": reg}}, None

    def load_state(self, flat, serialized=True):
        f = self._frame(flat)
        self.offsets = {"target_cup": f["cup"]}
        self.hand = f["hand"]


class _Demo:
    def __init__(self, d, name="demo_a"):
        self.dir, self.name, self.T, self.tro_path, self.campaign_meta = d, name, 3, None, None

    def state_at(self, i):
        return i


def test_plan_on_fake_scene(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"tracked_objects": {"cup.n.01_1": "target_cup"}, "task_family": "x"}))
    scene, demo = _Scene(), _Demo(str(tmp_path))
    rec = RSP.plan(scene, demo, 0, 0, {"mode": "subset", "p_episode": 1.0, "p_unit": 1.0})
    # table: touches the target cup; vase: same unit as the table; shelf: robot contact at frame 1; lamp holds a light; walls structural
    assert rec["units"] == [["chair"], ["pillow", "sofa"]]
    assert rec["removed"] == ["chair", "pillow", "sofa"]
    p = rec["protected"]
    assert p["target_cup"].startswith("tracked_objects") and p["table"] == "task/moved contact@0" and p["vase"].startswith("unit:")
    assert p["shelf"].startswith("robot contact@1") and p["lamp"] == "holds light" and "walls_a" not in p
    assert rec["offstage"] == ["parked_chair"] and "parked_chair" not in p
    assert rec["robot_boxes_min"] == 1 and rec["removed_min_gap_m"]["chair"]["robot"] > 3.0
    hidden = RSP.apply(scene, rec["removed"])
    assert sorted(o.name for o in scene.objs if not o.visible) == hidden
    other = _Scene()
    vm = RSP.visibility_matrix([scene, other], [hidden, []])
    assert vm["independent"] and vm["visible"][0] == [False] * 3 and vm["visible"][1] == [True] * 3
    RSP.clear(scene, hidden)
    assert all(o.visible for o in scene.objs)
    forced = RSP.plan(_Scene(), demo, 5, 0, forced=["chair"])
    assert forced["removed"] == ["chair"] and forced["forced"]
    try:
        RSP.plan(_Scene(), demo, 5, 0, forced=["table"])
        raise AssertionError("protected forced removal accepted")
    except ValueError:
        pass


def test_plan_camera_relocalization_not_exempt_obstacle_protected(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"task_family": "camera_relocalization", "tracked_objects": {}}))
    tro = tmp_path / "t-tro_state.json"
    tro.write_text(json.dumps({"ispatialgym": {"object_bindings": {"obstacle.n.01_1": {"scene_object_name": "sofa"}}}}))
    demo = _Demo(str(tmp_path))
    demo.tro_path = str(tro)
    rec = RSP.plan(_Scene(), demo, 0, 0, {"p_episode": 1.0, "p_unit": 1.0})
    assert "exempt" not in rec and rec["protected"]["sofa"].startswith("tro_binding:obstacle")
    assert "sofa" not in rec["removed"] and "pillow" not in rec["removed"]  # the obstacle's unit stays whole


def test_aabb_gap():
    a = box([0, 0, 0], [1, 1, 1])
    assert RSP.aabb_gap(a, []) is None
    assert RSP.aabb_gap(a, [box([0.5, 0.5, 0.5], [2, 2, 2])]) == 0.0
    assert abs(RSP.aabb_gap(a, [box([4, 1, 1], [5, 2, 2]), box([1, 1, 3], [2, 2, 4])]) - 2.0) < 1e-12


def test_probe_targets_and_look_at():
    assert RSP.probe_targets([["a", "b"], ["a"], ["c"], []]) == ["a", "b", "c"]
    assert RSP.probe_targets([["a"], ["a"]]) == []
    from scipy.spatial.transform import Rotation

    q = RSP.look_at_quat(np.array([0.0, -2.0, 1.0]), np.array([0.0, 0.0, 1.0]))
    fwd = Rotation.from_quat(q).apply([0.0, 0.0, -1.0])
    assert np.allclose(fwd, [0.0, 1.0, 0.0], atol=1e-9)
    up = Rotation.from_quat(q).apply([0.0, 1.0, 0.0])
    assert up[2] > 0.99


def test_plan_outside_crop_scope(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"tracked_objects": {"cup.n.01_1": "target_cup"}, "task_family": "x"}))
    scene = _Scene()
    # the crop = table, cup, vase, shelf, sofa; outside = chair, pillow (pillow rests on the inside sofa), parked_chair (offstage)
    inside = {"table", "target_cup", "vase", "shelf", "sofa", "walls_a"}
    rec = RSP.plan(scene, _Demo(str(tmp_path)), 0, 0, {"p_episode": 1.0, "p_unit": 1.0}, inside_crop=inside)
    assert rec["units"] == [["chair"]] and rec["removed"] == ["chair"]
    assert rec["protected"]["pillow"] == "touches crop" and rec["inside_crop_excluded"] == 6
    assert "sofa" not in rec["protected"] and "table" not in [n for u in rec["units"] for n in u]


class _Attr:
    def __init__(self, v):
        self.v = v

    def Get(self):
        return self.v

    def Set(self, v):
        self.v = v


class _LPrim:
    def __init__(self, path, typ="Xform", children=(), intensity=None):
        self.path, self.typ, self.children = path, typ, list(children)
        self.attr = _Attr(intensity) if intensity is not None else None

    def GetTypeName(self):
        return self.typ

    def GetChildren(self):
        return self.children

    def GetPath(self):
        return self.path

    def GetAttribute(self, name):
        return self.attr


class _LObj:
    def __init__(self, name, intensity):
        self.name = name
        self.prim = _LPrim(f"/W/{name}", children=[_LPrim(f"/W/{name}/l", "SphereLight", intensity=intensity)])


class _Sky:
    intensity, color = 2500.0, (1.0, 1.0, 1.0)
    _isg_campaign_defaults = None


class _Sim:
    skybox = _Sky()

    def editing_usd(self):
        import contextlib

        return contextlib.nullcontext()


class _LScene:
    def __init__(self):
        self.objects = [_LObj("lamp_in", 100.0), _LObj("lamp_out", 50.0), _LObj("desk_out", None)]
        self.objects[2].prim = _LPrim("/W/desk_out")

    def object_registry(self, key, name):
        return next((o for o in self.objects if o.name == name), None)


def test_merged_lighting_crop_view_and_outside():
    from ispatialgym.multienv import campaign_rendering as CR
    from ispatialgym.sampling import campaign_augmentation as A

    scene, sim = _LScene(), _Sim()
    view = CR.crop_view(scene, {"lamp_in"})
    assert [o.name for o in view.objects] == ["lamp_in"] and view.object_registry("name", "lamp_out") is not None
    req = {"schema": A.LIGHT_SCHEMA, "intensity_factor": 1.2, "skybox_color_factors": [1.0, 1.0, 1.0]}
    rec = A.apply_lighting(view, sim, req)
    assert set(rec["fixtures"]) == {"lamp_in:/l"} and abs(scene.objects[0].prim.children[0].attr.v - 120.0) < 1e-9
    out = CR.apply_outside_lighting(scene, sim, {"lamp_in"}, req)
    assert out["outside_fixtures"] == 1 and abs(scene.objects[1].prim.children[0].attr.v - 60.0) < 1e-9
    # a second episode with another draw starts from the untouched defaults, inside and outside
    assert CR.crop_view(scene, {"lamp_in"}) is view
    req2 = dict(req, intensity_factor=0.8)
    A.apply_lighting(view, sim, req2, expected=None)
    CR.apply_outside_lighting(scene, sim, {"lamp_in"}, req2)
    assert abs(scene.objects[0].prim.children[0].attr.v - 80.0) < 1e-9 and abs(scene.objects[1].prim.children[0].attr.v - 40.0) < 1e-9
    # the recorded record of the crop (as collected in the cropped world) is accepted against the view
    A.apply_lighting(view, sim, req, expected=rec)


def test_draw_one_single_object_group_keyed():
    units = [["chair"], ["pillow", "sofa"], ["lamp_table"], ["stool"]]
    keys = [RSP.group_key(0, f"group{g}") for g in range(40)]
    picks = [RSP.draw_one(units, k, 1.0) for k in keys]
    assert all(len(p["removed"]) == 1 and p["removed"][0] in ("chair", "lamp_table", "stool") for p in picks)  # groups never split
    assert len({p["removed"][0] for p in picks}) == 3
    assert RSP.draw_one(units, keys[0], 1.0) == picks[0]  # same group -> same object
    assert RSP.draw_one([["a", "b"]], keys[0], 1.0) == {"perturbed": False, "removed": [], "no_single": True,
                                                         "order_head": RSP.draw_one([["a", "b"]], keys[0], 1.0)["order_head"]}
    n = sum(RSP.draw_one(units, RSP.group_key(1, f"g{i}"), 0.5)["perturbed"] for i in range(2000))
    assert 900 < n < 1100
    # a member whose removable set lacks the group's first pick takes the next one in the same order
    k = keys[0]
    first = picks[0]["removed"][0]
    other = RSP.draw_one([u for u in units if u != [first]], k, 1.0)["removed"][0]
    assert other != first


def test_plan_one_object_same_for_group(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"tracked_objects": {"cup.n.01_1": "target_cup"}, "task_family": "x"}))
    a = RSP.plan(_Scene(), _Demo(str(tmp_path), "member_a"), 0, 0, {"p_episode": 1.0}, group="G1")
    b = RSP.plan(_Scene(), _Demo(str(tmp_path), "member_b"), 0, 0, {"p_episode": 1.0}, group="G1")
    assert a["removed"] == b["removed"] and len(a["removed"]) == 1 and a["group"] == "G1"
    assert a["removed"][0] == "chair"  # the only single-object unit (pillow+sofa is a group)
