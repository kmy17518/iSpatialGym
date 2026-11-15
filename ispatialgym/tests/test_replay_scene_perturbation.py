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
    assert RSP.exempt({"task_family": "camera_relocalization", "variant": "camera_relocalization-standard"})


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
                     _Obj(self, "walls_a", "walls", [0, -1, 0], [10, -0.9, 3])]
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
    rec = RSP.plan(scene, demo, 0, 0, {"p_episode": 1.0, "p_unit": 1.0})
    # table: touches the target cup; vase: same unit as the table; shelf: robot contact at frame 1; lamp holds a light; walls structural
    assert rec["units"] == [["chair"], ["pillow", "sofa"]]
    assert rec["removed"] == ["chair", "pillow", "sofa"]
    p = rec["protected"]
    assert p["target_cup"].startswith("tracked_objects") and p["table"] == "task/moved contact@0" and p["vase"].startswith("unit:")
    assert p["shelf"].startswith("robot contact@1") and p["lamp"] == "holds light" and "walls_a" not in p
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


def test_plan_exempt_camera_relocalization(tmp_path):
    (tmp_path / "meta.json").write_text(json.dumps({"task_family": "camera_relocalization"}))
    rec = RSP.plan(_Scene(), _Demo(str(tmp_path)), 0, 0, {"p_episode": 1.0, "p_unit": 1.0})
    assert rec["exempt"] == "camera_relocalization" and rec["removed"] == []
