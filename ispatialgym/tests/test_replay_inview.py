"""CPU tests of the offline in-view scoring (ispatialgym.multienv.replay_inview) and the replay's in-view preference."""
import math

import numpy as np

from ispatialgym.multienv import replay_inview as IV
from ispatialgym.multienv import replay_scene_perturbation as RSP


def _look(yaw):
    """xyzw quaternion of a USD camera looking horizontally along world yaw (camera -z -> (cos, sin, 0)), +y up."""
    from scipy.spatial.transform import Rotation

    f = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    z = -f
    x = np.cross([0, 0, 1.0], z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return Rotation.from_matrix(np.stack([x, y, z], axis=1)).as_quat()


def test_projection_front_back_and_size():
    box = {"size": [0.5, 0.5, 0.5], "offset": [0, 0, 0]}
    obj = {"pos": [3.0, 0.0, 1.0], "quat": [0, 0, 0, 1], "scale": [1, 1, 1]}
    c = IV.world_corners(obj, box)
    front = IV.projected_area(c, [0, 0, 1.0], _look(0.0), IV.DEFAULT_INTRINSICS)
    back = IV.projected_area(c, [0, 0, 1.0], _look(math.pi), IV.DEFAULT_INTRINSICS)
    far = IV.projected_area(IV.world_corners({**obj, "pos": [30.0, 0, 1.0]}, box), [0, 0, 1.0], _look(0.0), IV.DEFAULT_INTRINSICS)
    assert back == 0.0 and front > 0.004 and 0 < far < front / 50
    # the box centre projects to the principal point
    assert abs(front - (306 * 0.5 / 2.75) ** 2 / 720 ** 2) < 1e-6


def test_rooms_doors_and_rankings():
    rooms = {"a": (np.array([0.0, 0.0]), np.array([5.0, 5.0])), "b": (np.array([5.0, 0.0]), np.array([10.0, 5.0]))}
    assert IV.room_of(np.array([1.0, 1.0]), rooms) == "a" and IV.room_of(np.array([7.0, 1.0]), rooms) == "b"
    assert IV.room_of(np.array([20.0, 1.0]), rooms) is None
    assert IV._segment_hits_box([1, 2.5], [8, 2.5], np.array([4.7, 2.0]), np.array([5.3, 3.0]))
    assert not IV._segment_hits_box([1, 0.5], [8, 0.5], np.array([4.7, 2.0]), np.array([5.3, 3.0]))
    per = {"d1": {"scores": {"tv": 0.5, "chair": 0.02}}, "d2": {"scores": {"tv": 0.3, "lamp": 0.4}}, "d3": {"scores": {}}}
    r = IV.group_rankings(per, {"d1": "g", "d2": "g", "d3": "h"}, threshold=0.05)
    assert r == {"g": ["tv", "lamp"], "h": []}


def test_draw_one_prefers_in_view_single():
    units = [["chair"], ["sofa", "pillow"], ["tv"]]
    k = RSP.group_key(0, "g")
    d = RSP.draw_one(units, k, 1.0, prefer=["pillow", "sofa", "tv"])
    assert d["chosen"] == "sofa" and d["removed"] == ["pillow", "sofa"] and d["inview"]  # pillow is carried, not a head
    fb = RSP.draw_one(units, k, 1.0, prefer=["lamp"])
    assert "inview" not in fb and fb["chosen"] in ("chair", "sofa", "tv")
    assert RSP.draw_one(units, RSP.group_key(0, "x"), 0.0, prefer=["tv"])["removed"] == []  # p gate first


def test_merged_extent_is_task_rooms_plus_adjacent(tmp_path):
    import json
    from types import SimpleNamespace

    from ispatialgym.multienv.replay import merged_extent

    tro = tmp_path / "x-tro_state.json"
    tro.write_text(json.dumps({"ispatialgym": {"scene": {"task_rooms": ["meeting_room_0"], "model": "office"}}}))
    demo = SimpleNamespace(tro_path=str(tro), scene_file={}, full_rooms=["meeting_room_0", "conference_hall_0", "corridor_0"])
    adj = lambda model, rooms: ["conference_hall_0", "corridor_0"] if model == "office" and rooms == ["meeting_room_0"] else []
    ok = merged_extent(demo, adjacent=adj)
    assert ok["equal"] and ok["scene_extent"] == "task_rooms+adjacent"
    demo.full_rooms = demo.full_rooms + ["kitchen_0"]  # the whole house (or any extra room) is refused
    assert not merged_extent(demo, adjacent=adj)["equal"]
