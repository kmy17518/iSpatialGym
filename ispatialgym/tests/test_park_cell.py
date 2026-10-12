"""Park cells (owner 2026-10-12 ~02:40): the world-load cell (``SimWorld.park_cell``) and the per-attempt cell
(``navigation.park_pose``) keep the tucked robot - its footprint grown by ``PARK_CLEARANCE_M`` - off every scene object's
AABB and over the loaded rooms, nearest to the centre the cell came from before (``SimWorld.choose_park_cell``); a cell that
was clear stays the choice. ``reset_scene_objects`` names what it puts back. Synthetic 6.1 m x 6.1 m rooms on OmniGibson's own
traversability / segmentation map classes (0.1 m cells, world x = (column - 40) * 0.1, y = (row - 40) * 0.1)."""

import logging
from types import SimpleNamespace as NS

import torch as th

from omnigibson.maps.segmentation_map import SegmentationMap
from omnigibson.maps.traversable_map import TraversableMap

from ispatialgym.sampling import sim_utils as SU
from ispatialgym.sampling.v2 import navigation as NAV

EXTENT = th.tensor([0.728, 0.684, 1.478])  # R1Pro reset_joint_pos_aabb_extent
SIZE = 80


def _cells(x0, x1, y0, y1):
    """Map index slices (rows, columns) of the world rectangle [x0, x1) x [y0, y1)."""
    return slice(int(round(y0 / 0.1)) + SIZE // 2, int(round(y1 / 0.1)) + SIZE // 2), slice(int(round(x0 / 0.1)) + SIZE // 2, int(round(x1 / 0.1)) + SIZE // 2)


def _box(name, lo, hi, category="ottoman", baked=None):
    """``baked``: the object's footprint on the traversability map, ``(x0, x1, y0, y1)`` or True for its AABB's."""
    if baked is True:
        baked = (lo[0], hi[0], lo[1], hi[1])
    return NS(name=name, category=category, aabb=(th.tensor(lo, dtype=th.float32), th.tensor(hi, dtype=th.float32)), baked=baked)


def _world(objects=(), rooms=("bedroom_0",), loaded=None, split_x=None):
    """A ``SimWorld`` over one room, cells at x and y in [-3, 3] (walls around it); ``split_x``: the room east of it is ``corridor_0``
    (open plan, same floor). Objects with a ``baked`` footprint are marked on the traversability map, as the scenes'
    furniture is."""
    floor = th.zeros(SIZE, SIZE, dtype=th.uint8)
    floor[_cells(-3.0, 3.1, -3.0, 3.1)] = 255
    for o in objects:
        if o.baked:
            floor[_cells(*o.baked)] = 0
    trav = TraversableMap(map_resolution=0.1)
    trav.map_size, trav.floor_map, trav.floor_heights = SIZE, [floor], [0.0]
    seg = object.__new__(SegmentationMap)
    seg.map_size, seg.map_resolution = SIZE, 0.1
    seg.room_ins_map = th.zeros(SIZE, SIZE, dtype=th.int32)
    seg.room_ins_map[_cells(-3.0, 3.1, -3.0, 3.1)] = 1
    if split_x is not None:
        seg.room_ins_map[_cells(split_x, 3.1, -3.0, 3.1)] = 2
    seg.room_ins_id_to_ins_name = {1: "bedroom_0", 2: "corridor_0"}
    seg.room_ins_name_to_ins_id = {v: k for k, v in seg.room_ins_id_to_ins_name.items()}
    w = object.__new__(SU.SimWorld)
    w.robot = NS(name="robot_r1", reset_joint_pos_aabb_extent=EXTENT.clone())
    w.scene = NS(trav_map=trav, seg_map=seg, objects=[w.robot] + list(objects))
    w.rooms, w.loaded_rooms, w.floor_z = list(rooms), list(loaded or rooms), 0.0
    return w


def _xy(t):
    return round(float(t[0]), 3), round(float(t[1]), 3)


def _free_cells(w):
    trav = w.scene.trav_map
    m = trav._erode_trav_map(th.clone(trav.floor_map[0]), robot=w.robot)
    return trav.map_to_world(th.nonzero(m == 255))


def _clear_of(w, xy, objects, margin=SU.PARK_CLEARANCE_M):
    half = EXTENT[:2] / 2 + margin
    for o in objects:
        lo, hi = o.aabb
        if float(xy[0]) - half[0] < hi[0] and float(xy[0]) + half[0] > lo[0] and float(xy[1]) - half[1] < hi[1] and float(xy[1]) + half[1] > lo[1]:
            return False
    return True


def test_world_load_park_cell_leaves_a_clear_room_as_it_was():
    # the eroded free cells run from -2.7 to 2.7 in x and y: centre (0, 0), the nearest cell is the centre cell itself
    assert _xy(_world().park_cell()) == (0.0, 0.0)
    # objects outside the parked robot's body do not count: a carpet within the floor band, a ceiling fan above the
    # chassis height, the floors it stands on
    w = _world([_box("carpet_0", (-1.0, -1.0, 0.0), (1.0, 1.0, 0.012), "carpet"), _box("ceiling_fan_0", (-0.5, -0.5, 2.1), (0.5, 0.5, 2.8), "ceiling_fan"),
                _box("floors_0", (-3.0, -3.0, -0.1), (3.0, 3.0, 0.0), "floors")])
    assert _xy(w.park_cell()) == (0.0, 0.0)


def test_world_load_park_cell_moves_off_an_object_the_eroded_map_lets_it_touch(caplog):
    """The ottoman is on the traversability map (a footprint 10 cm inside its AABB), and the erosion (a 7-cell kernel: 0.3 m)
    leaves cells closer to its AABB than the chassis' half-extent (0.364 m in x): the old choice puts the robot into it
    (B8: 6.5 cm), the new one is the nearest cell to the same centre whose footprint + margin is clear."""
    ottoman = _box("ottoman_hnmndv_1", (0.0, -0.6, 0.0), (0.5, 0.6, 0.47), baked=(0.1, 0.4, -0.5, 0.5))
    w = _world([ottoman])
    pts = _free_cells(w)
    rows = th.nonzero(w.scene.trav_map._erode_trav_map(th.clone(w.scene.trav_map.floor_map[0]), robot=w.robot) == 255).float()
    dist = th.linalg.norm(rows - rows.mean(dim=0), dim=1)
    old = pts[int(th.argmin(dist))]
    assert not _clear_of(w, old, [ottoman], margin=0.0)  # the cell before 2026-10-12 overlaps the ottoman itself
    with caplog.at_level(logging.INFO, logger="ispatialgym.sampling"):
        new = w.park_cell()
    assert _clear_of(w, new, [ottoman])
    clear = th.tensor([_clear_of(w, p, [ottoman]) for p in pts])
    assert abs(float(dist[clear].min()) - float(dist[int(th.nonzero((pts == new).all(dim=1))[0])])) < 1e-6  # nearest clear cell
    assert "world load: park cell" in caplog.text and "ottoman_hnmndv_1 by" in caplog.text


def test_park_pose_moves_off_the_object_on_the_old_cell():
    """axial#02's case (hotel_suite_large): 1 m from the support pulls the centre next to an ottoman the map does not keep
    the chassis off; ``park_pose`` takes the nearest clear cell to the same centre, still >= 1 m from the support."""
    desk = _box("desk_0", (-2.9, -2.9, 0.0), (-2.3, 2.9, 0.75), "desk", baked=True)
    w = _world([desk])
    kit = NS(world=w, eroded_trav_map=lambda floor=0: w.scene.trav_map._erode_trav_map(th.clone(w.scene.trav_map.floor_map[0]), robot=w.robot))
    # a desk along the west wall: x >= -1.3 (1 m from it), 41 columns up to 2.7, centre (0.7, 0)
    assert _xy(NAV.park_pose(kit, ["bedroom_0"], keep_away=[desk])) == (0.7, 0.0)
    ottoman = _box("ottoman_hnmndv_1", (0.9, -0.3, 0.0), (1.3, 0.3, 0.47))
    w = _world([desk, ottoman])
    kit.world = w
    xy = NAV.park_pose(kit, ["bedroom_0"], keep_away=[desk])
    # x + 0.364 + 0.05 <= 0.9 -> 0.4 (0.3 from the centre); |y| >= 0.3 + 0.342 + 0.05 -> 0.7 (0.7 away)
    assert _xy(xy) == (0.4, 0.0) and _clear_of(w, xy, [ottoman])


def test_the_footprint_stays_over_the_loaded_rooms():
    """Partial-room loading: no floor outside the loaded rooms, so a footprint corner over an unloaded neighbour (open plan,
    no wall to keep it off) rules the cell out."""
    w = _world(split_x=0.0, loaded=("bedroom_0",))
    pts = th.tensor([[-0.3, 0.0], [-0.4, 0.0], [-1.5, 1.0]])
    assert w.park_clear_mask(pts).tolist() == [False, True, True]  # -0.3 + 0.364 > 0: the corridor
    assert w.choose_park_cell(pts, th.tensor([0.0, 0.1, 1.2]), "test") == 1
    w.loaded_rooms = ["bedroom_0", "corridor_0"]
    assert w.park_clear_mask(pts).tolist() == [True, True, True]


def test_the_boxes_are_per_collision_piece():
    """An L-shaped bar along the south and west walls: its object AABB covers the centre cell, its two collision pieces stay
    2 m away - the centre cell stays the cell. The pieces' boxes come from the render-side link transform, as ``obj.aabb``'s."""
    arm_x = th.tensor([[-3.5, -3.5, -1.0], [0.0, -3.0, 0.1]])  # link frame; the link sits at (1, 1, 1)
    arm_y = th.tensor([[-3.5, -3.5, -1.0], [-3.0, 1.5, 0.1]])
    tf = th.eye(4)
    tf[:3, 3] = th.tensor([1.0, 1.0, 1.0])
    link = NS(collision_meshes={"m0": NS(points_in_parent_frame=arm_x), "m1": NS(points_in_parent_frame=arm_y)}, scaled_transform=tf)
    bar = NS(name="bar_0", category="bar", links={"base_link": link}, aabb=(th.tensor([-2.5, -2.5, 0.0]), th.tensor([1.0, 2.5, 1.1])), baked=None)
    w = _world([bar])
    names, lo, hi = w.park_obstacles()
    assert names == ["bar_0", "bar_0"] and _xy(lo[0]) == (-2.5, -2.5) and _xy(hi[0]) == (1.0, -2.0) and _xy(hi[1]) == (-2.0, 2.5)
    assert w.park_blockers(th.tensor([0.0, 0.0])) == [] and _xy(w.park_cell()) == (0.0, 0.0)
    # 1.7 m from the centre the footprint + margin reaches 11.4 cm into the west arm
    assert [(n, round(d, 3)) for n, d in w.park_blockers(th.tensor([-1.7, 0.0]))] == [("bar_0", 0.114)]
    # judged by the object box the centre cell would be ruled out (x up to 1.0 + 0.414)
    w = _world([_box("bar_0", (-2.5, -2.5, 0.0), (1.0, 2.5, 1.1), "bar")])
    assert float(w.park_cell()[0]) >= 1.414


def test_no_clear_cell_keeps_the_nearest_cell_with_a_warning(caplog):
    w = _world([_box("bar_0", (-3.0, -3.0, 0.0), (3.0, 3.0, 1.1), "bar")])
    with caplog.at_level(logging.WARNING, logger="ispatialgym.sampling"):
        assert _xy(w.park_cell()) == (0.0, 0.0)
    assert "world load: no park cell keeps the tucked robot 0.05 m clear" in caplog.text and "bar_0 by" in caplog.text


def test_reset_scene_objects_names_what_it_puts_back(caplog):
    class Obj:
        def __init__(self, name, pos):
            self.name, self.pos, self.n_joints = name, th.tensor(pos), 0

        def get_position_orientation(self):
            return self.pos, th.tensor([0.0, 0.0, 0.0, 1.0])

        def load_state(self, state, serialized=False):
            loaded.append(self.name)
            self.pos = th.as_tensor(state["root_link"]["pos"])

        def keep_still(self):
            pass

    loaded = []
    root = lambda pos: {"root_link": {"pos": pos, "ori": [0.0, 0.0, 0.0, 1.0], "lin_vel": [0.0] * 3, "ang_vel": [0.0] * 3}}  # noqa: E731
    w = object.__new__(SU.SimWorld)
    w.stable_scene = {"state": {"registry": {"object_registry": {"ottoman_hnmndv_1": root([0.2, 0.0, 0.29]), "bed_0": root([-1.0, 0.0, 0.2]),
                                                                  "chair_0": root([1.0, 1.0, 0.4])}}}}
    w.robot, w.spawned = NS(name="robot_r1"), {}
    w.scene = NS(objects=[Obj("ottoman_hnmndv_1", [0.33, 0.0, 0.29]), Obj("bed_0", [-1.0, 0.0, 0.2]), Obj("chair_0", [1.0, 1.0025, 0.4])])
    with caplog.at_level(logging.INFO, logger="ispatialgym.sampling"):
        assert w.reset_scene_objects(context="alias-x: instance 0 attempt 2") == 2
    assert loaded == ["ottoman_hnmndv_1", "chair_0"] and [n for n, _, _ in w.last_reset] == ["ottoman_hnmndv_1", "chair_0"]
    assert ("alias-x: instance 0 attempt 2: 2 scene objects put back to the scene file: ottoman_hnmndv_1 130.0 mm / 0.0 deg, "
            "chair_0 2.5 mm / 0.0 deg") in caplog.text
    caplog.clear()
    with caplog.at_level(logging.INFO, logger="ispatialgym.sampling"):
        assert w.reset_scene_objects() == 0
    assert w.last_reset == [] and "put back" not in caplog.text
