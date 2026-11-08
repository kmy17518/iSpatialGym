"""i7-lean 2026-11-07: ``interiors.measure`` on a fake scene (rays against boxes) - the open-top rule from the wall RIM (inside_large_scale_03's
drop-in sink under a cabinet: the headroom 0.67 m is no cap, the walls stop at ~0.18 m), its near wall, the cabinet as an ``above_<object>``
obstacle box, and a closed compartment (walls up to the board above) staying closed. Simulator-free."""

import math
from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.sampling import interiors as IN

BOARD_Z = 0.7751


def _ray_box(o, e, lo, hi):
    """Entry distance of the segment o -> e into the box [lo, hi], or None."""
    d = [e[i] - o[i] for i in range(3)]
    t0, t1 = 0.0, 1.0
    for i in range(3):
        if abs(d[i]) < 1e-12:
            if o[i] < lo[i] or o[i] > hi[i]:
                return None
            continue
        a, b = (lo[i] - o[i]) / d[i], (hi[i] - o[i]) / d[i]
        a, b = min(a, b), max(a, b)
        t0, t1 = max(t0, a), min(t1, b)
        if t0 > t1:
            return None
    return t0 * math.sqrt(sum(v * v for v in d))


class Scene:
    """Boxes (body prim path, lo, hi); ``raycast`` = the closest hit like ``SimWorld.raycast`` (``hit``, ``rigidBody``, ``distance``)."""

    def __init__(self, boxes, raster, objects):
        self.boxes, self.raster = boxes, raster
        self.floor_z = 0.0
        self.scene = SimpleNamespace(objects=objects)
        self.robot = SimpleNamespace(links={})

    def support_interior(self, support, z, z_margin=0.03):
        return self.raster

    def raycast(self, starts, ends):
        out = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            best = None
            for body, lo, hi in self.boxes:
                t = _ray_box(s, e, lo, hi)
                if t is not None and (best is None or t < best[0]):
                    best = (t, body)
            out.append({"hit": best is not None, "rigidBody": best[1] if best else "", "distance": best[0] if best else 0.0})
        return out


def _raster(x0, x1, y0, y1, headroom):
    xs = th.arange(x0 + 0.01, x1, 0.02)
    ys = th.arange(y0 + 0.01, y1, 0.02)
    mask = th.ones(len(xs), len(ys), dtype=th.bool)

    def cells(free_only=True):
        idx = th.nonzero(mask)
        return th.stack([xs[idx[:, 0]], ys[idx[:, 1]]], dim=1)

    def edges():  # (name, mid, outward normal, length): the robot stands at -x
        return [("-u", th.tensor([x0, 0.5 * (y0 + y1)]), th.tensor([-1.0, 0.0]), y1 - y0), ("+u", th.tensor([x1, 0.5 * (y0 + y1)]), th.tensor([1.0, 0.0]), y1 - y0),
                ("-v", th.tensor([0.5 * (x0 + x1), y0]), th.tensor([0.0, -1.0]), x1 - x0), ("+v", th.tensor([0.5 * (x0 + x1), y1]), th.tensor([0.0, 1.0]), x1 - x0)]

    return SimpleNamespace(z=BOARD_Z, resolution=0.02, mask=mask, headroom=th.full(mask.shape, headroom), headroom_cap=1.5, open_edges=[], cells=cells,
                           edges=edges)


def _support(name, boxes):
    lo = [min(b[1][i] for b in boxes) for i in range(3)]
    hi = [max(b[2][i] for b in boxes) for i in range(3)]
    link = SimpleNamespace(name="base_link", prim_path=f"/World/{name}/base_link", collision_meshes={}, aabb=(th.tensor(lo), th.tensor(hi)))
    return SimpleNamespace(name=name, links={"base_link": link}, aabb=(th.tensor(lo), th.tensor(hi)))


def _sink_scene(cabinet=True):
    z0, rim = BOARD_Z, BOARD_Z + 0.18
    sink = "/World/drop_in_sink_lkklqs_0/base_link"
    walls = [(sink, [-0.03, -0.20, z0 - 0.02], [0.0, 0.20, rim]),    # near wall (toward the robot at -x)
             (sink, [0.40, -0.20, z0 - 0.02], [0.43, 0.20, rim]),    # back wall
             (sink, [-0.03, 0.17, z0 - 0.02], [0.43, 0.20, rim]),    # left
             (sink, [-0.03, -0.20, z0 - 0.02], [0.43, -0.17, rim]),  # right
             (sink, [-0.03, -0.20, z0 - 0.04], [0.43, 0.20, z0])]    # the basin floor
    support = _support("drop_in_sink_lkklqs_0", walls)
    boxes, objects = list(walls), [support]
    if cabinet:
        cab = "/World/cabinet_x_0/base_link"
        lo, hi = [-0.10, -0.40, z0 + 0.67], [0.50, 0.40, z0 + 1.10]
        boxes.append((cab, lo, hi))
        objects.append(SimpleNamespace(name="cabinet_x_0", links={"base_link": SimpleNamespace(name="base_link", prim_path=cab, aabb=(th.tensor(lo), th.tensor(hi)))}))
    return Scene(boxes, _raster(0.0, 0.40, -0.17, 0.17, 0.665 if cabinet else 1.5), objects), support


def test_a_sink_under_a_cabinet_is_an_open_top_basin():
    world, sink = _sink_scene()
    sp = IN.measure(world, sink, 0.77, name="drop_in_sink_lkklqs_0@basin", scene="house_double_floor_lower", side=(-1.0, 0.0),
                    edge_distances=(0.35, 0.45, 0.55, 0.65), z_margin=0.25)
    assert sp.open_top and sp.open_sides == ["top"] and sp.height_m == pytest.approx(0.7751)
    assert sp.rim_m == pytest.approx(0.17, abs=0.011) and sp.headroom_m == pytest.approx(0.67)  # the walls' top, not the cabinet's underside
    assert sp.front_wall_x_m == pytest.approx(0.0, abs=1e-6) and sp.back_thickness_m is not None  # the near wall, the back wall
    assert sp.depth_m == pytest.approx(0.40, abs=1e-6) and sp.width_m == pytest.approx(0.34, abs=1e-6) and abs(sp.centre_y_m) < 1e-6
    (name, ctr, half, yaw), = sp.extra_boxes
    assert name == "above_cabinet_x_0" and yaw == 0.0 and ctr[2] - half[2] == pytest.approx(0.67, abs=1e-4)  # the cabinet over the sink
    assert sp.source["overhead"]["object"] == "cabinet_x_0" and sp.source["overhead"]["bottom_above_board_m"] == pytest.approx(0.67, abs=1e-4)
    names = [n for n, _ in sp.boxes(sp.height_m, 0.45)]
    assert names == ["board", "wall_left", "wall_right", "back", "front", "above_cabinet_x_0"]  # no board above: the cabinet is its own box
    wl = dict(sp.boxes(sp.height_m, 0.45))["wall_left"]
    assert wl.centre[2] + wl.half[2] == pytest.approx(sp.height_m + sp.rim_m)
    forced = IN.measure(world, sink, 0.77, name="x", scene="s", side=(-1.0, 0.0), z_margin=0.25, open_top=False)
    assert not forced.open_top and forced.rim_m is None and not forced.extra_boxes  # the override


def test_a_basin_with_nothing_above_and_a_closed_compartment():
    world, sink = _sink_scene(cabinet=False)
    sp = IN.measure(world, sink, 0.77, name="open", scene="s", side=(-1.0, 0.0), z_margin=0.25)
    assert sp.open_top and sp.rim_m == pytest.approx(0.17, abs=0.011) and not sp.extra_boxes
    # a closed compartment: the walls reach the board above (a fridge / bookcase cube) - not open-top, whatever its headroom
    z0 = BOARD_Z
    c = "/World/cube/base_link"
    boxes = [(c, [0.0, -0.20, z0 - 0.02], [0.40, 0.20, z0]), (c, [0.0, -0.20, z0 + 0.30], [0.40, 0.20, z0 + 0.32]),
             (c, [0.0, 0.17, z0], [0.40, 0.20, z0 + 0.30]), (c, [0.0, -0.20, z0], [0.40, -0.17, z0 + 0.30]), (c, [0.40, -0.20, z0], [0.42, 0.20, z0 + 0.30])]
    cube = _support("cube", boxes)
    r = _raster(0.0, 0.40, -0.17, 0.17, 0.295)
    r.open_edges = ["-u"]
    w = Scene(boxes, r, [cube])
    cs = IN.measure(w, cube, 0.77, name="cube", scene="s", z_margin=0.03)
    assert not cs.open_top and cs.rim_m is None and cs.front_wall_x_m is None and cs.headroom_m == pytest.approx(0.30) and cs.back_thickness_m is not None
    assert [n for n, _ in cs.boxes(cs.height_m, 0.45)] == ["board", "above", "wall_left", "wall_right", "back"]


def test_the_registered_digests_do_not_move():
    reg = IN.load_registry(IN.REGISTRY)
    # the fridge entry: the door-per-piece spec since 2026-11-08 (integration, lead decision; was c623427d265d, one door box)
    assert reg["bookcase_kmuacd_0@0.8354"].digest() == "b58d4e5c790e" and reg["fridge_petcxr_0@0.7064/j_left_door2.3"].digest() == "0b539d8a83ab"


def test_an_open_door_is_boxed_per_collision_piece(monkeypatch):
    """The fridge door (job 18368160: one box over the door and its bins held the idle shoulder at every near stand-off): one oriented box
    per collision piece of the moved link by default, one box over all of them with door_boxes="link" (the registered fridge spec's)."""
    import sys
    import types

    fake_tu = types.ModuleType("omnigibson.utils.transform_utils")
    fake_tu.transform_points = lambda pts, tf: th.as_tensor(pts, dtype=th.float32)
    monkeypatch.setitem(sys.modules, "omnigibson", types.ModuleType("omnigibson"))
    utils = types.ModuleType("omnigibson.utils")
    utils.transform_utils = fake_tu
    monkeypatch.setitem(sys.modules, "omnigibson.utils", utils)
    monkeypatch.setitem(sys.modules, "omnigibson.utils.transform_utils", fake_tu)
    z0 = BOARD_Z
    c = "/World/fridge/base_link"
    boxes = [(c, [0.0, -0.20, z0 - 0.02], [0.40, 0.20, z0]), (c, [0.0, -0.20, z0 + 0.24], [0.40, 0.20, z0 + 0.26]),
             (c, [0.0, 0.17, z0], [0.40, 0.20, z0 + 0.24]), (c, [0.0, -0.20, z0], [0.40, -0.17, z0 + 0.24]), (c, [0.40, -0.20, z0], [0.42, 0.20, z0 + 0.24])]
    fr = _support("fridge", boxes)
    yaw = 0.8
    R = th.tensor([[math.cos(yaw), -math.sin(yaw), 0.0], [math.sin(yaw), math.cos(yaw), 0.0], [0.0, 0.0, 1.0]])

    def piece(lo, hi):  # a convex piece in the door's frame, turned by the door's yaw, hinged at (-0.05, 0.22)
        P = th.tensor([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        return SimpleNamespace(points_in_parent_frame=P @ R.T + th.tensor([-0.05, 0.22, 0.0]))

    door = SimpleNamespace(name="left_door", prim_path="/World/fridge/left_door", scaled_transform=None,
                           collision_meshes={"panel": piece([0.0, 0.0, 0.0], [0.04, 0.44, 1.9]), "bin": piece([0.04, 0.05, 0.9], [0.25, 0.39, 1.0])},
                           get_position_orientation=lambda: (th.zeros(3), th.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)])))
    fr.links["left_door"] = door
    r = _raster(0.0, 0.40, -0.17, 0.17, 0.235)
    r.open_edges = ["-u"]
    w = Scene(boxes, r, [fr])
    sp = IN.measure(w, fr, 0.77, name="f", scene="s", z_margin=0.03, joints=None, moved_links=["left_door"])
    names = [b[0] for b in sp.extra_boxes]
    assert names == ["door_left_door_0", "door_left_door_1"] and all(abs(b[3] - yaw) < 1e-5 for b in sp.extra_boxes)
    panel, bin_ = sp.extra_boxes
    assert sorted(round(v, 3) for v in panel[2]) == [0.02, 0.22, 0.95] and sorted(round(v, 3) for v in bin_[2]) == [0.05, 0.105, 0.17]
    one = IN.measure(w, fr, 0.77, name="f", scene="s", z_margin=0.03, moved_links=["left_door"], door_boxes="link")
    assert [b[0] for b in one.extra_boxes] == ["door_left_door"] and one.extra_boxes[0][2][0] > 0.12 and sp.source["door_boxes"] == "pieces"


def test_a_two_bay_board_is_measured_in_the_bay_at_the_lateral():
    """i7-t03specs 2026-11-08: bookcase_otwukr_1's boards span two bays (0.303 / 0.3105 m) either side of a 5 cm divider at the raster's
    lateral centre: from the centre the wall rays start inside the divider (no hit) and the width falls back to the whole raster; with
    ``lateral_m`` the rays start in the bay at that offset (the robot's left facing inward = -y here: the robot stands at -x)."""
    z0 = BOARD_Z
    c = "/World/two_bay/base_link"
    boxes = [(c, [0.0, -0.3845, z0 - 0.05], [0.30, 0.3845, z0]),            # the board (both bays)
             (c, [0.0, -0.3845, z0 + 0.28], [0.30, 0.3845, z0 + 0.33]),     # the board above
             (c, [0.0, 0.3327, z0], [0.30, 0.3845, z0 + 0.28]), (c, [0.0, -0.3845, z0], [0.30, -0.3327, z0 + 0.28]),  # side walls
             (c, [0.0, -0.0222, z0], [0.30, 0.0296, z0 + 0.28]),            # the divider
             (c, [0.288, -0.3845, z0], [0.30, 0.3845, z0 + 0.28])]          # the back
    case = _support("two_bay", boxes)
    r = _raster(0.0, 0.288, -0.3327, 0.3327, 0.275)
    r.open_edges = ["-u"]
    w = Scene(boxes, r, [case])
    whole = IN.measure(w, case, 0.77, name="w", scene="s", z_margin=0.03)
    assert whole.width_m == pytest.approx(0.66, abs=0.011)  # the fallback: the raster's span (both bays and the divider as one compartment)
    # the edge "-u" (outward -x): inward +x, tangent (compartment_frame: [-n_y, n_x]) = +y
    left = IN.measure(w, case, 0.77, name="l", scene="s", z_margin=0.03, lateral_m=0.19)
    right = IN.measure(w, case, 0.77, name="r", scene="s", z_margin=0.03, lateral_m=-0.19)
    assert left.width_m == pytest.approx(0.3327 - 0.0296, abs=1e-4) and left.centre_y_m == pytest.approx(0.5 * (0.3327 + 0.0296), abs=1e-4)
    assert right.width_m == pytest.approx(0.3327 - 0.0222, abs=1e-4) and right.centre_y_m == pytest.approx(-0.5 * (0.3327 + 0.0222), abs=1e-4)
    assert left.back_thickness_m is not None and left.depth_m == pytest.approx(0.298, abs=0.011) and "lateral +0.190" in left.source["measured"]
    assert whole.digest() != left.digest() and "lateral" not in whole.source["measured"]  # the default is unchanged
