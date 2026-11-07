"""of-v2maps 2026-11-07: the band view's scene mask (ispatialgym.sampling.v2.scene_mask) on a fake sampler."""

from contextlib import contextmanager
from types import SimpleNamespace

import torch as th

from ispatialgym.sampling.v2 import scene_mask as SM


def _fake(obstacle_z=0.80, trav=True):
    hull = th.tensor([[-0.02, -0.02, 0.0], [0.02, -0.02, 0.0], [0.02, 0.02, 0.0], [-0.02, 0.02, 0.0], [0.0, 0.0, 0.03]])

    def footprint(h, pose):
        return h[:, :2] + pose[0][:2]

    top = SimpleNamespace(z=0.75, polygon_supported=lambda fp, margin=0.0: bool((fp[:, 0] < 1.0).all()))

    def sphere_hits(sp, margin=0.0):  # an obstacle filling everything below obstacle_z beyond x 1.2 (a wall foot / a shelf)
        return (sp[..., 2] < obstacle_z) & (sp[..., 0] > 1.2)

    def robot_spheres_world(q):  # the eef-fixed links 5 cm behind the eef along -z, one arm link elsewhere (not eef-fixed)
        sp = th.tensor([[[0.0, 0.0, 0.05, 0.02], [0.0, 0.0, 0.10, 0.03], [0.5, 0.5, 0.5, 0.05]]])
        return sp, ["right_gripper_link", "right_arm_link7", "right_arm_link5"]

    kit = SimpleNamespace(trav_free=lambda xy, eroded=False: trav, world_sphere_hits=sphere_hits, stats={},
                          backend=SimpleNamespace(robot_spheres_world=robot_spheres_world),
                          eef_pose=lambda q, arm: (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])),
                          robot=SimpleNamespace(gripper_control_idx={"right": th.tensor([0, 1])}))

    @contextmanager
    def ignoring(objs):
        yield

    ikc = SimpleNamespace(q_locked=lambda st: th.zeros(4), ignoring=ignoring, eef_in_frame=lambda pose, cam: True)
    s = SimpleNamespace(world=SimpleNamespace(footprint=footprint, floor_z=0.0, scene=SimpleNamespace(objects=[])), kit=kit, ikc=ikc,
                        camera_pose_at=lambda st: None)
    stance = SimpleNamespace(base_xy=th.tensor([0.0, 0.0]))
    return s, stance, top, hull


def _pose_at(cell, yaw):
    return (th.tensor([float(cell[0]), float(cell[1]), 0.76]), th.tensor([0.0, 0.0, 0.0, 1.0]))


def test_categories_mark_only_the_certain_pairs():
    s, stance, top, hull = _fake()
    cells = th.tensor([[0.5, 0.0], [0.99, 0.0], [1.3, 0.0]])  # on the top / footprint over the edge / off the top and at the obstacle
    g = SimpleNamespace(name="top_0")
    eio = (th.tensor([0.0, 0.0, 0.05]), th.tensor([0.0, 0.0, 0.0, 1.0]))
    ev = SM.evaluate(s, stance, top, "right", cells, [0.0, 1.57], [g], _pose_at, lambda gg: eio, support_obj=None, hull=hull,
                     anchor_z=0.75, cats=["support", "height", "trav", "static", "hand"])
    m = ev["masks"]
    assert m["support"][:, 0, 0].tolist() == [False, True, True]
    assert not m["height"].any() and not m["trav"].any()
    assert m["static"][:, 0, 0].tolist() == [False, False, True] and m["hand"][:, 0, 0].tolist() == [False, False, True]
    assert ev["any"][:, :, 0].sum().item() == 4 and ev["n"] == 6 and ev["removed"]["support"] == 4


def test_the_eef_fixed_links_only_and_the_switches(monkeypatch):
    s, stance, top, hull = _fake(trav=False)
    c, r, links = SM.hand_local(s, stance, "right", 0.02)
    assert links == ["right_gripper_link", "right_arm_link7"] and [round(v, 4) for v in r.tolist()] == [0.02, 0.03]
    ev = SM.evaluate(s, stance, top, "right", th.tensor([[0.5, 0.0]]), [0.0], [SimpleNamespace(name="g")], _pose_at,
                     lambda gg: (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])), support_obj=None, hull=hull, anchor_z=0.70, cats=["height", "trav"])
    assert bool(ev["masks"]["height"].all()) and bool(ev["masks"]["trav"].all())  # 5 cm from the anchor; the base off the eroded map
    monkeypatch.setenv("ISPATIALGYM_MAP_SCENE_MASK_CATS", "hand,frame,bogus")
    assert SM.categories() == ["hand", "frame"]
    monkeypatch.delenv("ISPATIALGYM_MAP_SCENE_MASK_CATS")
    assert SM.categories() == list(SM.CATEGORIES)
    monkeypatch.setenv("ISPATIALGYM_MAP_SCENE_MASK_DIAG", "1")
    assert SM.active() and SM.diag_on() and not SM.mask_on()
