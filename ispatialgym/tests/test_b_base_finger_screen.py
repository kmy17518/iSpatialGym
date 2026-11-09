"""i7-goalmaps r4 2026-11-09 (lead decision #68): the base_link x hand screen of the F_init pre-screen at B (ISPATIALGYM_B_BASE_FINGER_SCREEN)."""
import math

import torch as th

from ispatialgym.sampling.v2.tasks import iors_large as IL


class _GP:
    def __init__(self, gen):
        self._g = dict(gen)

    def gen(self, key, default=None):
        return self._g.get(key, default)


def test_the_mode_is_off_by_default_and_reads_env_then_customs(monkeypatch):
    monkeypatch.delenv(IL.B_BASE_FINGER_SCREEN_ENV, raising=False)
    assert IL.b_base_finger_mode(None) == "off" and IL.b_base_finger_mode(_GP({})) == "off"
    assert IL.b_base_finger_mode(_GP({"b_base_finger_screen": True})) == "on"
    assert IL.b_base_finger_mode(_GP({"b_base_finger_screen": "shadow"})) == "shadow"
    monkeypatch.setenv(IL.B_BASE_FINGER_SCREEN_ENV, "0")
    assert IL.b_base_finger_mode(_GP({"b_base_finger_screen": True})) == "off"
    monkeypatch.setenv(IL.B_BASE_FINGER_SCREEN_ENV, "shadow")
    assert IL.b_base_finger_mode(None) == "shadow"


def test_penetration_along_the_approach_line():
    # one hand sphere at the eef origin (r 0.02), one base sphere at the world origin (r 0.10): an approach passing 0.11 m from it
    # overlaps by 0.01 m in its middle, one passing 0.20 m away does not
    hand_local, hand_r = th.zeros(1, 3), th.tensor([0.02])
    base_w, base_r = th.zeros(1, 3), th.tensor([0.10])
    pairs = th.tensor([[0, 0]])
    ts = th.linspace(0, 1, 11)[:, None]
    near = th.tensor([-0.05, 0.11, 0.0]) + ts * th.tensor([0.10, 0.0, 0.0])   # passes x = 0 at y = 0.11
    far = th.tensor([-0.05, 0.20, 0.0]) + ts * th.tensor([0.10, 0.0, 0.0])
    pen = IL.hand_base_penetration(th.stack([near, far]), th.eye(3).repeat(2, 1, 1), hand_local, hand_r, base_w, base_r, pairs)
    assert abs(float(pen[0]) - 0.01) < 1e-6 and float(pen[1]) < 0


def test_the_hand_sphere_offset_turns_with_the_eef():
    # the hand sphere 0.05 m along the eef x: turning the eef by +90 deg about z moves it to +y
    hand_local, hand_r = th.tensor([[0.05, 0.0, 0.0]]), th.tensor([0.01])
    base_w, base_r = th.tensor([[0.0, 0.05, 0.0]]), th.tensor([0.01])
    pairs = th.tensor([[0, 0]])
    c, s = math.cos(math.pi / 2), math.sin(math.pi / 2)
    Rz = th.tensor([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    line = th.zeros(1, 2, 3)
    pen = IL.hand_base_penetration(line, Rz[None], hand_local, hand_r, base_w, base_r, pairs)
    assert float(pen[0]) > 0.019
    pen0 = IL.hand_base_penetration(line, th.eye(3)[None], hand_local, hand_r, base_w, base_r, pairs)
    assert float(pen0[0]) < 0


class _G:
    def __init__(self, name):
        self.name, self.pose = name, (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0]))


def _fake(monkeypatch, mode, pens):
    s = IL.IORSLarge.__new__(IL.IORSLarge)
    s.gp = _GP({"b_base_finger_screen": mode})
    s.kit = type("K", (), {"stats": {}})()
    monkeypatch.setattr(IL.TT, "grasp_twins_on", lambda _s: True)
    monkeypatch.setattr(IL.IORSLarge, "yaws", lambda self: [k * 2 * math.pi / 16 for k in range(16)])
    monkeypatch.setattr(IL.IORSLarge, "pose_on", lambda self, xy, yaw, z: (th.zeros(3), th.tensor([0.0, 0.0, 0.0, 1.0])))
    monkeypatch.setattr(IL.IORSLarge, "_base_finger_pen", lambda self, st, arm, eefs: th.tensor(pens))
    return s


class _St:
    base_pose2d = (0.0, 0.0, 0.0)


def test_a_pair_is_dropped_only_when_row_and_twin_collide(monkeypatch):
    monkeypatch.delenv(IL.B_BASE_FINGER_SCREEN_ENV, raising=False)
    # pairs: A row+twin hit -> dropped; B row hits, twin clear -> kept; C clear -> kept
    s = _fake(monkeypatch, True, [0.01, 0.02, 0.01, -0.01, -0.02, -0.03])
    F = [(0.0, _G("a")), (0.3927, _G("b")), (0.7854, _G("c"))]
    out = s._base_finger_filter(F, th.zeros(2), 0.0, _St(), "left")
    assert [g.name for _, g in out] == ["b", "c"]
    assert s.kit.stats["b_base_finger_screen.dropped"] == 1 and s.kit.stats["b_base_finger_screen.kept"] == 2


def test_shadow_keeps_every_pair_and_scores_the_b_outcome(monkeypatch):
    monkeypatch.delenv(IL.B_BASE_FINGER_SCREEN_ENV, raising=False)
    s = _fake(monkeypatch, "shadow", [0.01, 0.02, -0.01, -0.02])
    F = [(0.0, _G("a")), (0.3927, _G("b"))]
    out = s._base_finger_filter(F, th.zeros(2), 0.0, _St(), "left")
    assert len(out) == 2 and s.kit.stats["b_base_finger_screen.shadow_drop"] == 1
    s._base_finger_shadow_outcome(_St(), "left", 0.0, _G("a"), ok=False)
    s._base_finger_shadow_outcome(_St(), "left", 0.3927, _G("b"), ok=True)  # not predicted: not scored
    assert s.kit.stats["b_base_finger_screen.shadow_confirmed"] == 1 and "b_base_finger_screen.shadow_false" not in s.kit.stats


def test_only_touching_large_02_turns_the_screen_on_in_the_customs():
    import json
    import os

    from ispatialgym import REPO_ROOT

    d = json.load(open(os.path.join(REPO_ROOT, "ispatialgym", "assets", "task_specs", "isg_customs.json")))
    setters = sorted(k for k, v in d.items() if isinstance(v, dict) and "b_base_finger_screen" in (v.get("generation") or {}))
    assert setters == ["inter_object_relation_satisfaction-touching_large_scale-umbrella-hall_tree-floor"], setters
    assert IL.b_base_finger_mode(_GP(d[setters[0]]["generation"])) == "on"
