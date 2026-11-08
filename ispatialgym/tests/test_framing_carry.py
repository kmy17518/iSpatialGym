"""i7-framing (owner decisions 2026-11-08 5.1 / 5.2): the pre-grasp and touching pre-contact framing by the pre-place's fingertip rule
(``ISPATIALGYM_PRECONTACT_FRAME_RULE``), the lifted-carry framing check (``ISPATIALGYM_CARRY_FRAME_CHECK``) and the torso lift for the
carry (``ISPATIALGYM_CARRY_TORSO_LIFT``). Simulator-free; the carry cases replay under_small_03 300150 (i7pq2_us03: gripper_out_of_fov
in the via-above carry, u 122 of 128 at depth 0.50 m)."""

from types import SimpleNamespace

import pytest
import torch as th

from ispatialgym.robot import conventions as C
from ispatialgym.sampling import geometry as G
from ispatialgym.sampling.sim_utils import Reject
from ispatialgym.sampling.v2 import ik as IKm
from ispatialgym.sampling.v2 import program_check as PC
from ispatialgym.sampling.v2.tasks import iors_small as IS

CM = G.CameraModel(640, 480, 400.0, 400.0, 320.0, 240.0)
CAM = (th.tensor([0.0, 0.0, 1.5]), th.tensor([0.0, 0.0, 0.0, 1.0]))  # looking down -z
DOWN = th.tensor([1.0, 0.0, 0.0, 0.0])
HEAD = G.CameraModel.from_sensor(17.0, 40.0, 1080, 1080)  # the head camera (USD focal 17 mm, aperture 40 mm)

# under_small_03 300150 (i7pq2_us03, trunk_14634552): the manipulation stance and the grasp record
US03_STANCE = {"base_pose2d": [5.4898200035095215, 3.7469544410705566, -0.00255163110387004], "translate": 0.75, "tilt": -0.1879333257675171}
US03_QUAT = th.tensor([0.680962324142456, -0.4856150448322296, 0.340906023979187, -0.42924511432647705])
US03_LIFT = (th.tensor([6.337222099304199, 3.621920347213745, 0.853350043296814 + 0.03277376174926758]), US03_QUAT)
US03_PRE = (th.tensor([6.2480998039245605, 3.2073490619659424, 0.8961318731307983]), th.tensor([0.6808849573135376, -0.4856342077255249, 0.34088239073753357, -0.42936477065086365]))


def _head_cam(translate, tilt, st=US03_STANCE):
    x, y, yaw = st["base_pose2d"]
    return C.camera_pose(C.torso_q(translate, tilt), base_pose=(th.tensor([x, y, C.NOMINAL_BASE_Z]), G.quat_from_yaw(yaw)))


def _fake(**gen):
    cls = IS.IORSSmall
    ikc = SimpleNamespace(kit=SimpleNamespace(world=SimpleNamespace(camera_model=CM)))
    ikc.eef_in_frame = lambda pose, cam: IKm.IKChecker.eef_in_frame(ikc, pose, cam)
    s = SimpleNamespace(ikc=ikc, kit=SimpleNamespace(world=SimpleNamespace(camera_model=CM), stats={}), world=SimpleNamespace(camera_model=CM), gen=lambda k, d=None: gen.get(k, d),
                        hull={"target": th.tensor([[-0.05, -0.05, 0.0], [0.05, 0.05, 0.05]])})
    for m in ("precontact_frame_ok", "pre_place_frame_ok", "require_in_frame"):
        setattr(s, m, getattr(cls, m).__get__(s))
    return s


def _edge_pose():
    """An eef pose the fingertip rule frames and the +-6 cm eef box does not (near the image's right border)."""
    for x in [i * 0.005 for i in range(0, 200)]:
        pose = (th.tensor([x, 0.0, 0.9]), DOWN)
        box = IKm.IKChecker.eef_in_frame(SimpleNamespace(kit=SimpleNamespace(world=SimpleNamespace(camera_model=CM))), pose, CAM)
        if PC.fingertip_frame_margin(CM, CAM, pose)["inside"] and not box:
            return pose
    raise AssertionError("no pose between the two rules")


@pytest.mark.parametrize("what", IS.PRECONTACT_FRAME_POSES)
def test_precontact_rule_box_counts_the_disagreement(monkeypatch, what):
    monkeypatch.setenv(IS.PRECONTACT_FRAME_RULE_ENV, "box")
    assert IS.precontact_frame_rule() == "box"
    s, pose = _fake(), _edge_pose()
    far_obj = ("target", (th.tensor([3.0, 0.0, 0.5]), th.tensor([0.0, 0.0, 0.0, 1.0])))  # the object outside the image
    with pytest.raises(Reject):  # the box (the rule before) rejects; the shadow counters record the fingertip verdict
        s.require_in_frame(CAM, {what: pose}, objects={what: far_obj})
    k = "precontact_frame." + what.replace(" ", "_").replace("-", "_")
    assert s.kit.stats == {f"{k}.checked": 1, f"{k}.removed": 1, f"{k}.removed_object_out": 1}


def test_precontact_rule_fingertip_admits_and_keeps_contacts_on_the_box(monkeypatch):
    monkeypatch.delenv(IS.PRECONTACT_FRAME_RULE_ENV, raising=False)
    assert IS.precontact_frame_rule() == "fingertip"  # the default since the count (owner 5.2)
    s, pose = _fake(), _edge_pose()
    s.require_in_frame(CAM, {"pre-grasp": pose, "push pre-contact": pose, "brace pre-contact": pose})  # all pass the fingertip rule
    with pytest.raises(Reject) as e:  # the grasp (a contact pose) keeps the eef box
        s.require_in_frame(CAM, {"pre-grasp": pose, "grasp": pose})
    assert "grasp eef pose" in str(e.value)
    far = (th.tensor([2.0, 0.0, 0.9]), DOWN)
    with pytest.raises(Reject) as e:
        s.require_in_frame(CAM, {"pre-grasp": far})
    assert "pre-grasp fingertip (+2 cm slack)" in str(e.value)
    monkeypatch.setenv(IS.PRECONTACT_FRAME_RULE_ENV, "wide")
    with pytest.raises(ValueError):
        IS.precontact_frame_rule()


def test_oracle_protect_above_constant_matches_the_oracle():
    from ispatialgym.oracle.motions import MotionsConfig

    assert PC.ORACLE_PROTECT_ABOVE_M == MotionsConfig().protect_above_m


def test_via_above_poses_follow_the_oracle_legs():
    legs = PC.via_above_carry_poses(US03_LIFT, US03_PRE)
    z_top = max(float(US03_LIFT[0][2]), float(US03_PRE[0][2])) + PC.ORACLE_PROTECT_ABOVE_M
    assert set(legs) == {"rise", "cross", "descend"} and all(len(v) == PC.CARRY_FRAME_LEG_WAYPOINTS for v in legs.values())
    assert abs(float(legs["rise"][-1][0][2]) - z_top) < 1e-5 and abs(float(legs["cross"][-1][0][1]) - float(US03_PRE[0][1])) < 1e-5
    assert float(th.linalg.norm(legs["descend"][-1][0] - US03_PRE[0])) < 1e-5


def test_us03_300150_only_the_arm_lift_leaves_the_frame():
    cam = _head_cam(US03_STANCE["translate"], US03_STANCE["tilt"])
    out = PC.first_out_of_frame(HEAD, cam, PC.via_above_carry_poses(US03_LIFT, US03_PRE))
    assert out is not None and out["u"] > 1080 * 0.95 - 50  # the right border (the oracle logged u 122 of 128)
    straight = {"straight": list(C.line_waypoints(US03_LIFT, US03_PRE, 8, endpoints=True))}
    assert PC.first_out_of_frame(HEAD, cam, straight) is None


def test_torso_choice_and_image_invariance():
    t0, k0 = US03_STANCE["translate"], US03_STANCE["tilt"]
    ch = PC.carry_torso_choice(US03_STANCE["base_pose2d"], (t0, k0), [US03_LIFT[0], US03_PRE[0]])
    assert ch is not None and ch["tilt"] == pytest.approx(k0, abs=1e-4) and ch["translate"] < t0  # the translate first (no turn)
    assert min(ch["rise_m"]) >= PC.CARRY_TORSO_MIN_RISE_M and ch["shift_m"] <= PC.CARRY_TORSO_MAX_SHIFT_M and ch["rot_rad"] < 0.01
    Mt = PC.torso_move(US03_STANCE["base_pose2d"], (t0, k0), (ch["translate"], ch["tilt"]))
    cam0, cam1 = _head_cam(t0, k0), _head_cam(ch["translate"], ch["tilt"])
    for p in C.line_waypoints(US03_LIFT, US03_PRE, 6, endpoints=True):
        a, b = PC.fingertip_frame_margin(HEAD, cam0, p), PC.fingertip_frame_margin(HEAD, cam1, G.compose(Mt, p))
        assert abs(a["u"] - b["u"]) < 0.5 and abs(a["v"] - b["v"]) < 0.5  # the hand moves with the camera
    # the displaced straight leg (what the oracle tracks) is framed, the displaced via-above's rise is met by the torso
    leg = {"torso_leg": [G.compose(Mt, p) for p in C.line_waypoints(US03_LIFT, US03_PRE, 8, endpoints=True)]}
    assert PC.first_out_of_frame(HEAD, cam1, leg) is None


def test_torso_choice_none_when_the_torso_cannot_rise_enough():
    # trunk translate 0 (upright already): only the tilt is left, <= 0.1 rad lifts the hand ~7 cm < 0.12 m
    assert PC.carry_torso_choice(US03_STANCE["base_pose2d"], (0.0, -0.611), [US03_LIFT[0], US03_PRE[0]]) is None


def test_switches_default_off(monkeypatch):
    for env in (PC.CARRY_FRAME_CHECK_ENV, PC.CARRY_TORSO_LIFT_ENV):
        monkeypatch.delenv(env, raising=False)
    assert not PC.carry_frame_check_on() and not PC.carry_torso_lift_on()
    monkeypatch.setenv(PC.CARRY_FRAME_CHECK_ENV, "1")
    assert PC.carry_frame_check_on()


def test_oracle_reads_carry_torso_only_with_its_switch(monkeypatch):
    from ispatialgym.oracle import execute as EX

    assert EX.CARRY_TORSO_LIFT_ENV == PC.CARRY_TORSO_LIFT_ENV
    g = SimpleNamespace(object="plate_0", arm="right", extra={"carry_torso": {"translate": 0.5, "tilt": -0.19}})
    ctx = {"sidecar": SimpleNamespace(grasps=[g])}
    act = SimpleNamespace(object="plate_0", arm="right")
    monkeypatch.delenv(EX.CARRY_TORSO_LIFT_ENV, raising=False)
    assert EX._carry_torso_record(act, ctx) is None
    monkeypatch.setenv(EX.CARRY_TORSO_LIFT_ENV, "1")
    assert EX._carry_torso_record(act, ctx) == (0.5, -0.19)
    g.extra = {}
    assert EX._carry_torso_record(act, ctx) is None


# ------------------------------------------------------------------ 5d.3: the middle option (ISPATIALGYM_CARRY_BAND)

def _carry_fake(margin_m, torso=None):
    """A sampler stand-in for IORSSmall.carry_frame_check at 300150's stance (the via-above out of frame, the straight carry framed)."""

    cam = _head_cam(US03_STANCE["translate"], US03_STANCE["tilt"])
    s = SimpleNamespace(kit=SimpleNamespace(world=SimpleNamespace(camera_model=HEAD), stats={}), camera_pose_at=lambda st: cam,
                        carry_nominal_margin=lambda g, a, b: {"margin_m": margin_m},
                        carry_torso_lift=lambda *a: (torso, "") if torso is not None else (None, "no_trunk"))
    s.carry_frame_check = IS.IORSSmall.carry_frame_check.__get__(s)
    st = SimpleNamespace()
    return s, st, SimpleNamespace(summary={}), SimpleNamespace(extras={})


def _clear_switches(monkeypatch):
    for env in (PC.CARRY_FRAME_CHECK_ENV, PC.CARRY_TORSO_LIFT_ENV, PC.CARRY_BAND_ENV):
        monkeypatch.delenv(env, raising=False)


def test_band_switch_off_is_today(monkeypatch):
    _clear_switches(monkeypatch)
    assert not PC.carry_band_on() and not PC.carry_frame_check_on() and not PC.carry_torso_lift_on()


def test_band_edges():
    assert PC.carry_band(None) == "above" and PC.carry_band(float("inf")) == "above"
    assert PC.carry_band(PC.CARRY_BAND_ABOVE_M + 0.001) == "above"
    assert PC.carry_band(PC.CARRY_BAND_BELOW_M - 0.001) == "below"
    assert PC.carry_band(0.5 * (PC.CARRY_BAND_BELOW_M + PC.CARRY_BAND_ABOVE_M)) == "near"


def test_band_far_above_is_never_rejected(monkeypatch):
    _clear_switches(monkeypatch)
    monkeypatch.setenv(PC.CARRY_BAND_ENV, "1")
    s, st, draft, g = _carry_fake(PC.CARRY_BAND_ABOVE_M + 0.05)
    s.carry_frame_check(draft, g, st, st, US03_LIFT, US03_PRE, None)  # out of frame, but the lift will not fire: no check
    assert s.kit.stats == {"framing.carry.band.above": 1}


def test_band_far_below_out_of_frame_is_rejected(monkeypatch):
    _clear_switches(monkeypatch)
    monkeypatch.setenv(PC.CARRY_BAND_ENV, "1")
    s, st, draft, g = _carry_fake(PC.CARRY_BAND_BELOW_M - 0.01, torso={"translate": 0.5, "tilt": -0.19})
    with pytest.raises(Reject) as e:  # below: the check, no torso rescue
        s.carry_frame_check(draft, g, st, st, US03_LIFT, US03_PRE, None)
    assert "band below" in str(e.value) and s.kit.stats["framing.carry.band.rejected"] == 1 and "carry_torso" not in g.extras


def test_band_near_takes_the_torso_lift_else_the_check(monkeypatch):
    _clear_switches(monkeypatch)
    monkeypatch.setenv(PC.CARRY_BAND_ENV, "1")
    mid = 0.5 * (PC.CARRY_BAND_BELOW_M + PC.CARRY_BAND_ABOVE_M)
    s, st, draft, g = _carry_fake(mid, torso={"translate": 0.5, "tilt": -0.19})
    s.carry_frame_check(draft, g, st, st, US03_LIFT, US03_PRE, None)
    assert g.extras["carry_torso"]["translate"] == 0.5 and s.kit.stats["framing.carry.band.torso_lift_used"] == 1
    s, st, draft, g = _carry_fake(mid, torso=None)
    with pytest.raises(Reject):
        s.carry_frame_check(draft, g, st, st, US03_LIFT, US03_PRE, None)
    assert s.kit.stats["framing.carry.torso_lift_failed.no_trunk"] == 1 and s.kit.stats["framing.carry.band.rejected"] == 1


def test_torso_only_mode_never_rejects(monkeypatch):
    _clear_switches(monkeypatch)
    monkeypatch.setenv(PC.CARRY_TORSO_LIFT_ENV, "1")
    s, st, draft, g = _carry_fake(None, torso=None)
    s.carry_frame_check(draft, g, st, st, US03_LIFT, US03_PRE, None)
    assert s.kit.stats["framing.carry.accepted_out"] == 1 and "framing.carry.band.above" not in s.kit.stats


def test_carry_margin_constants_and_geometry():
    import numpy as np

    from ispatialgym.oracle import protection as PROT
    from ispatialgym.sampling.v2 import carry_margin as CMg

    assert CMg.INFLATION_M == PROT.INFLATION_M and CMg.ENDPOINT_SLACK_M == PROT.ENDPOINT_SLACK_M
    down = [1.0, 0.0, 0.0, 0.0]  # x y z w: the approach axis down
    eio = ([0.0, 0.0, 0.05], [1.0, 0.0, 0.0, 0.0])
    obj = (np.zeros(3), np.array([0.05, 0.05, 0.01]))
    box = CMg.world_box(([0.5, 0.0, 0.0], [0, 0, 0, 1]), (np.zeros(3), np.array([0.05, 0.05, 0.05])))
    high = CMg.carry_protect_margin([([0.0, 0.0, 0.5], down), ([1.0, 0.0, 0.5], down)], eio, obj, [box])
    low = CMg.carry_protect_margin([([0.0, 0.0, 0.25], down), ([1.0, 0.0, 0.25], down)], eio, obj, [box])  # the hand box passes 2 cm off
    assert high["margin_m"] > 0.0 and low["margin_m"] < high["margin_m"]
    through = CMg.carry_protect_margin([([0.0, 0.0, 0.08], down), ([1.0, 0.0, 0.08], down)], eio, obj, [box])  # the object (eef - 5 cm) at 3 cm
    assert through["margin_m"] < 0.0 and through["interior_min_m"] < 0.0  # passes through the box: the oracle fires
    assert CMg.carry_protect_margin([([0.0, 0.0, 0.15], down), ([1.0, 0.0, 0.15], down)], eio, obj, [])["margin_m"] == float("inf")
