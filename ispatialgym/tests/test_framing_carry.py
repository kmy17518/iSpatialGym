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
def test_precontact_rule_box_default_counts_the_disagreement(monkeypatch, what):
    monkeypatch.delenv(IS.PRECONTACT_FRAME_RULE_ENV, raising=False)
    assert IS.precontact_frame_rule() == "box"
    s, pose = _fake(), _edge_pose()
    far_obj = ("target", (th.tensor([3.0, 0.0, 0.5]), th.tensor([0.0, 0.0, 0.0, 1.0])))  # the object outside the image
    with pytest.raises(Reject):  # the box (the rule before) rejects; the shadow counters record the fingertip verdict
        s.require_in_frame(CAM, {what: pose}, objects={what: far_obj})
    k = "precontact_frame." + what.replace(" ", "_").replace("-", "_")
    assert s.kit.stats == {f"{k}.checked": 1, f"{k}.removed": 1, f"{k}.removed_object_out": 1}


def test_precontact_rule_fingertip_admits_and_keeps_contacts_on_the_box(monkeypatch):
    monkeypatch.setenv(IS.PRECONTACT_FRAME_RULE_ENV, "fingertip")
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
