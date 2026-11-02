"""The ``arm_stutter`` QA check (owner 2026-10-13, mental_rotation-walker: "the robot seems to stutter ... make the QA check catch
it"): repeated jolts of an arm's commanded joint motion inside one action are a ``robot_stutter`` verdict of demos and replays; a
smooth leg, its start and stop, a single kink where two legs join and a one-step speed blip are not."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch as th

from ispatialgym.oracle import r1pro_model as M
from ispatialgym.oracle.executor import ExecutorConfig
from ispatialgym.oracle.programs import expand
from ispatialgym.oracle.qa import ArmStutter, StutterConfig, find_stutters, stutter_report_hdf5
from ispatialgym.tests.test_v2_oracle import FakeRobot, _place_goal, iors_world, make_executor, success_eval

DT = 1.0 / 30.0


def trace(velocities, q0=None):
    """Joint positions (T + 1, n) integrating per-step velocities (T, n) in rad/s."""
    v = np.asarray(velocities, dtype=np.float64)
    q = np.zeros((len(v) + 1, v.shape[1])) if q0 is None else np.tile(np.asarray(q0, dtype=np.float64), (len(v) + 1, 1))
    q[1:] += np.cumsum(v * DT, axis=0)
    return q


def lurch(n_before=10, n_after=10):
    """The walker's lurch (steps 36-40 of its orbit): joint 5 at 0.24 -> 1.41 -> 0.77 -> 0.77 -> 1.43 rad/s, joint 7 at -1.1 rad/s throughout."""
    j5 = [0.24] * n_before + [1.41, 0.77, 0.77] + [1.43] * n_after
    v = np.zeros((len(j5), 7))
    v[:, 4] = j5
    v[:, 6] = -1.1
    return v


def test_repeated_jolts_while_moving_are_a_stutter():
    q = trace(lurch())
    res = find_stutters(q, np.zeros(len(q), dtype=int))
    assert [t for t, _ in res["jolts"]] == [10, 11, 13]
    assert [dv for _, dv in res["jolts"]] == pytest.approx([1.17, 0.64, 0.66], abs=1e-3)
    (s,) = res["stutters"]
    assert (s["first"], s["last"], s["n_jolts"]) == (10, 13, 3) and s["max_jolt_rad_s"] == pytest.approx(1.17, abs=1e-3)
    # a reversal at speed counts like any other change of the velocity vector
    v = np.zeros((12, 7))
    v[:, 2] = [0.9] * 4 + [-0.33, 0.9, -0.33] + [0.9] * 5
    assert len(find_stutters(trace(v), np.zeros(13, dtype=int))["stutters"]) == 1


def test_a_smooth_leg_its_start_and_stop_a_single_kink_and_a_speed_blip_are_not():
    # a leg at constant joint velocity that starts from and stops at rest (the oracle streams legs without a ramp)
    v = np.zeros((40, 7))
    v[5:35, 2], v[5:35, 6] = 0.9, -1.0
    assert find_stutters(trace(v), np.zeros(41, dtype=int)) == {"jolts": [], "stutters": []}
    # two legs joined without a stop: the velocity vector turns once (a kink), 1 jolt
    v = np.zeros((30, 7))
    v[:15, 0], v[15:, 3] = 0.8, 0.8
    res = find_stutters(trace(v), np.zeros(31, dtype=int))
    assert len(res["jolts"]) == 1 and res["stutters"] == []
    # the per-waypoint step rounding: one step at twice the speed between steps at the leg's speed, 2 jolts
    v = np.full((20, 7), 0.0)
    v[:, 1] = 0.6
    v[10, 1] = 1.3
    res = find_stutters(trace(v), np.zeros(21, dtype=int))
    assert len(res["jolts"]) == 2 and res["stutters"] == []
    # jolts below the threshold, or while the arm creeps below move_rad_s, are not jolts
    slow = lurch() * 0.2
    assert find_stutters(trace(slow), np.zeros(len(slow) + 1, dtype=int))["jolts"] == []


def test_jolts_of_different_motions_or_outside_actions_do_not_add_up():
    q = trace(lurch())
    seg = np.zeros(len(q), dtype=int)
    seg[12:] = 1  # the second and third jolt belong to the next action attempt
    assert find_stutters(q, seg)["stutters"] == []
    assert find_stutters(q, np.full(len(q), -1))["jolts"] == []  # the epilogue / settle steps are no action
    # the window and the count are the config's
    assert find_stutters(q, np.zeros(len(q), dtype=int), StutterConfig(min_jolts=4))["stutters"] == []
    assert find_stutters(q, np.zeros(len(q), dtype=int), StutterConfig(window_steps=3))["stutters"] == []


def _command(robot, arm_q):
    a = th.zeros(M.ACTION_DIM)
    a[robot.controller_action_idx["arm_right"]] = th.as_tensor(arm_q, dtype=th.float32)
    a[robot.controller_action_idx["arm_left"]] = robot.q[robot.arm_control_idx["left"]]
    return a


def test_arm_stutter_judges_the_commanded_arm_targets_and_forgets_a_rolled_back_attempt():
    r = FakeRobot()
    st = ArmStutter(r)
    q = trace(lurch(), r.q[r.arm_control_idx["right"]].numpy())
    for i, qi in enumerate(q):
        st.observe(100 + i, _command(r, qi), 3000, "reorient")
    rep = st.report()
    assert rep["flagged"] and rep["worst"]["arm"] == "right" and rep["worst"]["action"] == "reorient"
    assert (rep["worst"]["first_step"], rep["worst"]["last_step"], rep["worst"]["n_jolts"]) == (110, 113, 3)
    assert rep["thresholds"] == {"jolt_rad_s": 0.6, "move_rad_s": 0.3, "window_steps": 6, "min_jolts": 3}
    st.trim(111)  # the attempt rolled back after step 111: its later steps leave the record
    assert not st.report()["flagged"]
    assert ArmStutter(r).report()["flagged"] is False


def _write_demo(path, arm_right, action_index, attempt):
    import h5py

    order = list(M.CONTROLLER_ORDER)
    dims = {k: int(M.CONTROLLER_DIMS[k]) for k in order}
    off = sum(dims[k] for k in order[: order.index("arm_right")])
    act = np.zeros((len(arm_right), sum(dims.values())), dtype=np.float32)
    act[:, off:off + 7] = arm_right
    with h5py.File(path, "w") as f:
        d = f.create_group("data")
        d.attrs["controller_order"] = json.dumps(order)
        d.attrs["controller_dims"] = json.dumps(dims)
        g = d.create_group("demo_0")
        g.attrs["isg_annotations"] = json.dumps({"actions": [{"index": 0, "name": "reorient"}]})
        g.create_dataset("action", data=act)
        g.create_dataset("isg/action_index", data=np.asarray(action_index, dtype=np.int64))
        g.create_dataset("isg/attempt", data=np.asarray(attempt, dtype=np.int64))
        g.create_dataset("isg/step", data=np.arange(1, len(arm_right) + 1, dtype=np.int64))


def test_stutter_report_of_a_recorded_demo(tmp_path):
    q = trace(lurch())
    n = len(q)
    settle = np.repeat(q[-1:], 20, axis=0)
    _write_demo(tmp_path / "stutter.hdf5", np.vstack([q, settle]), [0] * n + [-1] * 20, [0] * (n + 20))
    rep = stutter_report_hdf5(str(tmp_path / "stutter.hdf5"))
    assert rep["flagged"] and rep["worst"]["action"] == "reorient" and (rep["worst"]["first_step"], rep["worst"]["last_step"]) == (11, 14)
    v = np.zeros((40, 7))
    v[5:35, 2] = 0.9
    smooth = trace(v)
    _write_demo(tmp_path / "smooth.hdf5", smooth, [0] * len(smooth), [0] * len(smooth))
    assert not stutter_report_hdf5(str(tmp_path / "smooth.hdf5"))["flagged"]


@pytest.mark.parametrize("gated", [True, False])
def test_arm_stutter_gate_discards_a_demo_whose_arm_stutters(tmp_path, gated, monkeypatch):
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path, exec_cfg=ExecutorConfig(settle_min_steps=50, settle_fraction=0.2,
                                                                                        demo_gating=("arm_stutter",) if gated else ()))
    real = ArmStutter.observe
    lurch_q = trace(lurch(2, 4))
    seen = {"n": 0, "q0": None}

    def observe(self, step, action, segment, action_name=None):
        if action_name == "place" and seen["n"] < len(lurch_q):
            # the place's commanded right arm replaced by the walker's lurch (the fake's own motion is smooth)
            a = th.as_tensor(action).clone()
            idx = self.idx_action["right"]
            if seen["q0"] is None:
                seen["q0"] = a[idx].clone()
            a[idx] = seen["q0"] + th.as_tensor(lurch_q[seen["n"]], dtype=th.float32)
            seen["n"] += 1
            action = a
        real(self, step, action, segment, action_name)

    monkeypatch.setattr(ArmStutter, "observe", observe)
    result = ex.run_program(expand(sc.variant, sc))
    assert result["arm_stutter"]["flagged"] and result["arm_stutter"]["worst"]["action"] == "place"
    if gated:
        assert not result["success"] and result["reason"] == "robot_stutter"
        assert result["qa"]["stats"]["by_category"].get("robot_stutter") == 1
    else:
        assert result["success"], result["verdict"]


def test_an_unremarkable_program_is_not_a_stutter(tmp_path):
    sc, env = iors_world()
    env.evaluate = success_eval("apple.n.01_1", _place_goal(sc), tol=0.08)
    ex, motions, backend, rec = make_executor(env, sc, tmp_path, exec_cfg=ExecutorConfig(settle_min_steps=50, settle_fraction=0.2, demo_gating=("arm_stutter",)))
    result = ex.run_program(expand(sc.variant, sc))
    assert result["success"], result["verdict"]
    assert result["arm_stutter"]["flagged"] is False and result["arm_stutter"]["stutters"] == []
