"""i7-yield2 r2 (lead decision #65): the gripper close wait (60 + 5 still steps) outlasted the QA pause limit (50), so a closure that grips
nothing ended in a hard [pause] instead of the executor's retried failed_grasp (int2_7 disembedding_10 300019 / 300020, disembedding_01
300021). ``ISPATIALGYM_GRIP_WAIT_NO_PAUSE`` (default on since r2; =0 off) emits the close-wait steps as settle steps (not counted by the pause / stalled rules)."""
from __future__ import annotations

from types import SimpleNamespace

import torch as th

from ispatialgym.oracle import motions as MO
from ispatialgym.oracle.qa import QA


class _State:
    def __init__(self):
        self.gripper_cmd = {}
        self.closed = None

    def stop_base(self):
        pass

    def set_gripper(self, arm, close):
        self.closed = close

    def action(self):
        return th.zeros(23)


def _motions(pads_touch=False, attach_after=None):
    m = MO.Motions.__new__(MO.Motions)
    m.state = _State()
    q = th.zeros(28)
    q[[26, 27]] = 0.015
    m.robot = SimpleNamespace(gripper_control_idx={"right": th.tensor([26, 27])}, get_joint_positions=lambda: q,
                              get_joint_velocities=lambda: th.full((28,), 0.12))  # the reported finger jitter: never below 1e-3
    m._gripper_limits = lambda arm: (0.0, 0.05)
    m.gripper_intent_closed = {}
    m.cfg = SimpleNamespace(gripper_contact_squeeze_m=0.001, gripper_steps=60)
    m.gripper_command = lambda arm, q_: float(q_)
    m.release_events = []
    m.last_steps = 0
    m.settling = False
    m.n_emitted = 0

    def held(arm):
        return "fork" if attach_after is not None and m.n_emitted >= attach_after else None

    m.held_object = held
    m.object_between_pads = lambda arm: pads_touch
    orig = MO.Motions._emit

    def emit(self=m, settling=False):
        m.n_emitted += 1
        out = orig(m, settling)
        m.flags.append(m.settling)
        return out

    m.flags = []
    m._emit = emit
    return m


def _run(m, **kw):
    for _ in m.gripper("right", close=True, **kw):
        pass
    return m


def test_default_on_and_off_restores_the_counted_wait(monkeypatch):
    monkeypatch.delenv("ISPATIALGYM_GRIP_WAIT_NO_PAUSE", raising=False)
    assert MO.grip_wait_no_pause() is True  # default on (i7-yield2 r2)
    monkeypatch.setenv("ISPATIALGYM_GRIP_WAIT_NO_PAUSE", "0")
    m = _run(_motions(), rate=0.00225)
    ramp = 7  # 15 mm -> 0 at 2.25 mm per step
    assert m.flags[:ramp] == [False] * ramp
    wait = m.flags[ramp:-5]
    assert len(wait) == 60 and not any(wait)  # 60 counted still steps > the pause limit 50
    assert m.flags[-5:] == [True] * 5  # the hold was a settle step before too
    assert not any(e["event"] == "close_wait_exempt" for e in m.release_events)


def test_on_the_close_wait_is_a_settle_wait_and_recorded(monkeypatch):
    monkeypatch.setenv("ISPATIALGYM_GRIP_WAIT_NO_PAUSE", "1")
    m = _run(_motions(), rate=0.00225)
    assert m.flags[:7] == [False] * 7  # the ramp moves the fingers: counted as before
    assert all(m.flags[7:]) and len(m.flags[7:]) == 65
    assert m.release_events[-1] == {"arm": "right", "event": "close_wait_exempt", "steps": 60, "held": False}


def test_on_the_contact_hold_wait_is_exempt_too_and_attach_ends_it(monkeypatch):
    monkeypatch.setenv("ISPATIALGYM_GRIP_WAIT_NO_PAUSE", "1")
    m = _run(_motions(pads_touch=True, attach_after=12), rate=0.00225)
    ev = [e["event"] for e in m.release_events]
    assert ev == ["contact_hold"] and m.release_events[-1]["close_wait_exempt_steps"] >= 1  # recorded on the closure's contact hold
    assert m.flags[0] is False and all(m.flags[1:])


def test_opening_and_other_waits_still_count(monkeypatch):
    monkeypatch.setenv("ISPATIALGYM_GRIP_WAIT_NO_PAUSE", "1")
    m = _motions()
    for _ in m.gripper("right", close=False):
        pass
    assert not any(m.flags[:-5]) and all(m.flags[-5:])


def test_qa_pause_skips_settle_steps_and_still_fires_on_a_real_stall():
    qa = QA.__new__(QA)
    qa.t = SimpleNamespace(step_dt=1 / 30, pause_vel=0.001, pause_steps=50)
    qa._q_prev, qa._pause_streak, qa._pause_start, qa.max_pause, qa.pause_runs = th.zeros(3), 0, None, 0, []
    hard = []
    for k in range(80):  # a 65-step close wait seen as settle steps: _close_pause, no streak
        qa._close_pause(k - 1)
    assert qa._pause_streak == 0 and hard == []
    for k in range(80, 132):  # a stall elsewhere (counted steps)
        qa._pause(k, th.zeros(3), hard)
    assert [h[0] for h in hard] == ["pause"]
