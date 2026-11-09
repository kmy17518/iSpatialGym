"""i7-stancememo r4 (lead decision #70): IORSLarge._push_under_chain memoizes the chain per attempt on the exact inputs (one under_large_03
attempt solved 85 chains, 43 distinct). Same inputs -> the stored result (a copy); any input differing in a bit -> a solve; the memo is
cleared per attempt; default OFF (lead #70b), ISPATIALGYM_PUSH_CHAIN_MEMO=1 turns it on."""
from types import SimpleNamespace

import torch as th

from ispatialgym.sampling.v2.stances import Stance
from ispatialgym.sampling.v2.tasks import iors_large as IL


def _s():
    s = IL.IORSLarge.__new__(IL.IORSLarge)
    s.kit = SimpleNamespace(stats={})
    s.gp = SimpleNamespace(variant="stub")
    s.seg_arm = "right"
    s.__class__ = type("IL", (IL.IORSLarge,), {"push_hand": "pitched"})
    calls = []

    def solve(place, quat, drop_xy, goal_xy, d2):
        calls.append(float(quat[2]))
        return None if float(quat[2]) > 0.9 else {"pitch_deg": 60.0, "start": [th.tensor([1.0, 2.0, 0.1])], "q": float(quat[2])}

    s._push_under_chain_solve = solve
    s._push_chain_memo = {}  # what IORSLarge.sample sets at every attempt (no memo outside an attempt)
    return s, calls


def test_same_inputs_return_the_stored_chain(monkeypatch):
    monkeypatch.setenv("ISPATIALGYM_PUSH_CHAIN_MEMO", "1")  # default off (lead #70b)
    s, calls = _s()
    st = Stance("push", (1.0, 2.0, 0.5), 2.0, 0.0)
    q = th.tensor([0.0, 0.0, 0.3, 0.9539392])
    a = s._push_under_chain(st, q, th.tensor([1.0, 1.0]), th.tensor([1.3, 1.0]), [1.0, 0.0])
    b = s._push_under_chain(st, q.clone(), th.tensor([1.0, 1.0]), th.tensor([1.3, 1.0]), (1.0, 0.0))
    assert calls == [q[2].item()] and a is not b and a["start"][0] is not b["start"][0]
    assert a["pitch_deg"] == b["pitch_deg"] and a["q"] == b["q"] and th.equal(a["start"][0], b["start"][0])
    b["pitch_deg"] = 0.0  # a caller's change never reaches the memo
    assert s._push_under_chain(st, q, th.tensor([1.0, 1.0]), th.tensor([1.3, 1.0]), [1.0, 0.0])["pitch_deg"] == 60.0
    q2 = q.clone()
    q2[2] = th.nextafter(q2[2], th.tensor(1.0))  # one ulp: another key
    s._push_under_chain(st, q2, th.tensor([1.0, 1.0]), th.tensor([1.3, 1.0]), [1.0, 0.0])
    s._push_under_chain(st.with_trunk(1.75, 0.0, "push"), q, th.tensor([1.0, 1.0]), th.tensor([1.3, 1.0]), [1.0, 0.0])
    qn = th.tensor([0.0, 0.0, 0.95, 0.3122499])
    assert s._push_under_chain(st, qn, th.tensor([1.0, 1.0]), th.tensor([1.3, 1.0]), [1.0, 0.0]) is None
    assert s._push_under_chain(st, qn, th.tensor([1.0, 1.0]), th.tensor([1.3, 1.0]), [1.0, 0.0]) is None  # a failure is remembered too
    assert len(calls) == 4 and s.kit.stats["push_chain.memo_hit"] == 3 and s.kit.stats["push_chain.solved"] == 4


def test_memo_off_and_per_attempt(monkeypatch):
    s, calls = _s()
    st = Stance("push", (1.0, 2.0, 0.5), 2.0, 0.0)
    q = th.tensor([0.0, 0.0, 0.3, 0.9539392])
    monkeypatch.setenv("ISPATIALGYM_PUSH_CHAIN_MEMO", "0")
    for _ in range(2):
        s._push_under_chain(st, q, th.tensor([1.0, 1.0]), th.tensor([1.3, 1.0]), [1.0, 0.0])
    assert len(calls) == 2
    monkeypatch.setenv("ISPATIALGYM_PUSH_CHAIN_MEMO", "1")
    s._push_under_chain(st, q, th.tensor([1.0, 1.0]), th.tensor([1.3, 1.0]), [1.0, 0.0])
    s._push_chain_memo = {}  # what IORSLarge.sample does at every attempt
    s._push_under_chain(st, q, th.tensor([1.0, 1.0]), th.tensor([1.3, 1.0]), [1.0, 0.0])
    assert len(calls) == 4
    import inspect
    assert "self._push_chain_memo = {}" in inspect.getsource(IL.IORSLarge.sample)
    del s._push_chain_memo  # outside an attempt: no memo
    s._push_under_chain(st, q, th.tensor([1.0, 1.0]), th.tensor([1.3, 1.0]), [1.0, 0.0])
    s._push_under_chain(st, q, th.tensor([1.0, 1.0]), th.tensor([1.3, 1.0]), [1.0, 0.0])
    assert len(calls) == 6


def _pose(x, yaw=0.0):
    import math
    return (th.tensor([x, 0.0, 0.1]), th.tensor([0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)], dtype=th.float64))


def test_quantized_memo_off_shadow_on(monkeypatch):
    """Lead #70c: ISPATIALGYM_CHAIN_MEMO off (default) / shadow (counts and compares, every call solves) / on (a key hit returns the stored
    chain). The key is quantized to 1e-4: inputs 1e-6 apart share it."""
    monkeypatch.delenv("ISPATIALGYM_PUSH_CHAIN_MEMO", raising=False)
    calls = []

    def make():
        s, _ = _s()
        x = {"v": 0.5}

        def solve(place, quat, drop_xy, goal_xy, d2):
            calls.append(1)
            x["v"] += 0.0004  # each fresh solve 0.4 mm further: the shadow sees the gap
            return {"pre": _pose(x["v"]), "start": _pose(x["v"]), "mid": _pose(x["v"]), "goal": _pose(x["v"] + 0.3), "pitch_deg": 60.0,
                    "record": {"wrist_turn": "lateral", "contact_height_m": 0.086}}
        s._push_under_chain_solve = solve
        return s

    st = Stance("push", (1.0, 2.0, 0.5), 2.0, 0.0)
    q1 = th.tensor([0.0, 0.0, 0.3, 0.9539392])
    q2 = q1 + th.tensor([0.0, 0.0, 1e-6, 0.0])  # the same quantum
    args = (th.tensor([1.0, 1.0]), th.tensor([1.3, 1.0]), [1.0, 0.0])
    assert IL.chain_memo_mode() == "off"
    s = make()
    s._push_under_chain(st, q1, *args)
    s._push_under_chain(st, q2, *args)
    assert len(calls) == 2 and not any(k.startswith("chain_memo") for k in s.kit.stats)
    calls.clear()
    monkeypatch.setenv("ISPATIALGYM_CHAIN_MEMO", "shadow")
    s = make()
    a = s._push_under_chain(st, q1, *args)
    b = s._push_under_chain(st, q2, *args)
    assert len(calls) == 2 and float(b["start"][0][0]) != float(a["start"][0][0])  # shadow: the fresh result
    assert s.kit.stats["chain_memo.hit"] == 1 and s.kit.stats["chain_memo.miss"] == 1 and "chain_memo.shadow_mismatch" not in s.kit.stats  # 0.4 mm
    calls.clear()
    monkeypatch.setenv("ISPATIALGYM_CHAIN_MEMO", "on")
    s = make()
    a = s._push_under_chain(st, q1, *args)
    b = s._push_under_chain(st, q2, *args)
    assert len(calls) == 1 and float(b["start"][0][0]) == float(a["start"][0][0]) and b is not a
    s._push_under_chain(st, q1 + th.tensor([0.0, 0.0, 2e-4, 0.0]), *args)  # another quantum
    assert len(calls) == 2


def test_chain_gap_flags():
    a = {"start": _pose(0.5), "goal": _pose(0.8), "pitch_deg": 60.0, "record": {"wrist_turn": "lateral", "contact_height_m": 0.086}}
    b = {"start": _pose(0.502, 0.002), "goal": _pose(0.8), "pitch_deg": 65.0, "record": {"wrist_turn": "plane", "contact_height_m": 0.087}}
    g = IL.chain_gap(a, b)
    assert abs(g["pos_mm"] - 2.0) < 1e-3 and abs(g["rot_mrad"] - 2.0) < 1e-2 and g["pitch"] and g["form"] and g["contact"]
    assert IL.chain_gap(a, None) == {"feasible": True} and IL.chain_gap(None, None) == {}
