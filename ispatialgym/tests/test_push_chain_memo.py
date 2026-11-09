"""i7-stancememo r4 (lead decision #70): IORSLarge._push_under_chain memoizes the chain per attempt on the exact inputs (one under_large_03
attempt solved 85 chains, 43 distinct). Same inputs -> the stored result (a copy); any input differing in a bit -> a solve; the memo is
cleared per attempt; ISPATIALGYM_PUSH_CHAIN_MEMO=0 solves every call."""
from types import SimpleNamespace

import torch as th

from ispatialgym.sampling.v2.stances import Stance
from ispatialgym.sampling.v2.tasks import iors_large as IL


def _s():
    s = IL.IORSLarge.__new__(IL.IORSLarge)
    s.kit = SimpleNamespace(stats={})
    s.seg_arm = "right"
    s.__class__ = type("IL", (IL.IORSLarge,), {"push_hand": "pitched"})
    calls = []

    def solve(place, quat, drop_xy, goal_xy, d2):
        calls.append(float(quat[2]))
        return None if float(quat[2]) > 0.9 else {"pitch_deg": 60.0, "start": [th.tensor([1.0, 2.0, 0.1])], "q": float(quat[2])}

    s._push_under_chain_solve = solve
    return s, calls


def test_same_inputs_return_the_stored_chain(monkeypatch):
    monkeypatch.delenv("ISPATIALGYM_PUSH_CHAIN_MEMO", raising=False)
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
    monkeypatch.delenv("ISPATIALGYM_PUSH_CHAIN_MEMO")
    s._push_under_chain(st, q, th.tensor([1.0, 1.0]), th.tensor([1.3, 1.0]), [1.0, 0.0])
    s._push_chain_memo = {}  # what IORSLarge.sample does at every attempt
    s._push_under_chain(st, q, th.tensor([1.0, 1.0]), th.tensor([1.3, 1.0]), [1.0, 0.0])
    assert len(calls) == 4
    import inspect
    assert "self._push_chain_memo = {}" in inspect.getsource(IL.IORSLarge.sample)
