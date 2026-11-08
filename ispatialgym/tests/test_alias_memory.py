"""i7-memory (owner 2026-11-08 01:45 UTC, part C): the per-alias outcome memory - attribution, posterior, back-off, decay / reset,
exploration floor, shard merge, eval-generation exclusion, the prior order kept."""
import json
import os
import types

import numpy as np
import pytest

from ispatialgym.sampling import alias_memory as AM

PROG_PLACE = ["navigate[]()", "transition[]()", "push[right](b)", "retreat[right](b)", "transition[]()", "grasp[right](b)", "lift[right](b)",
              "transition[](b)", "navigate[](b)", "transition[](b)", "place[right](b, c)", "retreat[right](b)"]


def row(alias="a", ok=True, stage=None, place=None, pick=None, ctx=None, t="2026-11-07T00:00:00", ftype=None, excluded=None, tag="t1"):
    r = {"alias": alias, "outcome": "success" if ok else "failure", "stage": stage, "excluded": excluded, "failure_type": ftype, "tag": tag,
         "place": place, "pick": pick, "ctx": ctx or {"snapshot": "trunk_a", "spec": "s1", "keys": "k1"}, "finished": t, "inst": "x", "jobid": "1"}
    r["place_obs"], r["pick_obs"] = AM.attribute(r)
    return r


def P(arm="right", tr=0.25, ti=-0.6, so=0.85, yaw=0):
    return {"arm": arm, "stance": "placement", "stand_off": so, "translate": tr, "tilt": ti, "yaw_slot": yaw}


def K(arm="right", tr=0.25, ti=-0.8, so=0.55, fam="side_2"):
    return {"arm": arm, "stance": "pick", "stand_off": so, "translate": tr, "tilt": ti, "family": fam}


# ------------------------------------------------------------------------------------------------------------- attribution

def test_attribution_by_stage():
    assert AM.attribute(row(ok=True, place=P(), pick=K()))[:2] == (1, 1)
    assert AM.attribute(row(ok=False, stage="pick", place=P(), pick=K())) == (None, 0)  # the place was never exercised
    assert AM.attribute(row(ok=False, stage="place", place=P(), pick=K())) == (0, 1)    # the pick passed its stage
    assert AM.attribute(row(ok=False, stage="carry", place=P(), pick=K())) == (0, 1)
    assert AM.attribute(row(ok=False, stage="outcome", place=P(), pick=K())) == (0, 1)
    assert AM.attribute(row(ok=False, stage="outcome", place=None, pick=K())) == (None, 0)  # a task without a place
    assert AM.attribute(row(ok=False, stage="manipulation", place=None, pick=K())) == (None, 0)
    assert AM.attribute(row(ok=False, stage="place", place=P(), pick=K(), excluded="infra.rc")) == (None, None)


def test_stage_of_program():
    meta = {"program": PROG_PLACE, "actions": [{"name": n.split("[")[0]} for n in PROG_PLACE[:6]],  # executed up to the failure only
            "verdict": {"category": "plan_failed", "action": "grasp", "action_index": 5}}
    assert AM.stage_of(meta) == "pick"
    meta["verdict"] = {"category": "robot_wobble", "action": "place", "action_index": 10}
    assert AM.stage_of(meta) == "place"
    meta["verdict"] = {"category": "collision", "action": "navigate", "action_index": 8}
    assert AM.stage_of(meta) == "carry"
    meta["verdict"] = {"category": "robot_wobble", "action": "place", "action_index": 9}  # an index off by one: the name decides
    assert AM.stage_of(meta) == "place"
    meta["verdict"] = {"category": "symbolic_only"}
    assert AM.stage_of(meta) == "outcome"
    meta["verdict"] = {"category": "plan_failed", "action": "push", "action_index": 2}  # the push-to-pick belongs to the pick
    assert AM.stage_of(meta) == "pick"
    art = {"program": ["navigate[]()", "transition[]()", "grasp[right](d)", "base_follow[right](d)", "retreat[right](d)"],
           "verdict": {"category": "object_tilt", "action": "base_follow", "action_index": 3}}
    assert AM.stage_of(art) == "manipulation"
    stepped = {"program": PROG_PLACE, "actions": [{"name": "navigate", "step_start": 0, "step_end": 10}, {"name": "transition", "step_start": 10, "step_end": 20}],
               "verdict": {"category": "arm_stutter", "step": 12}}
    assert AM.stage_of(stepped) == "pick"


def test_exclusions_file_and_rules():
    rules = AM.load_exclusions()
    ids = {r["id"] for r in rules}
    assert {"infra.status_error", "infra.rc", "infra.timeout", "infra.missing_meta", "infra.missing_sidecar"} <= ids
    for r in rules:
        assert r.get("why") and r.get("kind") in ("infrastructure", "known_bug")
    base = {"status": "oracle_failed", "rc": 2, "reason": "robot_wobble", "meta": True, "sidecar": True, "wall_clock_exceeded": False, "ctx": {"snapshot": "trunk_a"}}
    assert AM.exclusion_of(base, rules) is None
    assert AM.exclusion_of({**base, "status": "success", "rc": 0, "reason": None}, rules) is None
    assert AM.exclusion_of({**base, "status": "error", "rc": 4}, rules) == "infra.status_error"
    assert AM.exclusion_of({**base, "rc": 137}, rules) == "infra.rc"
    assert AM.exclusion_of({**base, "reason": "timeout"}, rules) == "infra.timeout"
    assert AM.exclusion_of({**base, "wall_clock_exceeded": True}, rules) == "infra.wall_clock"
    assert AM.exclusion_of({**base, "meta": False}, rules) == "infra.missing_meta"
    assert AM.exclusion_of({**base, "sidecar": False}, rules) == "infra.missing_sidecar"
    bug = [{"id": "bug.x", "kind": "known_bug", "match": {"failure_type": "pause", "snapshot": ["trunk_a"]}, "why": "t"}]
    assert AM.exclusion_of({**base, "failure_type": "pause"}, bug) == "bug.x"
    assert AM.exclusion_of({**base, "failure_type": "pause", "ctx": {"snapshot": "trunk_b"}}, bug) is None  # fixed later: counted again


# ------------------------------------------------------------------------------------------------------------- posterior

def test_posterior_math():
    assert AM.p_less(1, 4, 5, 1) > 0.95
    assert abs(AM.p_less(3, 3, 3, 3) - 0.5) < 0.02
    assert AM.p_less(5, 1, 1, 4) < 0.05
    assert abs(AM.beta_quantile(1, 1, 0.95) - 0.95) < 0.01
    # the owner's example cutoff (95 % upper bound < 10 %) needs 28 straight failures under a flat prior
    assert AM.beta_quantile(1, 1 + 27, 0.95) >= 0.1 > AM.beta_quantile(1, 1 + 29, 0.95)


def inside_large_01_rows():
    """The owner's case: 0.85 m stand-off, translate 0.25 went 4 / 4, 0.50 went 0 / 4 (2 robot_wobble, 1 robot_touching_itself at the
    place, 1 plan_failed at the grasp = a pick-stage failure)."""
    rs = [row(ok=True, place=P(tr=0.25), pick=K(), t=f"2026-11-07T07:3{i}") for i in range(4)]
    rs += [row(ok=False, stage="place", ftype=f, place=P(tr=0.5), pick=K(), t=f"2026-11-07T07:4{i}")
           for i, f in enumerate(["robot_wobble", "robot_wobble", "robot_touching_itself"])]
    rs += [row(ok=False, stage="pick", ftype="plan_failed", place=P(tr=0.5), pick=K(), t="2026-11-07T07:50")]
    return rs


def test_catches_inside_large_01_and_nothing_healthy():
    mem = AM.AliasMemory("a", inside_large_01_rows(), AM.Thresholds())
    mem.weights({"snapshot": "trunk_a", "spec": "s1", "keys": "k1"})
    bad = mem.decide("place", P(tr=0.5), max_level=AM.PLACE_HOOK_LEVELS)
    good = mem.decide("place", P(tr=0.25), max_level=AM.PLACE_HOOK_LEVELS)
    assert bad is not None and bad.down and bad.n_raw == 3 and bad.f == 3 and bad.s == 0  # the pick-stage plan_failed is not counted here
    assert good is not None and not good.down
    downs = mem.down_bins("place", max_level=AM.PLACE_HOOK_LEVELS)
    assert len(downs) == 1 and downs[0].key[:3] == ("right", 0.5, -0.6)


def test_healthy_and_all_failing_aliases_not_downweighted():
    healthy = [row(ok=True, place=P(tr=tr), t=f"2026-11-07T0{i}") for i, tr in enumerate([0.25, 0.25, 0.5, 0.5, 0.5, 0.25, 0.5])]
    healthy.append(row(ok=False, stage="place", place=P(tr=0.5), ftype="collision", t="2026-11-07T09"))
    mem = AM.AliasMemory("h", healthy, AM.Thresholds())
    mem.weights(None)
    assert mem.down_bins("place") == []
    allfail = [row(alias="f", ok=False, stage="place", place=P(tr=tr), ftype="plan_failed", t=f"2026-11-07T0{i}", tag="b2")
               for i, tr in enumerate([0.25] * 4 + [0.5] * 4)]
    mem = AM.AliasMemory("f", allfail, AM.Thresholds())
    mem.weights(None)
    assert mem.down_bins("place") == []  # every configuration fails: a bug, not a configuration
    rep = AM.guardrails(allfail, AM.Thresholds(), batch="b2")
    assert any(a["kind"] == "all_failed" and a["alias"] == "f" for a in rep["alerts"])


def test_backoff_to_coarser_level():
    th = AM.Thresholds(min_n=3)
    rs = [row(ok=True, place=P(tr=0.25, so=so), t=f"2026-11-07T0{i}") for i, so in enumerate([0.35, 0.45, 0.55, 0.65, 0.75])]
    # 0.50 fails at three DIFFERENT stand-offs: no stand-off bin has 3 demos -> the trunk level decides
    rs += [row(ok=False, stage="place", place=P(tr=0.5, so=so), t=f"2026-11-07T1{i}") for i, so in enumerate([0.35, 0.45, 0.55])]
    mem = AM.AliasMemory("a", rs, th)
    mem.weights(None)
    d = mem.decide("place", P(tr=0.5, so=0.45), max_level=AM.PLACE_HOOK_LEVELS)
    assert d is not None and d.level == 0 and d.down
    # with enough demos at one stand-off the finer level decides (and here says healthy: 0.65 m works at 0.50)
    rs += [row(ok=True, place=P(tr=0.5, so=0.65), t=f"2026-11-07T2{i}") for i in range(3)]
    mem = AM.AliasMemory("a", rs, th)
    mem.weights(None)
    d = mem.decide("place", P(tr=0.5, so=0.65), max_level=AM.PLACE_HOOK_LEVELS)
    assert d is not None and d.level == 1 and not d.down
    assert mem.decide("place", P(tr=0.75, so=0.35), max_level=AM.PLACE_HOOK_LEVELS) is None  # no data at any level: the prior stands


# ------------------------------------------------------------------------------------------------------------- forgetting

def test_decay_and_reset():
    th = AM.Thresholds()
    rs = [row(ok=False, stage="place", place=P(), ctx={"snapshot": s, "spec": "s1", "keys": "k1"}, t=t)
          for s, t in [("trunk_a", "2026-11-01"), ("trunk_b", "2026-11-02"), ("trunk_c", "2026-11-03")]]
    mem = AM.AliasMemory("a", rs, th)
    assert mem.weights({"snapshot": "trunk_c", "spec": "s1", "keys": "k1"}) == pytest.approx([0.64, 0.8, 1.0])
    assert mem.weights({"snapshot": "trunk_new", "spec": "s1", "keys": "k1"}) == pytest.approx([0.512, 0.64, 0.8])  # a new snapshot ages all
    assert mem.weights({"snapshot": "trunk_c", "spec": "s2", "keys": "k1"}) == [0.0, 0.0, 0.0]  # customs changed: reset
    assert mem.weights({"snapshot": "trunk_c", "spec": "s1", "keys": "k2"}) == [0.0, 0.0, 0.0]  # entry / map keys changed: reset
    mem.th = AM.Thresholds(customs_change=0.5)  # i7-memory2: a carried-over demo is discounted once (flat), not also by its age
    assert mem.weights({"snapshot": "trunk_c", "spec": "s2", "keys": "k1"}) == pytest.approx([0.5, 0.5, 0.5])
    # reset evidence does not count towards min_n either
    mem = AM.AliasMemory("a", inside_large_01_rows(), th)
    mem.weights({"snapshot": "trunk_a", "spec": "s_other", "keys": "k1"})
    assert mem.decide("place", P(tr=0.5), max_level=AM.PLACE_HOOK_LEVELS) is None


def test_keys_fingerprint(tmp_path):
    glib, maps = tmp_path / "glib", tmp_path / "maps"
    (glib / "models" / "hardback__bqvvvx__s1").mkdir(parents=True)
    (glib / "models" / "hardback__bqvvvx__s1" / "library.json").write_text(json.dumps({"grasps": [{"name": "g0"}], "provenance": {"t": 1}}))
    (maps / "hardback__bqvvvx__s1").mkdir(parents=True)
    (maps / "hardback__bqvvvx__s1" / "map__a.json").write_text("{}")
    entry = {"roles": {"target": {"category": "hardback", "representative": "bqvvvx", "grasp_keys": {"hardback/bqvvvx": "hardback__bqvvvx__bo"}}}}
    pre = AM.model_prefixes(entry)
    assert pre == ["hardback__bqvvvx__"]
    AM._FP_CACHE.clear()
    f0 = AM.keys_fingerprint(str(glib), str(maps), None, pre)
    (glib / "models" / "hardback__bqvvvx__s1" / "library.json").write_text(json.dumps({"grasps": [{"name": "g0"}], "provenance": {"t": 2}}))
    AM._FP_CACHE.clear()
    assert AM.keys_fingerprint(str(glib), str(maps), None, pre) == f0  # provenance only: same keys
    (maps / "hardback__bqvvvx__s1" / "map__b.json").write_text("{}")
    AM._FP_CACHE.clear()
    assert AM.keys_fingerprint(str(glib), str(maps), None, pre) != f0  # a new map: new keys


# ------------------------------------------------------------------------------------------------------------- the sampling hook

class _Sampler:
    def __init__(self):
        self.kit = types.SimpleNamespace(stats={})
        self.gp = types.SimpleNamespace(variant="v", entry={}, spec_hash=lambda: "s1")
        self.ctx = types.SimpleNamespace(spec_sha256="s1")


def _mem(rows):
    m = AM.AliasMemory("a", rows, AM.Thresholds())
    m.weights({"snapshot": "trunk_a", "spec": "s1", "keys": "k1"})
    return m


def test_reorder_keeps_prior_and_defers_down_bins():
    mem = _mem(inside_large_01_rows())
    cfgs = [P(tr=0.5), P(tr=0.25), P(tr=0.5, so=0.65), P(tr=0.25, so=0.65), P(tr=0.5)]
    prior = [4, 0, 1, 2, 3]  # e.g. i7-placeorder's order
    deferred = 0
    for seed in range(200):
        rng = np.random.default_rng(seed)
        new, prov = AM.reorder(_Sampler(), "place", prior, cfgs, rng, max_level=AM.PLACE_HOOK_LEVELS, mem=mem)
        assert sorted(new) == sorted(prior)  # nothing removed
        front = [k for k in new if not prov.get(k, {}).get("deferred")]
        back = [k for k in new if prov.get(k, {}).get("deferred")]
        assert new == front + back
        assert front == [k for k in prior if k in front] and back == [k for k in prior if k in back]  # the prior order within each part
        if back:
            deferred += 1
            # the 0.85 m / 0.50 candidates (one bin: they move together), and 0.50 at 0.65 m: no demo there, so the trunk level (its only
            # evidence the three failures) decides - another bin with its own draw
            assert set(back) <= {4, 0, 2} and (0 in back) == (4 in back)
        assert all(prov[k]["version"] == "-" for k in prov)
    assert 0 < deferred < 200


def test_exploration_floor():
    rs = [row(ok=True, place=P(tr=0.25), t=f"2026-11-0{1 + i % 7}T{i:02d}") for i in range(30)]
    rs += [row(ok=False, stage="place", place=P(tr=0.5), t=f"2026-11-0{1 + i % 7}T{i:02d}") for i in range(60)]
    mem = _mem(rs)
    kept = 0
    n = 4000
    for seed in range(n):
        new, prov = AM.reorder(_Sampler(), "place", [0, 1], [P(tr=0.5), P(tr=0.25)], np.random.default_rng(seed), mem=mem)
        assert prov[0]["weight"] >= 0.05
        kept += new[0] == 0
    assert 0.03 < kept / n < 0.09  # never zero: the floor (5 %) keeps the hopeless bin's place now and then


def test_no_down_bin_is_bit_identical():
    healthy = [row(ok=True, place=P(tr=tr), t=f"2026-11-07T0{i}") for i, tr in enumerate([0.25, 0.5] * 4)]
    mem = _mem(healthy)
    rng = np.random.default_rng(7)
    st0 = json.dumps(rng.bit_generator.state, sort_keys=True)
    s = _Sampler()
    new, prov = AM.reorder(s, "place", [3, 1, 2, 0], [P(tr=0.25), P(tr=0.5), P(tr=0.25), P(tr=0.5)], rng, mem=mem)
    assert new == [3, 1, 2, 0]
    assert json.dumps(rng.bit_generator.state, sort_keys=True) == st0  # the sampler's stream untouched
    assert s.kit.stats.get("alias_memory.place.no_down") == 1
    # and the memory's own draws never advance the sampler's generator either
    mem = _mem(inside_large_01_rows())
    AM.reorder(s, "place", [0, 1], [P(tr=0.5), P(tr=0.25)], rng, mem=mem)
    assert json.dumps(rng.bit_generator.state, sort_keys=True) == st0


def test_eval_generation_never_uses_memory(tmp_path, monkeypatch):
    v = AM.write_version(str(tmp_path / "mem"), inside_large_01_rows(), AM.Thresholds(), batch="t1")
    monkeypatch.setenv(AM.ENV_PATH, v)
    monkeypatch.setenv(AM.ENV_ALIAS, "a")
    monkeypatch.setenv("ISGE_CODE", "/x/trunk_a")
    AM._LOADED.clear()
    for split in (None, "eval"):
        AM.enable_for_split(split)
        s = _Sampler()
        assert AM.memory_for(s) is None
        new, prov = AM.reorder(s, "place", [0, 1], [P(tr=0.5), P(tr=0.25)], np.random.default_rng(0))
        assert new == [0, 1] and prov == {}
        assert s.kit.stats.get("alias_memory.blocked_not_train") == 2
    AM.enable_for_split("train")
    try:
        s = _Sampler()
        mem = AM.memory_for(s)
        assert mem is not None and s.kit.stats.get("alias_memory.loaded") == 1
    finally:
        AM.enable_for_split(None)
        AM._LOADED.clear()


# ------------------------------------------------------------------------------------------------------------- record / shards / versions

def _fake_job(root, tag, alias, idx, ok, stage_action=None, translate=0.25, ftype="robot_wobble", mode=None):
    d = root / "out" / tag / alias / f"g{idx}"
    (d / "inst").mkdir(parents=True)
    (d / "demo").mkdir()
    (d / "job.env").write_text(f"ALIAS=var\nALIAS_KEY={alias}\nISGE_CODE=/c/trunk_x\n" + (f"{AM.ENV_MODE}={mode}\n" if mode else ""))
    (d / "jobid").write_text(f"{idx}\n")
    (d / "job.txt").write_text("ISGF_END 1\n")
    (d / "inst" / "task_custom_lists.json").write_text(json.dumps({"var": {"roles": {}}}))
    tro = d / "inst" / f"var__t__{idx}-tro_state.json"
    tro.write_text(json.dumps({"ispatialgym": {"task_spec_sha256": "spec0"}}))
    sc = d / "inst" / f"var__t__{idx}.oracle.json"
    sc.write_text(json.dumps({"stances": {"placement": {"translate": translate, "tilt": -0.6, "stand_off": 0.85, "arms": ["right"]},
                                          "manipulation": {"translate": 0.25, "tilt": -0.8, "stand_off": 0.45, "arms": ["right"]}},
                              "grasps": [{"arm": "right", "grasp_name": "side_2~r180", "grasp_key": "k@x", "yaw": 3.14159, "place": {"position": [0, 0, 0]}}]}))
    demo = d / "demo" / f"var__t__{idx}"
    demo.mkdir()
    meta = {"program": PROG_PLACE, "sidecar": str(sc)}
    if not ok:
        meta["verdict"] = {"category": ftype, "action": stage_action, "action_index": [n.split("[")[0] for n in PROG_PLACE].index(stage_action)}
    (demo / "meta.json").write_text(json.dumps(meta))
    (d / "results.jsonl").write_text(json.dumps({"instance": f"var__t__{idx}", "rc": 0 if ok else 2, "status": "success" if ok else "oracle_failed",
                                                 "success": ok, "reason": None if ok else ftype, "tro": str(tro), "out": str(demo),
                                                 "finished": f"2026-11-07T0{idx % 10}:00:00"}) + "\n")
    return d


def test_record_shards_merge_and_versions(tmp_path):
    root = tmp_path / "root"
    jobs = [_fake_job(root, "t1", "al_01", 1, True), _fake_job(root, "t1", "al_01", 2, False, "place", translate=0.5),
            _fake_job(root, "t2", "al_01", 3, False, "grasp", translate=0.5), _fake_job(root, "t2", "al_02", 4, True)]
    p = AM.record_job(str(jobs[0]))
    assert os.path.basename(p) == AM.SHARD_NAME
    r0 = json.loads(open(p).read())
    assert r0["place"] == {"arm": "right", "stance": "placement", "stand_off": 0.85, "translate": 0.25, "tilt": -0.6, "yaw_slot": 8, "edge": None}
    assert r0["pick"]["family"] == "side_2" and r0["pick"]["stance"] == "manipulation"
    assert r0["ctx"]["snapshot"] == "trunk_x" and r0["ctx"]["spec"] == "spec0" and (r0["place_obs"], r0["pick_obs"]) == (1, 1)
    rows = AM.collect_rows(str(root))  # job 1 from its shard, the others back-filled
    assert len(rows) == 4 and sum(1 for r in rows if r.get("backfilled")) == 3
    by = {r["index"]: r for r in rows}
    assert (by[2]["stage"], by[2]["place_obs"], by[2]["pick_obs"]) == ("place", 0, 1)
    assert (by[3]["stage"], by[3]["place_obs"], by[3]["pick_obs"]) == ("pick", None, 0)
    mroot = str(tmp_path / "mem")
    v1 = AM.write_version(mroot, rows, AM.Thresholds(), batch="t2")
    v2 = AM.write_version(mroot, rows, AM.Thresholds(), batch="t2")
    assert os.path.basename(v1) == "v0001" and os.path.basename(v2) == "v0002" and AM.latest(mroot) == v2
    assert sorted(os.listdir(os.path.join(v2, "aliases"))) == ["al_01.json", "al_02.json"]
    m = AM.load_version(v2, "al_01")
    assert m is not None and len(m.rows) == 3 and m.version == "v0002"
    rep = json.load(open(os.path.join(v2, "report.json")))
    assert set(rep["aliases"]) == {"al_01", "al_02"} and rep["batch"] == "t2"
    assert AM.load_version(v2, "nope") is None


def test_shared_fix_candidate_and_space_alert():
    rs = []
    for a in ("x1", "x2", "x3"):
        rs += [row(alias=a, ok=True, place=P(tr=0.25), t=f"2026-11-07T0{i}") for i in range(4)]
        rs += [row(alias=a, ok=False, stage="place", place=P(tr=0.5), t=f"2026-11-07T1{i}") for i in range(4)]
    rep = AM.guardrails(rs, AM.Thresholds())
    assert rep["shared_fix_candidates"] and set(rep["shared_fix_candidates"][0]["aliases"]) == {"x1", "x2", "x3"}
    assert all(rep["aliases"][a]["place"]["bins_down"] == 1 for a in ("x1", "x2", "x3"))
    assert not any(al["kind"] == "space_lost" for al in rep["alerts"])  # 4 of 8 demos = 50 %: not more than the alert share
    rep = AM.guardrails(rs, AM.Thresholds(alert_space_share=0.4))
    assert sum(1 for al in rep["alerts"] if al["kind"] == "space_lost") == 3


def test_sampling_key_matches_recorded_key_and_provenance_round_trips():
    """The hook's key of a stance-C candidate equals the key the recorder reads back from the written sidecar; the provenance extra
    survives the sampler's sidecar serialisation and the oracle's sidecar parser (kept in ``extra``)."""
    from ispatialgym.oracle import sidecar as OS
    from ispatialgym.sampling.v2.stances import Stance

    st = Stance("placement", (1.0, 2.0, 0.5), 0.5, -0.6000001, 0.85, arms=["right"])
    st.extras["alias_memory"] = {"version": "v0003", "level": 1, "bin": ["right", 0.5, -0.6, 0.85], "weight": 0.27, "deferred": False, "down": True}
    block = st.to_sidecar()
    json.dumps(block)
    place, _ = AM.config_keys({"stances": {"placement": block}, "grasps": [{"arm": "right", "yaw": 0.0, "place": {}}]})
    cfg = AM.place_cfg("right", st)
    assert {k: place[k] for k in cfg} == cfg
    parsed = OS.Stance.from_dict(block, "stances.placement")
    assert parsed.extra["alias_memory"]["version"] == "v0003"


# ------------------------------------------------------------------------------------------------------------- i7-memory2: modes, stores, pick hook

def test_owner_thresholds_are_the_code_defaults():
    th = AM.Thresholds()
    assert (th.min_n, th.p_worse, th.max_mean, th.floor, th.snapshot_decay) == (3, 0.95, 0.35, 0.05, 0.8)  # owner 2.1
    assert (th.customs_change, th.keys_change, th.seed_weight, th.weight_floor, th.mode) == (0.0, 0.0, 0.0, 0.0, "production")
    v = AM.Thresholds.for_mode("validation")
    assert (v.min_n, v.p_worse, v.max_mean, v.floor, v.snapshot_decay) == (3, 0.95, 0.35, 0.05, 0.8)
    assert v.customs_change == v.keys_change == v.seed_weight == v.weight_floor == pytest.approx(1.0 / 3.0) and v.mode == "validation"
    assert AM.Thresholds.for_mode("production") == th
    with pytest.raises(ValueError):
        AM.Thresholds.for_mode("other")


def _rows_ctx(spec="s1", keys="k1", snap="trunk_a", seed=None):
    rs = inside_large_01_rows()
    for r in rs:
        r["ctx"] = {"snapshot": snap, "spec": spec, "keys": keys}
        if seed:
            r["seed"] = seed
    return rs


def test_production_starts_empty_and_resets(tmp_path):
    root = tmp_path / "root"
    _fake_job(root, "t1", "al_01", 1, False, "place", translate=0.5)                       # a job from before the modes
    _fake_job(root, "t1", "al_01", 2, False, "place", translate=0.5, mode="validation")
    v = AM.aggregate(str(root), mode="production", gpu=False)
    assert os.path.dirname(v) == str(root / "alias_memory")
    meta = json.load(open(os.path.join(v, "meta.json")))
    assert meta["n_rows"] == 0 and meta["mode"] == "production" and meta["seed"] is None  # starts EMPTY: no history, no validation demo
    _fake_job(root, "t2", "al_01", 3, False, "place", translate=0.5, mode="production")
    v = AM.aggregate(str(root), mode="production", gpu=False)
    rows = AM.load_version(v, "al_01").rows
    assert [r["index"] for r in rows] == [3] and rows[0]["mode"] == "production"
    # production resets on a customs or keys change (the 0 / 3 evidence counts nothing there)
    mem = AM.AliasMemory("a", _rows_ctx(), AM.Thresholds())
    assert all(w == 0.0 for w in mem.weights({"snapshot": "trunk_b", "spec": "s2", "keys": "k1"}))
    assert all(w == 0.0 for w in mem.weights({"snapshot": "trunk_b", "spec": "s1", "keys": "k2"}))
    assert mem.decide("place", P(tr=0.5), max_level=AM.PLACE_HOOK_LEVELS) is None


def test_validation_seeds_and_carries_over_at_reduced_weight(tmp_path):
    w = AM.VALIDATION_WEIGHT
    th = AM.Thresholds.for_mode("validation")
    # carried over a customs / keys change: flat w (not also aged)
    mem = AM.AliasMemory("a", _rows_ctx(snap="trunk_a"), th, snap_order=["trunk_a", "trunk_b", "trunk_c"])
    assert mem.weights({"snapshot": "trunk_new", "spec": "s2", "keys": "k1"}) == pytest.approx([w] * 8)
    assert mem.weights({"snapshot": "trunk_new", "spec": "s1", "keys": "k2"}) == pytest.approx([w] * 8)
    # the run's own context: decays per newer snapshot, never below w
    assert mem.weights({"snapshot": "trunk_b", "spec": "s1", "keys": "k1"}) == pytest.approx([0.8] * 8)
    assert mem.weights({"snapshot": "trunk_new", "spec": "s1", "keys": "k1"}) == pytest.approx([max(0.8 ** 3, w)] * 8)
    mem = AM.AliasMemory("a", _rows_ctx(snap="trunk_a"), th, snap_order=["trunk_a"] + [f"trunk_{i}" for i in range(8)])
    assert mem.weights({"snapshot": "trunk_7", "spec": "s1", "keys": "k1"}) == pytest.approx([w] * 8)
    # seeded demos: w whatever their context
    mem = AM.AliasMemory("a", _rows_ctx(seed="seed:x/v0003"), th)
    assert mem.weights({"snapshot": "trunk_a", "spec": "s1", "keys": "k1"}) == pytest.approx([w] * 8)
    # the owner's case: inside_large_01's 0 / 3 at 0.85 m / 0.50, seeded, at a NEW snapshot under the changed customs and keys -
    # flagged in validation (mean 1 / (2 + 3 w) = 1 / 3 <= 0.35), the healthy 0.25 not; production would not see it at all
    rs = _rows_ctx(spec="s_old", keys="k_old", seed="seed:x/v0003")
    rs += [dict(row(ok=True, place=P(tr=0.25, so=0.65), t=f"2026-11-06T1{i % 10}", ctx={"snapshot": "trunk_a", "spec": "s_old", "keys": "k_old"}),
                seed="seed:x/v0003") for i in range(30)]  # the alias's healthy rest (inside_large_01: ~40 successes)
    new = {"snapshot": "trunk_new", "spec": "s_now", "keys": "k_now"}
    mem = AM.AliasMemory("a", rs, th)
    mem.weights(new)
    bad, good = mem.decide("place", P(tr=0.5), max_level=2), mem.decide("place", P(tr=0.25), max_level=2)
    assert bad.down and bad.mean == pytest.approx(1.0 / 3.0, abs=1e-6) and not good.down
    prod = AM.AliasMemory("a", [dict(r, seed=None) for r in rs], AM.Thresholds())
    prod.weights(new)
    assert prod.decide("place", P(tr=0.5), max_level=2) is None
    # a weight below 0.286 would not flag it (the reason w is 1/3, not less)
    low = AM.AliasMemory("a", rs, AM.Thresholds(seed_weight=0.25, customs_change=0.25, keys_change=0.25, weight_floor=0.25, mode="validation"))
    low.weights(new)
    assert not low.decide("place", P(tr=0.5), max_level=2).down


def _seed_version(tmp_path):
    rs = _rows_ctx()
    for i, r in enumerate(rs):
        r.update({"job": f"/seed/job{i}", "inst": f"i{i}", "alias": "al_01"})
    return AM.write_version(str(tmp_path / "seedroot"), rs, AM.Thresholds(), batch="replay")


def test_stores_are_separate(tmp_path, monkeypatch):
    root = tmp_path / "root"
    seed = _seed_version(tmp_path)
    _fake_job(root, "t1", "al_01", 1, True)                       # before the modes: validation only (as seed "legacy")
    _fake_job(root, "t1", "al_01", 2, True, mode="validation")
    _fake_job(root, "t1", "al_01", 3, True, mode="production")
    vv = AM.aggregate(str(root), mode="validation", seed=str(tmp_path / "seedroot"), gpu=False)
    vp = AM.aggregate(str(root), mode="production", gpu=False)
    assert os.path.dirname(vv) == str(root / "alias_memory_validation") and os.path.dirname(vp) == str(root / "alias_memory")
    rv = AM.load_version(vv, "al_01").rows
    rp = AM.load_version(vp, "al_01").rows
    assert sorted(r.get("index") for r in rv if not str(r.get("seed", "")).startswith("seed:")) == [1, 2]
    assert sum(1 for r in rv if str(r.get("seed", "")).startswith("seed:")) == 8 and {r.get("seed") for r in rv if r.get("index") == 1} == {"legacy"}
    assert [r["index"] for r in rp] == [3]
    mv = json.load(open(os.path.join(vv, "meta.json")))
    assert mv["mode"] == "validation" and mv["seed"] == os.path.abspath(seed) and mv["n_seeded"] == 9
    assert json.load(open(root / "alias_memory_validation" / "seed.json"))["seed"] == os.path.abspath(seed)
    assert json.load(open(root / "alias_memory" / "thresholds.json"))["mode"] == "production"
    # a store's thresholds file belongs to its mode: aggregating the other mode into it is refused
    with pytest.raises(SystemExit):
        AM.aggregate(str(root), mem_root=str(root / "alias_memory"), mode="validation", gpu=False)
    with pytest.raises(SystemExit):
        AM.aggregate(str(root), mem_root=str(root / "alias_memory_validation"), mode="production", gpu=False)
    # the sampler refuses a version of the other mode's store
    monkeypatch.setenv(AM.ENV_ALIAS, "al_01")
    monkeypatch.setenv("ISGE_CODE", "/x/trunk_a")
    AM.enable_for_split("train")
    try:
        for path, mode, ok in ((vv, "production", False), (vp, "validation", False), (vv, "validation", True), (vp, "production", True)):
            AM._LOADED.clear()
            AM._MISMATCH.clear()
            monkeypatch.setenv(AM.ENV_PATH, path)
            monkeypatch.setenv(AM.ENV_MODE, mode)
            s = _Sampler()
            assert (AM.memory_for(s) is not None) == ok
            assert bool(s.kit.stats.get("alias_memory.mode_mismatch")) == (not ok)
    finally:
        AM.enable_for_split(None)
        AM._LOADED.clear()
        AM._MISMATCH.clear()


def test_thresholds_read_from_the_file(tmp_path):
    root = tmp_path / "root"
    _fake_job(root, "t1", "al_01", 1, True, mode="production")
    store = root / "alias_memory"
    store.mkdir(parents=True)
    (store / "thresholds.json").write_text(json.dumps({"min_n": 5, "p_worse": 0.9, "floor": 0.1}))
    v = AM.aggregate(str(root), gpu=False)
    mem = AM.load_version(v, "al_01")
    assert (mem.th.min_n, mem.th.p_worse, mem.th.floor, mem.th.max_mean) == (5, 0.9, 0.1, 0.35)  # the file wins, the rest default
    # no file yet: written with the mode's defaults (the file the memory reads from then on)
    v2 = AM.aggregate(str(root), mode="validation", seed=_seed_version(tmp_path), gpu=False)
    f = json.load(open(root / "alias_memory_validation" / "thresholds.json"))
    assert f == json.loads(json.dumps(AM.dataclasses.asdict(AM.Thresholds.for_mode("validation"))))
    assert AM.load_version(v2, "al_01").th == AM.Thresholds.for_mode("validation")


class _St:
    def __init__(self, tr, ti=-0.53, so=0.35, arms=("right",)):
        self.translate, self.tilt, self.stand_off, self.arms, self.extras = tr, ti, so, list(arms), {}


def _pick_rows():
    rs = [row(ok=True, pick=K(tr=0.0, ti=-0.61, so=0.35), t=f"2026-11-07T0{i}") for i in range(6)]
    rs += [row(ok=False, stage="pick", ftype="robot_wobble", pick=K(tr=0.25, ti=-0.53, so=0.35), t=f"2026-11-07T1{i}") for i in range(4)]
    return rs


def test_pick_hook_reorders_but_never_blocks():
    mem = _mem(_pick_rows())
    stances = [_St(0.25), _St(0.0, -0.61), _St(0.25), _St(0.0, -0.61), _St(0.5, -0.45)]
    d = mem.decide("pick", AM.pick_cfg("right", stances[0]), max_level=AM.PICK_HOOK_LEVELS)
    assert d.down and d.key == ("right", 0.25, -0.53, 0.35)
    deferred = kept = 0
    for seed in range(300):
        s = _Sampler()
        AM._LOADED.clear()
        new, prov = AM.reorder(s, "pick", [2, 0, 1, 4, 3], [AM.pick_cfg("right", st) for st in stances], np.random.default_rng(seed),
                               max_level=AM.PICK_HOOK_LEVELS, mem=mem)
        assert sorted(new) == [0, 1, 2, 3, 4]  # never removed
        back = [k for k in new if prov.get(k, {}).get("deferred")]  # 4 (trunk 0.50): no evidence, no provenance
        assert new == [k for k in [2, 0, 1, 4, 3] if k not in back] + back and set(back) <= {0, 2}
        st = s.kit.stats
        assert st["alias_memory.pick.flagged"] == 2 and st["alias_memory.pick.reordered"] == len(back) and st["alias_memory.pick.explored"] == 2 - len(back)
        deferred += bool(back)
        kept += not back
    assert deferred > 200 and kept > 0  # mostly deferred; a Thompson draw (never below the floor) keeps the flagged bin in place now and then


def test_reorder_pick_wrapper_and_off(monkeypatch):
    stances = [_St(0.25), _St(0.0, -0.61)]
    rng = np.random.default_rng(3)
    st0 = json.dumps(rng.bit_generator.state, sort_keys=True)
    monkeypatch.delenv(AM.ENV_PATH, raising=False)
    AM.enable_for_split("train")
    try:
        s = _Sampler()
        assert AM.reorder_pick(s, np.array([1, 0]), stances, lambda st: "right", rng) == [1, 0]  # memory off: the order as given
        assert not s.kit.stats and all("alias_memory_pick" not in st.extras for st in stances)
        mem = _mem(_pick_rows())
        monkeypatch.setattr(AM, "memory_for", lambda sampler: mem)
        n_back = 0
        for _ in range(50):
            new = AM.reorder_pick(s, [0, 1], stances, lambda st: "right", np.random.default_rng(_))
            assert sorted(new) == [0, 1]
            n_back += new == [1, 0]
        assert n_back > 30
        assert stances[0].extras["alias_memory_pick"]["down"] and not stances[1].extras["alias_memory_pick"]["down"]
        assert json.dumps(rng.bit_generator.state, sort_keys=True) == st0
    finally:
        AM.enable_for_split(None)


def test_pick_provenance_recorded(tmp_path):
    root = tmp_path / "root"
    d = _fake_job(root, "t1", "al_01", 1, True, mode="validation")
    sc = next((d / "inst").glob("*.oracle.json"))
    j = json.loads(sc.read_text())
    j["stances"]["manipulation"]["alias_memory_pick"] = {"version": "v0001", "down": True, "deferred": False}
    sc.write_text(json.dumps(j))
    r = AM.demo_rows(str(d))[0]
    assert r["sampled_with_pick"]["version"] == "v0001" and r["mode"] == "validation"
