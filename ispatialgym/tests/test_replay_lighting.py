"""CPU tests of replay-time lighting (ispatialgym.multienv.replay_lighting)."""
import json

import pytest

from ispatialgym.multienv import replay_lighting as RL


def test_fixture_draw_deterministic_ranges_and_rate():
    a = RL.draw_fixtures(1, "group", 0, {"p": 1.0})
    assert a == RL.draw_fixtures(1, "group", 0, {"p": 1.0}) and a["perturbed"]  # one draw per goal group
    draws = [RL.draw_fixtures(1, "demo", r, {"p": 1.0}) for r in range(400)]
    f = [d["intensity_factor"] for d in draws]
    assert min(f) >= 0.5 and max(f) <= 1.5 and len(set(f)) > 300
    assert all(0.88 <= c <= 1.12 for d in draws for c in d["color_factors"])
    import statistics

    # default: uniform on [0.5, 1.5] (sd 1/sqrt(12) = 0.289), dist and range recorded in the draw
    assert 0.26 < statistics.pstdev(f) < 0.32 and abs(statistics.mean(f) - 1.0) < 0.04 and min(f) < 0.55 and max(f) > 1.45
    assert draws[0]["dist"] == "uniform" and draws[0]["range"] == [0.5, 1.5]
    g = [RL.draw_fixtures(1, "demo", r, {"p": 1.0, "dist": "normal"})["intensity_factor"] for r in range(400)]
    assert 0.14 < statistics.pstdev(g) < 0.19 and min(g) >= 0.5 and max(g) <= 1.5
    sk = [RL.draw_skybox(1, str(r), {"p": 1.0})["intensity_factor"] for r in range(400)]
    assert 0.5 <= min(sk) < 0.55 and 1.45 < max(sk) <= 1.5
    n = sum(RL.draw_fixtures(2, "d", r)["perturbed"] for r in range(2000))
    assert 900 < n < 1100
    off = RL.draw_fixtures(2, "d", next(r for r in range(50) if not RL.draw_fixtures(2, "d", r)["perturbed"]))
    assert off["intensity_factor"] == 1.0 and off["color_factors"] == [1.0, 1.0, 1.0]
    # the range is a parameter
    narrow = [RL.draw_fixtures(1, "demo", r, {"p": 1.0, "lo": 0.9, "hi": 1.1})["intensity_factor"] for r in range(100)]
    assert min(narrow) >= 0.9 and max(narrow) <= 1.1


def test_skybox_per_batch_and_guard():
    k = RL.batch_key(["b", "a", "a"])
    assert k == RL.batch_key(["a", "b"])
    rec = lambda g, sky: {"group": g, "lighting": {"skybox": sky, "fixtures_draw": RL.draw_fixtures(3, g, 0, {"p": 1.0})}}
    sky0 = RL.draw_skybox(3, k, {"p": 1.0})
    RL.check_group_skybox([rec("a", sky0), rec("a", sky0), rec("b", sky0)])
    with pytest.raises(ValueError):
        RL.check_group_skybox([rec("a", sky0), rec("a", RL.draw_skybox(3, "other", {"p": 1.0}))])
    s = RL.draw_skybox(3, k, {"p": 1.0})
    assert s == RL.draw_skybox(3, k, {"p": 1.0}) and s["batch_key"] == k
    RL.check_batch_skybox([{"skybox": s}, {"skybox": dict(s)}])
    with pytest.raises(ValueError):
        RL.check_batch_skybox([{"skybox": s}, {"skybox": RL.draw_skybox(3, RL.batch_key(["c"]), {"p": 1.0})}])


def test_records_roundtrip_and_without_lighting(tmp_path):
    rec = {"tag": "REPLAY_SCENE_PERT", "demo_name": "d", "replay_index": 1, "lighting": {"fixtures_draw": {"intensity_factor": 1.3}}}
    p = tmp_path / "r.jsonl"
    p.write_text(json.dumps({"tag": "REPLAY_DEMO"}) + "\n" + json.dumps(rec) + "\n")
    assert RL.load_from_records(str(p)) == {("d", 1): rec["lighting"]}
    meta = {"campaign": {"perturbations": {"lighting": {"intensity_factor": 1.1}, "scales": {}},
                         "runtime": {"augmentation": {"lighting": {"x": 1}, "moved_objects": {}}}}}
    out = RL.without_lighting(meta)
    assert out["campaign"]["perturbations"]["lighting"] == {} and "lighting" not in out["campaign"]["runtime"]["augmentation"]
    assert meta["campaign"]["perturbations"]["lighting"] == {"intensity_factor": 1.1}  # input untouched
    assert RL.ignored_recorded(meta) == {"intensity_factor": 1.1}


def test_replayed_tro_roundtrip(tmp_path):
    """Loader path: the replayed-instance TRO carries the replay perturbations; loading it gives back the same removals and lights."""
    tro = tmp_path / "inst__1-tro_state.json"
    camp = {"schema": "x", "perturbations": {"scales": {}, "removed_objects": [], "lighting": {"intensity_factor": 1.1}}}
    tro.write_text(json.dumps({"ispatialgym": {"campaign": camp, "object_bindings": {}}}))
    fx = RL.draw_fixtures(5, "inst__1", 0, {"p": 1.0})
    sky = RL.draw_skybox(5, RL.batch_key(["inst__1"]), {"p": 1.0})
    rec = {"schema": "isg-replay-scene-pert-v1", "demo_name": "inst__1", "replay_index": 0, "seed": 5, "removed": ["chair_0", "desk_2"],
           "perturbed": True, "scope": "outside_crop", "scene_mode": "full",
           "lighting": {"schema": RL.SCHEMA, "seed": 5, "cfg": {}, "fixtures_draw": fx, "skybox": sky, "recorded_ignored": {"intensity_factor": 1.1}}}
    out = RL.write_replayed_tro(str(tro), str(tmp_path / "instances"), rec)
    data = json.loads(open(out).read())["ispatialgym"]["campaign"]
    assert data["perturbations"] == camp["perturbations"]  # collection-time record untouched
    assert data[RL.REPLAY_BLOCK]["lighting"]["fixtures_draw"] == fx and data[RL.REPLAY_BLOCK]["source_tro"]["path"] == str(tro)
    removed, lighting = RL.load_replayed_tros(str(tmp_path / "instances"))
    assert removed == {("inst__1", 0): ["chair_0", "desk_2"]}
    assert lighting == {("inst__1", 0): {"fixtures_draw": fx, "skybox": sky}}
    RL.check_batch_skybox(list(lighting.values()))
    from scripts.campaign_20261110 import contracts

    assert "replay_perturbations" not in contracts.validate_campaign_block.__code__.co_consts  # no strict campaign key list


def test_sweep_stats():
    import numpy as np

    a = np.zeros((2, 2, 3), np.uint8)
    a[0, 0] = 255
    a[0, 1] = 128
    st = RL.sweep_stats(a)
    assert st["black_frac"] == 0.5 and st["white_frac"] == 0.25 and 70 < st["mean_luma"] < 100
