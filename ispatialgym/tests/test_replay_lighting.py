"""CPU tests of replay-time lighting (ispatialgym.multienv.replay_lighting)."""
import json

import pytest

from ispatialgym.multienv import replay_lighting as RL


def test_fixture_draw_deterministic_ranges_and_rate():
    a = RL.draw_fixtures(1, "demo", 0, {"p": 1.0})
    assert a == RL.draw_fixtures(1, "demo", 0, {"p": 1.0}) and a["perturbed"]
    draws = [RL.draw_fixtures(1, "demo", r, {"p": 1.0}) for r in range(400)]
    f = [d["intensity_factor"] for d in draws]
    assert min(f) >= 0.4 and max(f) <= 1.8 and len(set(f)) > 300
    assert all(0.82 <= c <= 1.18 for d in draws for c in d["color_factors"])
    n = sum(RL.draw_fixtures(2, "d", r)["perturbed"] for r in range(2000))
    assert 900 < n < 1100
    off = RL.draw_fixtures(2, "d", next(r for r in range(50) if not RL.draw_fixtures(2, "d", r)["perturbed"]))
    assert off["intensity_factor"] == 1.0 and off["color_factors"] == [1.0, 1.0, 1.0]
    # the range is a parameter
    narrow = [RL.draw_fixtures(1, "demo", r, {"p": 1.0, "lo": 0.9, "hi": 1.1})["intensity_factor"] for r in range(100)]
    assert min(narrow) >= 0.9 and max(narrow) <= 1.1


def test_skybox_per_batch_and_guard():
    k = RL.batch_key(["b", "a"])
    assert k == RL.batch_key(["a", "b"])
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
