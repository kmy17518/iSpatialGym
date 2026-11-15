"""Owner 2026-11-15: replay-time draws are independent per demo (--pert-unit demo): a demo's draw is reproducible, demos of one
goal group differ in the lighting coin / factor and the removed object (--scene-pert-inview-pick random)."""
from ispatialgym.multienv import replay_lighting as RL
from ispatialgym.multienv.replay_scene_perturbation import draw_one, group_key

CFG = {"dist": "uniform", "p": 0.5, "lo": 0.5, "hi": 1.5, "sigma": 0.17, "color_sigma": 0.04}
DEMOS = [f"demo_{i:03d}" for i in range(200)]


def test_lighting_per_demo_varies_and_reproduces():
    draws = [RL.draw_fixtures(20261115, d, 0, CFG) for d in DEMOS]
    assert draws == [RL.draw_fixtures(20261115, d, 0, CFG) for d in DEMOS]
    frac = sum(x["perturbed"] for x in draws) / len(draws)
    assert 0.35 < frac < 0.65
    assert len({round(x["intensity_factor"], 6) for x in draws if x["perturbed"]}) > 50


def test_scene_removal_per_demo_varies():
    units = [[f"obj_{k}"] for k in range(12)]
    prefer = [f"obj_{k}" for k in range(8)]
    picks = [draw_one(units, group_key(20261115, d, 0), 0.5, prefer=prefer, prefer_pick="random") for d in DEMOS]
    on = [p for p in picks if p["perturbed"]]
    assert 0.35 < len(on) / len(picks) < 0.65
    assert len({p["chosen"] for p in on}) == 8  # every in-view candidate gets picked by some demo
    assert all(p["chosen"] in prefer for p in on)
    first = [draw_one(units, group_key(20261115, d, 0), 1.0, prefer=prefer) for d in DEMOS]
    assert {p["chosen"] for p in first} == {"obj_0"}  # the old rule: always the top-ranked one
