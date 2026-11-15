from types import SimpleNamespace

import pytest

from ispatialgym.multienv.campaign_rendering import selected_records, restore_campaign_scene
from ispatialgym.tests.test_campaign_augmentation import light_world
from scripts.campaign_20261110 import contracts as c
from scripts.campaign_20261110 import export_release as ex, render_shard as render
from scripts.campaign_20261110.tests.test_export_release import exported_fixture


def test_render_plan_only_accepts_committed_members_and_is_exclusive(tmp_path):
    f, rows, groups, inputs = exported_fixture(tmp_path)
    (f.job / "job.env").write_text("# fixture environment\n")
    destination, _ = ex.publish(tmp_path, "render01", rows, groups, inputs)
    shard = render.read(destination / "shards.json")["shards"][0]
    output = tmp_path / "rendering/canary01"
    plan = render.prepare(tmp_path, destination, shard["id"], tmp_path / "code/snapshot", output, limit=1)
    assert plan["instances"] == [rows[0]["instance_id"]]
    assert plan["source_job"] == str(f.job)
    # render switch 2026-11-15: merged scene (task rooms + adjacent) with replay-time perturbation arguments from the shard policy
    assert shard["scene_mode"] == plan["scene_mode"] == "full" and plan["scene_extent"] == shard["scene_extent"] == "task_rooms+adjacent"
    args = (output / "replay.args").read_text().split()
    assert args[:2] == ["--scene", "full"] and "--scene-pert-plan" in args and args[args.index("--lighting-pert") + 1] == "replay"
    assert args[args.index("--scene-pert-visibility") + 1] == "off" and args[args.index("--scene-pert-mode") + 1] == "one"
    assert args[args.index("--scene-pert-seed") + 1] == args[args.index("--lighting-pert-seed") + 1] == str(shard["replay_perturbation"]["seed"])
    assert (output / "scene_pert_plan.json").exists() and plan["scene_pert_plan"]["sha256"]
    with pytest.raises(FileExistsError):
        render.prepare(tmp_path, destination, shard["id"], tmp_path / "code/snapshot", output)
    with pytest.raises(c.ContractError, match="not in accepted manifest"):
        selected_records(destination / "instances.jsonl", [str(tmp_path / "not-accepted")])


def test_renderer_restores_exact_scale_and_lighting_and_rejects_scale_drift():
    from ispatialgym.sampling.campaign_augmentation import apply_lighting, LIGHT_SCHEMA

    scene, sim, light = light_world()
    request = {"schema": LIGHT_SCHEMA, "intensity_factor": .8, "skybox_color_factors": [1, 1, 1]}
    expected = apply_lighting(scene, sim, request)
    apply_lighting(scene, sim, {})
    obj = SimpleNamespace(name="apple_0", category="apple", model="abc", scale=[.9] * 3)
    scene.object_registry = lambda kind, name: obj if name == obj.name else None
    perturb = c.Perturbations({"apple.n.01_1": c.ScaleState((1, 1, 1), (.9, .9, .9))}, lighting=request)
    meta = {"object_bindings": {"apple.n.01_1": {"scene_object_name": obj.name, "category": obj.category, "model": obj.model}},
            "campaign": {"schema": "isg-campaign-v1", "split": "train", "perturbations": {
                "scales": {"apple.n.01_1": {"default": [1] * 3, "applied": [.9] * 3, "mode": "uniform"}},
                "removed_objects": [], "lighting": request}, "validation_key": {"scale_sha256": perturb.scale_fingerprint},
                "runtime": {"augmentation": {"lighting": expected}}}}
    assert restore_campaign_scene(scene, sim, meta)["scale_verified"]
    assert light.attr.Get() == 960
    obj.scale = [1] * 3
    with pytest.raises(c.ContractError, match="scale mismatch"):
        restore_campaign_scene(scene, sim, meta)


def test_reconcile_fails_missing_or_occluded_episodes(tmp_path):
    c.write_json_atomic(tmp_path / "launch.json", {"instances": ["one", "two"]})
    def finish(rows):
        (tmp_path / "results.jsonl").write_text("".join(c.canonical_json(row) + "\n" for row in rows))
        return render.reconcile(tmp_path, 0)
    row = {"tag": "REPLAY_DEMO", "demo": "/data/one", "kept": True, "episode_index": 0,
           "campaign_restoration": {"scale_verified": True}}
    summary = {"tag": "REPLAY_SUMMARY"}
    assert not finish([row, summary])["success"]
    second = {**row, "demo": "/data/two", "episode_index": 1}
    assert not finish([row, {**second, "kept": False}, summary])["success"]
    assert finish([row, second, summary])["success"]


def test_render_batch_cannot_mix_global_skybox_settings(tmp_path):
    path = tmp_path / "instances.jsonl"
    path.write_text("\n".join(c.canonical_json({"artifacts": {"demo": {"path": f"/data/{name}/demo.hdf5"}},
        "render_compatibility_sha256": key}) for name, key in (("a", "light-a"), ("b", "light-b"))))
    with pytest.raises(c.ContractError, match="mixes scene, scale, light"):
        selected_records(path, ["/data/a", "/data/b"])


def test_restoration_probe_allows_only_exact_aba_and_preserves_configuration_isolation(tmp_path,monkeypatch):
    path=tmp_path/'probe.jsonl'
    rows=[{'artifacts':{'demo':{'path':f'/data/{name}/demo.hdf5'}},'render_compatibility_sha256':'same'}for name in ('a','b')]
    path.write_text('\n'.join(c.canonical_json(r)for r in rows))
    with pytest.raises(c.ContractError,match='duplicated'):selected_records(path,['/data/a','/data/b','/data/a'])
    monkeypatch.setenv('ISPATIALGYM_CAMPAIGN_RESTORATION_PROBE','1')
    assert len(selected_records(path,['/data/a','/data/b','/data/a']))==2
    with pytest.raises(c.ContractError,match='duplicated'):selected_records(path,['/data/a','/data/a'])
    rows[1]['render_compatibility_sha256']='changed'
    path.write_text('\n'.join(c.canonical_json(r)for r in rows))
    with pytest.raises(c.ContractError,match='mixes scene'):selected_records(path,['/data/a','/data/b','/data/a'])


def test_render_hash_ignores_recorded_lighting_and_shards_keep_goal_groups():
    p = {"scales": {"a": 1}, "removed_objects": ["x"], "lighting": {"intensity_factor": 1.2}}
    assert ex.render_perturbations(p) == {"scales": {"a": 1}, "removed_objects": ["x"]}
    assert ex.render_augmentation({"lighting": {"f": 1}, "moved_objects": {}}) == {"moved_objects": {}}
    rows = [{"instance_id": f"i{n:02d}", "group_id": f"g{n // 3}"} for n in range(10)] + [{"instance_id": "solo", "group_id": None}]
    batches = ex.group_batches(rows, 4)
    assert all(len(b) <= 4 for b in batches) and sum(len(b) for b in batches) == len(rows)
    where = {}
    for i, b in enumerate(batches):
        for r in b:
            where.setdefault(r["group_id"] or r["instance_id"], set()).add(i)
    assert all(len(v) == 1 for v in where.values())  # no goal group spans two shards
    with pytest.raises(c.ContractError):
        ex.group_batches([{"instance_id": f"j{n}", "group_id": "big"} for n in range(5)], 4)


def test_reconcile_refuses_a_group_rendered_with_two_lighting_draws(tmp_path):
    c.write_json_atomic(tmp_path / "launch.json", {"instances": ["one"], "scene_mode": "full"})
    row = {"tag": "REPLAY_DEMO", "demo": "/data/one", "kept": True, "episode_index": 0, "campaign_restoration": {"scale_verified": True}}
    light = lambda f: {"tag": "REPLAY_SCENE_PERT", "group": "g", "lighting": {"skybox": {"intensity_factor": 1.0}, "fixtures_draw": {"intensity_factor": f}}}
    def finish(extra):
        (tmp_path / "results.jsonl").write_text("".join(c.canonical_json(r) + "\n" for r in [row, *extra, {"tag": "REPLAY_SUMMARY"}]))
        return render.reconcile(tmp_path, 0)["success"]
    assert finish([light(1.1), light(1.1)])
    assert not finish([light(1.1), light(0.9)])
