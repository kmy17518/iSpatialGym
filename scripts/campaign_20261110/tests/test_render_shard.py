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
