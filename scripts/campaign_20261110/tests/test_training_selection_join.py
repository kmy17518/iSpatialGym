import copy
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from scripts.campaign_20261110.contracts import ObjectRef, Template, digest
from scripts.campaign_20261110.tests import test_prepared_articulation as prepared_tests
from scripts.campaign_20261110 import materialize_training_table as stream_binding
from scripts.campaign_20261110.training_selection_join import (
    join_generation, materialize_request, reference, verify_generated_binding,
)


case = prepared_tests.case
document = prepared_tests.document


@pytest.fixture
def selected(case):
    choice = json.loads(Path(case.options["selection_ref"]["path"]).read_text())["rows"][0]
    choice["rooms"] = ["garden_0"]
    selection = document(case.tmp / "selected.json", {"rows": [choice]})
    row = copy.deepcopy(case.row)
    row["scale_row_sha256"] = digest(choice)
    metadata = document(case.tmp / "selected_handles.json", {
        "schema": "isg-exact-articulation-handle-binding-v1", "source": selection, "rows": [row]})
    native_scene = document(case.tmp / "house_single_floor_stable.json", {
        "objects_info": {"init_info": {"door_model_0": {"args": {
            "category": "door", "model": "model", "scale": [1., 1., 1.]}}}}})
    target = ObjectRef("target", "target", "door", "model", "a" * 64,
                       joint="j_leaf", ready=True, readiness_source="asset_catalog")
    template = Template("selected_A", "configuration_matching-articulation_close_large_scale",
        "articulation_A", "train", "house_single_floor", "garden_0", (target,), digest("definition"),
        {"inst_to_name": {"target": "door_model_0"}, "task_rooms": ["garden_0"],
         "native_scene_sha256": native_scene["sha256"]})
    entry = copy.deepcopy(case.entry)
    entry["scenes"]["house_single_floor"]["bindings"] = {"target": ["door_model_0"]}
    customs = document(case.tmp / "source_customs.json", {"alias_A": entry})
    job = {"candidate_id": "selected_A", "class_id": "articulation_A", "scene": "house_single_floor",
           "alias": "alias_A", "production_released": False,
           "sample_argv": ["python", "-m", "ispatialgym.sampling.sample_isg", "--customs", customs["path"],
                           "--scene", "house_single_floor", "--out", "old_sampling",
                           "--no-write-back", "--seed", "123"]}
    options = dict(metadata_ref=metadata, selection_ref=selection, configuration_id="config_A",
                   native_scene_ref=native_scene, output=case.tmp / "joined")
    return template, job, options


def test_general_join_writes_normal_job_without_changing_world_or_sampler_arguments(selected):
    template, job, options = selected
    candidate, bound, manifest = join_generation(template, job, **options)
    assert candidate.objects == template.objects
    assert candidate.fingerprint != template.fingerprint
    assert candidate.template_id == bound["candidate_id"] != template.template_id
    assert bound["sample_argv"][-3:] == job["sample_argv"][-3:]
    assert bound["sample_argv"][4] != job["sample_argv"][4]
    assert bound["sample_argv"][8] == str(Path(options["output"]) / "sampling")
    assert bound["training_scale_plan"]["scales"]["target"]["mode"] == "fixed"
    assert manifest == reference(Path(options["output"]) / "validation_jobs.json")
    assert bound["selected_generation_join"]["native_actor_lookup_verified"] is False
    assert list(json.loads(Path(bound["sample_argv"][4]).read_text())["alias_A"]["roles"]) == ["target", "floor"]


@pytest.mark.parametrize("mutation", ["model", "asset", "joint", "rooms", "missing_actor", "scene_hash", "eval"])
def test_missing_or_incompatible_selected_context_is_blocked_before_materializing(selected, mutation):
    template, job, options = selected
    if mutation in ("model", "asset", "joint"):
        field = {"model": "model", "asset": "asset_sha256", "joint": "joint"}[mutation]
        template = replace(template, objects=(replace(template.objects[0], **{field: "b" * 64 if mutation == "asset" else "other"}),))
    if mutation == "rooms": template = replace(template, scene_setup={**template.scene_setup, "task_rooms": ["other"]})
    if mutation == "missing_actor": template = replace(template, scene_setup={**template.scene_setup, "inst_to_name": {}})
    if mutation == "scene_hash": template = replace(template, scene_setup={**template.scene_setup, "native_scene_sha256": "c" * 64})
    if mutation == "eval": template = replace(template, split="eval_standard")
    with pytest.raises(ValueError): join_generation(template, job, **options)
    assert not options["output"].exists()


def test_stream_binding_requires_native_lookup_then_preserves_it(selected, monkeypatch):
    template, job, options = selected
    candidate, bound, _ = join_generation(template, job, **options)
    path = Path(options["output"]) / "template_meta.json"
    path.write_text(json.dumps({"template_facts": {}}))
    with pytest.raises(ValueError, match="no exact prepared handle lookup"):
        stream_binding.bind_training_candidate(candidate, bound, path, "code", {}, {}, {})
    proof = bound["selected_generation_join"]
    observation = {"binding": proof["binding"], "native_selected_actor_handle_lookup": True,
                   "generation_role_options_binding": True,
                   "actor": {"name": "door_model_0", "scene": "house_single_floor"}}
    path.write_text(json.dumps({"customs": proof["customs"]["path"],
                               "template_facts": {"prepared_handle_lookup": observation}}))
    monkeypatch.setattr(stream_binding, "bind_one", lambda physical, *args: physical)
    actual, _ = stream_binding.bind_training_candidate(candidate, bound, path, "code", {}, {}, {})
    assert actual.scene_setup["prepared_handle_native_lookup"]["lookup"] == observation
    observation["actor"]["name"] = "another_actor"
    path.write_text(json.dumps({"customs": proof["customs"]["path"],
                               "template_facts": {"prepared_handle_lookup": observation}}))
    with pytest.raises(ValueError, match="actor differs"):
        verify_generated_binding(candidate, bound, path)


def test_hashed_request_rejects_changed_source_job(selected):
    template, job, options = selected
    base = Path(options["output"]).parent
    jobs = document(base / "jobs.json", {"jobs": [job]})
    templates = document(base / "templates.json", {"templates": [asdict(template)]})
    request = document(base / "request.json", {
        "schema": "isg-selected-generation-join-request-v1", "source_jobs": jobs,
        "source_templates": templates, "candidate_id": job["candidate_id"],
        "configuration_id": "config_A", "metadata": options["metadata_ref"],
        "selection": options["selection_ref"], "native_scene": options["native_scene_ref"]})
    manifest = materialize_request(request, options["output"])
    assert Path(manifest["path"]).is_file()
    Path(jobs["path"]).write_text("{}")
    with pytest.raises(ValueError, match="input changed"):
        materialize_request(request, base / "second_join")
