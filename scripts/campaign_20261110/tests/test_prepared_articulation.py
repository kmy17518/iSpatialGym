import copy
import json
from types import SimpleNamespace as N

import pytest

from ispatialgym.sampling.prepared_articulation import (
    KEY, bind_generation, verify_inputs, verify_selected,
)
from scripts.campaign_20261110.contracts import digest, file_digest
from scripts.campaign_20261110.training_bind_articulation import materialize


def document(path, value):
    path.write_text(json.dumps(value))
    return {"path": str(path), "sha256": file_digest(path)}


@pytest.fixture
def case(tmp_path):
    joint = {"name": "reserved_model:joint_j_leaf", "type": "RevoluteJoint",
             "lower_limit": 0., "upper_limit": 1.5}
    choice = {"configuration_id": "config_A", "class_id": "articulation_A",
              "scene": "house_single_floor", "roles": [{"role": "target",
              "model_key": "door/model", "applied_scale": [1., 1., 1.],
              "asset_sha256": "a" * 64, "joint": joint}]}
    selection = document(tmp_path / "selection.json", {"rows": [choice]})
    annotation = document(tmp_path / "handle.json", {"joints": [{"joint": "j_leaf"}]})
    csv = tmp_path / "handles.csv"
    csv.write_text("handle,source\nleaf,annotation\n")
    handle = {"contact": [.02, -.2, .1], "approach_dir": [-1., 0., 0.],
              "pinch_axis": [0., 0., 1.], "width": .05, "kind": "knob"}
    row = {"configuration_id": "config_A", "class_id": "articulation_A",
           "status": "consumed_handle_metadata_resolved", "annotation_lookup_passed": True,
           "scale_row_sha256": digest(choice), "model_key": "door/model",
           "applied_scale": [1., 1., 1.], "asset_sha256": "a" * 64,
           "selected_joint": joint, "consumer_joint": "j_leaf", "handle_key": "door__model",
           "handle_record": annotation, "handle_csv": {"path": str(csv), "sha256": file_digest(csv)},
           "native_selected_handle": {**handle, "source": "door__model:j_leaf#A"},
           "role_options": {"handle_file": annotation["path"], "handle_key": "door__model",
                            "handle_row": "door__model:j_leaf#A", "joint": "j_leaf"},
           "width_range": [.003, .08]}
    metadata = document(tmp_path / "metadata.json", {
        "schema": "isg-exact-articulation-handle-binding-v1", "source": selection, "rows": [row]})
    entry = {"roles": {"target": {"category": "door", "joint": "old_joint"},
                       "floor": {"category": "floors"}},
             "scenes": {"house_single_floor": {"rooms": ["garden_0"]}},
             "generation": {"unchanged": True}}
    options = dict(metadata_ref=metadata, selection_ref=selection,
                   configuration_id="config_A", variant="alias_A")
    bound = bind_generation(entry, **options)
    obj = N(name="door_model_0", category="door", model="model", scale=[1., 1., 1.],
            joints={"j_leaf": N(is_revolute=True, lower_limit=0., upper_limit=1.5)})
    sampler = N(gp=N(entry=bound, variant="alias_A"), scene="house_single_floor",
                target_role="target", target=obj)
    sampler.role_option = lambda k: sampler.gp.entry["roles"]["target"].get(k)
    return N(options=options, entry=entry, bound=bound, obj=obj, sampler=sampler,
             csv=csv, row=row, handle=handle, annotation=annotation, tmp=tmp_path)


def test_generation_join_preserves_original_geometry_and_binds_native_options(case):
    assert case.entry["roles"]["target"]["joint"] == "old_joint"
    assert case.bound["scenes"] == case.entry["scenes"]
    assert case.bound["generation"] == case.entry["generation"]
    assert case.bound["roles"]["target"]["joint"] == "j_leaf"
    original = document(case.tmp / "customs.json", {"alias_A": case.entry, "other": {"untouched": 1}})
    output = case.tmp / "joined.json"
    result = materialize(customs_ref=original, **case.options, output=output)
    assert file_digest(output) == result["sha256"]
    assert json.loads(output.read_text()) == {"alias_A": case.bound, "other": {"untouched": 1}}
    assert list(json.loads(output.read_text())["alias_A"]["roles"]) == ["target", "floor"]
    with pytest.raises(ValueError, match="already exists"):
        materialize(customs_ref=original, **case.options, output=output)


def test_native_observation_requires_the_selected_actor_and_unmodified_handle(case):
    prepared = verify_inputs(case.sampler, case.obj, handles_csv=case.csv)
    observed = verify_selected(prepared, handle=case.handle, source="door__model:j_leaf#A",
                               path=case.annotation["path"], frame="canonical", joint_name="j_leaf",
                               width_range=[.003, .08])
    assert observed["native_selected_actor_handle_lookup"] is True
    assert observed["motion_readiness_proven"] is False
    assert observed["actor"]["name"] == "door_model_0"
    with pytest.raises(ValueError, match="selected handle differs"):
        verify_selected(prepared, handle={**case.handle, "width": .06}, source="door__model:j_leaf#A",
                        path=case.annotation["path"], frame="canonical", joint_name="j_leaf",
                        width_range=[.003, .08])


@pytest.mark.parametrize("mutation", ["model", "scale", "joint", "options", "identity", "variant", "scene"])
def test_different_actor_or_declaration_is_rejected(case, mutation):
    if mutation == "model": case.obj.model = "different"
    if mutation == "scale": case.obj.scale = [1.1, 1., 1.]
    if mutation == "joint": case.obj.joints["j_leaf"].upper_limit = 2.
    if mutation == "options": case.bound["roles"]["target"]["joint"] = "another"
    if mutation == "identity": case.bound["roles"]["target"][KEY]["selected_row_sha256"] = "b" * 64
    if mutation == "variant": case.sampler.gp.variant = "different"
    if mutation == "scene": case.sampler.scene = "hotel_suite_large"
    with pytest.raises(ValueError): verify_inputs(case.sampler, case.obj, handles_csv=case.csv)


def test_changed_or_missing_annotation_cannot_use_native_fallback(case):
    path = case.tmp / "handle.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="input changed"):
        verify_inputs(case.sampler, case.obj, handles_csv=case.csv)
    path.unlink()
    with pytest.raises(FileNotFoundError): verify_inputs(case.sampler, case.obj, handles_csv=case.csv)


def test_blocked_annotation_stays_blocked_even_with_a_valid_new_document_hash(case):
    row = copy.deepcopy(case.row)
    row.update(status="handle_binding_blocked", annotation_lookup_passed=False, reason="pinch too wide")
    metadata = document(case.tmp / "blocked.json", {
        "schema": "isg-exact-articulation-handle-binding-v1", "source": case.options["selection_ref"], "rows": [row]})
    with pytest.raises(ValueError, match="pinch too wide"):
        bind_generation(case.entry, **{**case.options, "metadata_ref": metadata})


def test_legacy_sampler_needs_no_preparation_artifacts():
    sampler = N(gp=N(entry={"roles": {"target": {}}}), target_role="target")
    assert verify_inputs(sampler, object(), handles_csv="/missing/legacy.csv") is None


def test_real_load_handle_records_prepared_lookup_before_approach_repair(case, monkeypatch):
    """Exercise the real CPU consumer with a fake actor, not a native-world proof."""
    from ispatialgym.sampling.v2.tasks import articulation_large as al

    handle = {**case.handle, "kind": "bar", "approach_dir": [0., 0., 1.]}
    annotation = document(case.tmp / "native_handle.json", {"joints": [{
        "joint": "j_leaf", "type": "RevoluteJoint", "handles": [handle],
        "axis_dir_obj": [0., 1., 0.], "hinge_point_obj": [0., 0., 0.]}]})
    row = copy.deepcopy(case.row)
    row.update(handle_record=annotation, native_selected_handle={**handle, "source": "record:bar"})
    row["role_options"].update(handle_file=annotation["path"], handle_row="record:bar")
    metadata = document(case.tmp / "native_metadata.json", {
        "schema": "isg-exact-articulation-handle-binding-v1", "source": case.options["selection_ref"], "rows": [row]})
    entry = bind_generation(case.entry, **{**case.options, "metadata_ref": metadata})

    class Sampler(al.ArticulationLarge):
        @property
        def target(self):
            return case.obj

    sampler = Sampler.__new__(Sampler)
    sampler.gp = N(entry=entry, variant="alias_A", gen=lambda key, default=None: default)
    sampler.target_role = "target"
    sampler.scene = "house_single_floor"
    sampler.world = N(scene_model="house_single_floor")
    sampler.template_facts = {}
    case.obj.links = {}
    monkeypatch.setattr(al, "HANDLES_CSV", str(case.csv))
    sampler.load_handle(case.obj)
    observed = sampler.template_facts["prepared_handle_lookup"]
    assert observed["selected_handle"]["approach_dir"] == [0., 0., 1.]
    assert sampler.handle["approach_dir_recorded"] == [0., 0., 1.]
    assert sampler.handle["approach_dir"] != [0., 0., 1.]
    assert observed["motion_readiness_proven"] is False
