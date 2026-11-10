from dataclasses import asdict, replace

import pytest

from scripts.campaign_20261110 import binding_pools as p, contracts as c, group_lane as lane
from scripts.campaign_20261110.acceptance import audit_job, read
from scripts.campaign_20261110.tests.test_acceptance import fixture, audit
from scripts.campaign_20261110.tests.test_group_lane import reservations


def obj(name, role, category, model, **kwargs):
    return c.ObjectRef(name, role, category, model, c.digest([category, model]),
                       ready=True, readiness_source="frozen-native-ready-audit", **kwargs)


def pool():
    return p.BindingPool("disembedding-pool02", "disembedding_02", "eval_standard",
        "house_single_floor", "room",
        (obj("source.n.01_1", "support", "tray", "source", semantic_kind="tray"),
         obj("destination.n.01_1", "support", "tray", "destination", semantic_kind="tray")),
        (p.PoolCandidate("target01", obj("candidate", "target", "apple", "a")),
         p.PoolCandidate("distractor01", obj("candidate", "distractor", "shampoo", "s"), 9),
         p.PoolCandidate("distractor02", obj("candidate", "distractor", "tissue", "t"), 3)),
        {"target": (1, 1), "distractor": (1, 9)}, c.digest("original maximum-pool definition"))


def variant(root, count=3, maximum=None):
    maximum = maximum or pool()
    path = p.write_pool(root / "manifests/pool.json", maximum)
    choices = {"apple.n.01_1": "target01", **{f"shampoo.n.01_{i+1}": "distractor01" for i in range(count)}}
    candidates = {v.candidate_id: v.object for v in maximum.candidates}
    objects = maximum.fixed_objects + tuple(replace(candidates[key], instance=name) for name,key in choices.items())
    active = c.Template(f"disembedding02_count{count}", "disembedding", maximum.class_id, maximum.split,
        maximum.scene, maximum.room, objects, c.digest(["compiled exact BDDL", count]))
    return p.bind_variant(active, path, choices)


def test_active_templates_only_contain_used_models_with_distinct_physics_keys(tmp_path):
    small, large = variant(tmp_path, 3), variant(tmp_path, 6)
    assert len(small.objects) == 6 and len(large.objects) == 9
    assert all(obj.category != "tissue" for obj in small.objects)
    assert "tissue" not in c.canonical_json(asdict(small))
    assert lane.physical_template(small) != lane.physical_template(large)
    assert small.fingerprint != large.fingerprint
    assert lane.conceptual_template(small) == lane.conceptual_template(large)
    assert any(obj.category == "tissue" for obj in p.audit_objects(small))
    assert p.load_pool_for_template(small).fingerprint == pool().fingerprint


def test_immutable_pool_and_selection_cannot_be_changed(tmp_path):
    active = variant(tmp_path)
    changed = replace(pool(), role_limits={"target": (1, 1), "distractor": (1, 8)})
    with pytest.raises(c.ContractError, match="immutable maximum pool"):
        p.write_pool(tmp_path / "manifests/pool.json", changed)
    raw = asdict(active)
    raw["scene_setup"]["binding_pool"]["active_selection"]["shampoo.n.01_1"] = "distractor02"
    forged = replace(active, scene_setup=raw["scene_setup"])
    with pytest.raises(c.ContractError, match="selection digest"):
        p.load_pool_for_template(forged)
    path = tmp_path / "manifests/pool.json"
    path.write_text(path.read_text()+"\n")
    with pytest.raises(c.ContractError, match="pool bytes changed"):
        p.load_pool_for_template(active)


def test_inactive_candidates_need_readiness_and_global_reservation_clearance(tmp_path):
    maximum = pool()
    with pytest.raises(c.ContractError, match="unready pool candidate"):
        replace(maximum.candidates[-1], object=replace(maximum.candidates[-1].object, ready=False))
    active = variant(tmp_path)
    reserved = {**reservations(), "train_exclude_models": ["tissue/t"]}
    hashes = {"schema":"campaign-reserved-asset-hashes-v1", "reserved_model_keys":["tissue/t"],
        "ready_assets":{}, "reserved_asset_hashes":[], "pending_unready_model_keys":["tissue/t"]}
    with pytest.raises(c.ContractError, match="held-out model"):
        lane.check_reservations(active, reserved, hashes)


@pytest.mark.parametrize("change", ["model", "default_scale", "role", "asset_sha256", "readiness_source"])
def test_active_identity_must_exactly_match_admitted_candidate(tmp_path, change):
    active = variant(tmp_path)
    old = active.objects[-1]
    values = {"model":"unadmitted", "default_scale":(2,2,2), "role":"target",
              "asset_sha256":c.digest("changed bytes"), "readiness_source":"unbound source"}
    changed = replace(active, objects=active.objects[:-1]+(replace(old, **{change:values[change]}),))
    with pytest.raises(c.ContractError, match="differs from maximum pool"):
        p.load_pool_for_template(changed)


def test_count_and_fixed_layout_are_enforced_without_fabricating_coverage(tmp_path):
    active = variant(tmp_path)
    reference = active.scene_setup["binding_pool"]
    with pytest.raises(c.ContractError, match="multiplicity"):
        pool().check_selection({"target":"target01", **{str(i):"distractor01" for i in range(10)}})
    with pytest.raises(c.ContractError, match="only and all active bindings"):
        p.load_pool_for_template(replace(active, objects=active.objects[:-1]))
    fixed = replace(pool(), fixed_layout=True, fixed_selection=reference["active_selection"])
    fixed.check_selection(reference["active_selection"])
    with pytest.raises(c.ContractError, match="fixed puzzle layout"):
        fixed.check_selection({**reference["active_selection"], "extra":"distractor01"})
    # Fixed layout does not waive the requested strict coverage buckets.
    assert lane.bucket_index(active, {"actual_crowdedness":.17385}, strict=True) == 0
    assert lane.bucket_index(active, {"actual_crowdedness":.45}, strict=True) == 2


def pooled_fixture(root, number, count=3, goal=None):
    f = fixture(root, prefix=f"pool{number}_")
    f.template = variant(root, count)
    for path in (f.job / "inst").rglob("*-tro_state.json"):
        old = read(path)
        block = old["ispatialgym"]["campaign"]
        perturb = c.Perturbations({obj.instance:c.ScaleState(obj.default_scale,obj.default_scale) for obj in f.template.objects})
        key = c.ValidationKey(**{**block["validation_key"], "template_sha256":f.template.fingerprint,
                                "scale_sha256":perturb.scale_fingerprint})
        tro = c.attach_tro_metadata({"ispatialgym":{"instance_id":old["ispatialgym"]["instance_id"],
            "object_bindings":{obj.instance:{"category":obj.category,"model":obj.model} for obj in f.template.objects}}},
            campaign_id="campaign", split="eval_standard", template=f.template, created_at=block["created_at"], seed=block["seed"],
            perturbations=perturb, validation_key=key, goal_group_id=block["goal_group_id"])
        tro["ispatialgym"]["campaign"]["runtime"] = block["runtime"]
        group = tro["ispatialgym"]["campaign"]["runtime"]["goal_group"]
        group["requested_goal"] = [0,1,number if goal is None else goal]
        group["actual_crowdedness"] = .17385*count/3
        c.write_json_atomic(path,tro)
        iid = tro["ispatialgym"]["instance_id"]
        c.write_json_atomic(path.with_name(iid+".oracle.json"),{"tro_sha256":c.file_digest(path)})
    gate=read(f.gate);gate["tro_sha256"]=c.file_digest(gate["instance"]);c.write_json_atomic(f.gate,gate)
    return f


def claim(root, f):
    hashes={"schema":"campaign-reserved-asset-hashes-v1","reserved_model_keys":[],"ready_assets":{},
            "reserved_asset_hashes":[],"pending_unready_model_keys":[]}
    return lane.claim_group(audit(f),f.template,reservations(),reservation_hashes=hashes,
        root=root,started_at=f.since,state_path=root/"dataset/quotas.sqlite3")


def test_two_independently_gated_count_variants_share_quota_pool_not_physics_gate(tmp_path):
    small,large=pooled_fixture(tmp_path,1,3),pooled_fixture(tmp_path,2,6)
    with pytest.raises(c.ContractError,match="validation keys"):
        audit_job(large.job,small.gate,root=tmp_path,started_at=small.since,group_closed=True,group_id="pool2_group")
    first,second=claim(tmp_path,small),claim(tmp_path,large)
    assert sorted(first["accepted"].values()) == sorted(second["accepted"].values()) == ["eval_standard","train"]
    assert first["conceptual_template_sha256"] == second["conceptual_template_sha256"]
    assert first["physical_template_sha256"] != second["physical_template_sha256"]
    counts=read(tmp_path/"dataset/group_counts.json")
    assert counts["counts"] == {"eval_standard":2,"train":2}
    assert counts["conceptual_templates"] == 1 and counts["physically_validated_variants"] == 2


def test_pool_provenance_must_remain_inside_campaign(tmp_path):
    active=variant(tmp_path)
    with pytest.raises(c.ContractError,match="outside campaign"):
        p.load_pool_for_template(active,root=tmp_path/"unrelated")


def test_owner_policy_permits_same_goal_in_separate_exactly_gated_count_groups(tmp_path):
    small,large=pooled_fixture(tmp_path,1,3,goal=7),pooled_fixture(tmp_path,2,6,goal=7)
    claim(tmp_path,small)
    result=claim(tmp_path,large)
    assert sorted(result["accepted"].values()) == ["eval_standard","train"]


def test_pool_goal_identity_preserves_actual_scale_and_scene_changes(tmp_path):
    active=variant(tmp_path)
    scales={obj.instance:c.ScaleState(obj.default_scale,obj.default_scale) for obj in active.objects}
    default=p.canonical_pool_perturbations(active,c.Perturbations(scales))
    scales["apple.n.01_1"]=c.ScaleState((1,1,1),(1.1,1.1,1.1))
    perturbed=p.canonical_pool_perturbations(active,c.Perturbations(scales,lighting={"skybox_intensity":2}))
    assert default["scales"] == {}
    assert perturbed["scales"]["candidate:target01"][0]["applied"] == [1.1,1.1,1.1]
    assert perturbed["lighting"] == {"skybox_intensity":2}
