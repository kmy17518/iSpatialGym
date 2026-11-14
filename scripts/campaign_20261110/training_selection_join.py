"""Join an exact selected articulation to the normal training generation job.

The caller supplies an existing native scene context and an asset-ready
candidate. Missing scene actors are an explicit generation blocker. This
module does not invent placement, run preparation, or authorize allocation.
"""
from __future__ import annotations

import copy
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from ispatialgym.sampling.prepared_articulation import KEY, binding, checked
from scripts.campaign_20261110.bind_fresh_candidates import native_scale_equal
from scripts.campaign_20261110.contracts import (
    ObjectRef, Template, Perturbations, ScaleState, digest, file_digest, require,
)
from scripts.campaign_20261110.training_bind_articulation import materialize


def reference(path):
    return {"path": str(Path(path).resolve()), "sha256": file_digest(path)}


def join_generation(template, job, *, metadata_ref, selection_ref, configuration_id,
                    native_scene_ref, output):
    """Publish a private job/contract consumed by the existing training lane."""
    value, row, choice = binding(metadata_ref, selection_ref, configuration_id, job["alias"])
    require(template.split == "train" and job.get("production_released") is False,
            "selected generation join requires an unreleased training candidate")
    require(template.template_id == job["candidate_id"]
            and template.class_id == job["class_id"] == row["class_id"]
            and template.scene == job["scene"] == choice["scene"],
            "selected generation candidate class or scene differs")
    rooms = template.scene_setup.get("task_rooms", [template.room])
    require(set(rooms) == set(choice["rooms"]), "selected generation rooms differ")
    targets = [obj for obj in template.objects if obj.role == "target"]
    require(len(targets) == 1, "selected articulation requires exactly one target")
    target = targets[0]
    require(target.model_key == row["model_key"] and target.asset_sha256 == row["asset_sha256"]
            and target.joint == row["consumer_joint"], "selected generation target differs")
    plan = copy.deepcopy(job.get("training_scale_plan"))
    if plan is None:
        plan = asdict(Perturbations({obj.instance: ScaleState(obj.default_scale, obj.default_scale, "fixed")
                                    for obj in template.objects}))
    states = {name: ScaleState(**item) for name, item in plan["scales"].items()}
    physical = Perturbations(states, tuple(plan.get("removed_objects", [])), plan.get("lighting", {}))
    physical.validate("train", [obj.instance for obj in template.objects])
    require(set(states) == {obj.instance for obj in template.objects}
            and all(states[obj.instance].default == obj.default_scale for obj in template.objects)
            and list(states[target.instance].applied) == row["applied_scale"],
            "selected generation applied/default scale plan differs")

    scene = checked(native_scene_ref)
    require(Path(native_scene_ref["path"]).name == template.scene + "_stable.json",
            "selected context is not a native stable scene")
    require(template.scene_setup.get("native_scene_sha256") == native_scene_ref["sha256"],
            "selected scene context differs from the candidate")
    name = template.scene_setup.get("inst_to_name", {}).get(target.instance)
    actor = row.get("scene_actor_binding")
    if actor is not None:
        require(native_scene_ref == actor["source"] and name == actor["name"],
                "selected instance handle cannot change scene context or actor name")
    init = scene.get("objects_info", {}).get("init_info", {}).get(name, {}).get("args")
    require(init is not None, "selected target has no exact native scene actor; placement is unresolved")
    require((init.get("category"), init.get("model")) == (target.category, target.model)
            and native_scale_equal(row["applied_scale"], init.get("scale")),
            "native context actor model or scale differs from selection")

    argv = list(job["sample_argv"])
    require(argv[:3] == ["python", "-m", "ispatialgym.sampling.sample_isg"]
            and argv.count("--customs") == 1 and argv.count("--scene") == 1 and argv.count("--out") == 1
            and argv[argv.index("--scene") + 1] == template.scene,
            "selected generation must use the normal exact-scene sampler entry point")
    customs_ref = reference(argv[argv.index("--customs") + 1])
    entry = checked(customs_ref)[job["alias"]]
    scene_block = entry.get("scenes", {}).get(template.scene, {})
    require(scene_block.get("bindings", {}).get("target") == [name]
            and set(scene_block.get("rooms", [])) == set(rooms),
            "generation declaration is not bound to the selected native actor and rooms")
    output = Path(output).resolve()
    require(not output.exists(), "selected generation output already exists")
    output.mkdir(parents=True)
    customs = materialize(customs_ref=customs_ref, metadata_ref=metadata_ref,
                          selection_ref=selection_ref, configuration_id=configuration_id,
                          variant=job["alias"], output=output / "customs.json")
    proof = {"schema": "isg-selected-generation-join-v1", "configuration_id": configuration_id,
             "binding": value, "source_customs": customs_ref, "customs": customs,
             "native_scene": native_scene_ref, "native_actor_name": name,
             "source_job_sha256": digest(job),
             "source_template_fingerprint": template.fingerprint,
             "native_actor_lookup_verified": False, "production_released": False}
    setup = {**template.scene_setup, "prepared_handle_binding": value,
             "selected_generation_join": proof, "training_scale_plan_sha256": digest(plan)}
    candidate_id = "train_selected_" + digest({"source": template.fingerprint,
                   "selection": value, "native_scene": native_scene_ref})[:20]
    candidate = replace(template, template_id=candidate_id,
                        definition_sha256=digest({"source": template.definition_sha256,
                        "customs": customs, "selection": value}), scene_setup=setup)
    argv[argv.index("--customs") + 1] = customs["path"]
    argv[argv.index("--out") + 1] = str(output / "sampling")
    bound_job = {**copy.deepcopy(job), "candidate_id": candidate_id,
                 "sample_argv": argv, "training_scale_plan": plan,
                 "selected_generation_join": proof, "template_fingerprint": candidate.fingerprint,
                 "source_manifest": str(output / "contract_candidates.json"),
                 "environment": {**job.get("environment", {}),
                    "ISPATIALGYM_STABLE_SCENE_DIR": str(Path(native_scene_ref["path"]).parent)}}
    for filename, document in (("contract_candidates.json", {"templates": [asdict(candidate)]}),
                               ("validation_jobs.json", {"jobs": [bound_job], "production_released": False}),
                               ("selection_join.json", proof)):
        with (output / filename).open("x") as stream:
            stream.write(json.dumps(document, indent=2, allow_nan=False) + "\n")
    return candidate, bound_job, reference(output / "validation_jobs.json")


def verify_generated_binding(template, job, metadata_path):
    """Require the sampler's actual lookup before the annotated training step."""
    declared = template.scene_setup.get("selected_generation_join")
    if declared is None:
        require("selected_generation_join" not in job, "job join has no matching template binding")
        return None
    require(job.get("selected_generation_join") == declared, "job/template selected join differs")
    metadata = json.loads(Path(metadata_path).read_text())
    observed = metadata.get("template_facts", {}).get("prepared_handle_lookup")
    require(isinstance(observed, dict) and observed.get("binding") == declared["binding"]
            and observed.get("native_selected_actor_handle_lookup") is True
            and observed.get("generation_role_options_binding") is True,
            "native generation has no exact prepared handle lookup evidence")
    require(observed["actor"]["name"] == declared["native_actor_name"]
            and observed["actor"]["scene"] == template.scene,
            "native generated actor differs from selected context")
    customs = checked(declared["customs"])
    require(customs[job["alias"]]["roles"]["target"][KEY] == declared["binding"]
            and Path(metadata["customs"]).resolve() == Path(declared["customs"]["path"]).resolve(),
            "native generation used a different prepared declaration")
    return {"metadata": reference(metadata_path), "lookup": observed,
            "selected_row_sha256": declared["binding"]["selected_row_sha256"]}


def materialize_request(request_ref, output):
    request = checked(request_ref)
    require(request.get("schema") == "isg-selected-generation-join-request-v1",
            "unsupported selected generation join request")
    jobs = checked(request["source_jobs"])["jobs"]
    matches = [job for job in jobs if job["candidate_id"] == request["candidate_id"]]
    templates = checked(request["source_templates"])["templates"]
    candidates = [row for row in templates if row["template_id"] == request["candidate_id"]]
    require(len(matches) == len(candidates) == 1, "selected source job/template missing or ambiguous")
    raw = dict(candidates[0])
    fingerprint = raw.pop("fingerprint", None)
    raw["objects"] = tuple(ObjectRef(**obj) for obj in raw["objects"])
    template = Template(**raw)
    require(fingerprint is None or fingerprint == template.fingerprint, "selected source template changed")
    _, _, manifest = join_generation(template, matches[0], metadata_ref=request["metadata"],
        selection_ref=request["selection"], configuration_id=request["configuration_id"],
        native_scene_ref=request["native_scene"], output=output)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("request", "request-sha256", "output"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    print(json.dumps(materialize_request({"path": args.request, "sha256": args.request_sha256}, args.output)))
