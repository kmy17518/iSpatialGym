import json
from dataclasses import replace

import pytest

from scripts.campaign_20261110 import camera_definitions as d,contracts as c


def fixture(tmp_path):
    source=tmp_path/"source"
    source.mkdir()
    text="""(define (problem camera_relocalization-1) (:domain ispatialgym) (:variant camera_relocalization-obstructed-test)
    (:objects agent.n.01_1 - agent.n.01 hamper.n.02_1 - hamper.n.02 floor.n.01_1 - floor.n.01)
    (:init (ontop agent.n.01_1 floor.n.01_1) (ontop hamper.n.02_1 floor.n.01_1))
    (:goal (and (pose_reached ?agent.n.01_1) (pose_settled ?hamper.n.02_1)))
    (:measure (pose_reached ?agent.n.01_1)))"""
    path=source/"problem.bddl";path.write_text(text)
    meta={"activity_definition":"problem.bddl","problem":"camera_relocalization-1","bddl_sha256":c.file_digest(path),
          "bindings":[{"role":"obstacles","name":"hamper.n.02_1"}]}
    c.write_json_atomic(source/"template_meta.json",meta)
    obj=c.ObjectRef("hamper.n.02_1","target","hamper","a",c.digest("asset"),ready=True,readiness_source="test")
    template=c.Template("t","camera_relocalization-obstructed","hamper","eval_standard","house_single_floor","living_room_0",(obj,),c.digest("definition"),
                        {"source_template":str(source),"source_bddl_sha256":c.file_digest(path)})
    return source,template


def test_only_declared_obstacles_gain_goal_atoms_and_measure_is_camera_only(tmp_path):
    source,template=fixture(tmp_path)
    bound=d.bind_obstructed_definition(template,source,output_bddl=tmp_path/"rendered.bddl")
    proof=bound.scene_setup["runtime_bddl_adaptation"]
    assert proof["added_goal_atoms"]==[["pose_reached","?hamper.n.02_1"]]
    assert proof["measure"]==[["pose_reached","?agent.n.01_1"]]
    assert bound.scene_setup["runtime_bddl_sha256"]==c.file_digest(tmp_path/"rendered.bddl")
    assert bound.fingerprint!=template.fingerprint
    meta=json.loads((source/"template_meta.json").read_text());meta["bindings"][0]["name"]="missing.n.01_1"
    c.write_json_atomic(source/"template_meta.json",meta)
    with pytest.raises(c.ContractError,match="bindings differ"):
        d.bind_obstructed_definition(template,source)
