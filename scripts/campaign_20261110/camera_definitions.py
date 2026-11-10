"""Author camera obstacle gating without changing the camera measurement."""
from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from pathlib import Path

from scripts.campaign_20261110.contracts import digest, require
from scripts.campaign_20261110.goal_groups import bind_runtime_definition, read_json


def bind_obstructed_definition(template, runtime_template_dir, *, output_bddl=None):
    """Predict precisely the opt-in sampler's BDDL and prove its only adaptation."""
    from ispatialgym.activity.problem import parse_problem_string
    from ispatialgym.sampling.campaign_obstacles import with_stationary_goals

    require(template.task == "camera_relocalization-obstructed", "stationary goal adaptation is obstructed-camera-only")
    template = bind_runtime_definition(template, runtime_template_dir)
    meta = read_json(Path(runtime_template_dir)/"template_meta.json")
    obstacles = [row["name"] for row in meta.get("bindings", []) if row.get("role") == "obstacles"]
    require(obstacles and len(obstacles) == len(set(obstacles)), "missing or duplicated obstacle bindings")
    require(set(obstacles) == {obj.instance for obj in template.objects if obj.role == "target"},
            "obstacle goal bindings differ from ready contract targets")
    source_path = Path(template.scene_setup["runtime_bddl_equivalence"]["runtime"]["path"])
    original = source_path.read_text()
    rendered = with_stationary_goals(original, obstacles)
    problem = re.fullmatch(r"(.+)-(\d+)", meta["problem"])
    require(problem is not None, "invalid runtime camera definition identity")
    before, after = [parse_problem_string(text,problem[1],int(problem[2])) for text in (original,rendered)]
    for field in ("activity","domain","variant","objects","groups","init","measure"):
        require(getattr(before,field) == getattr(after,field), f"obstacle adaptation changed {field}")
    atoms = [["pose_reached","?"+name] for name in obstacles]
    added = [atom for atom in atoms if atom not in before.goal]
    require(after.goal == before.goal + added, "obstacle adaptation changed another gating atom")
    require(not any(atom in after.measure for atom in atoms), "obstacle poses must never become camera measures")
    sha = hashlib.sha256(rendered.encode()).hexdigest()
    proof = {"schema":"isg-stationary-obstacle-definition-v1","semantic_equivalence":False,
             "requested_change":"append declared stationary-obstacle pose goals; retain camera-only measurement",
             "base_bddl_sha256":template.scene_setup["runtime_bddl_sha256"],"runtime_bddl_sha256":sha,
             "obstacle_bindings":obstacles,"added_goal_atoms":added,"measure":after.measure,"measure_sha256":digest(after.measure)}
    if output_bddl is not None:
        path=Path(output_bddl)
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists():
            require(path.read_text()==rendered,"authored obstacle BDDL already has different content")
        else:
            path.write_text(rendered)
        proof["authored_bddl_path"]=str(path.resolve())
    return replace(template,scene_setup={**template.scene_setup,"runtime_bddl_sha256":sha,"runtime_bddl_adaptation":proof})
