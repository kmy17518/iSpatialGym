"""Release-owned class routing, shared by generation, collectors and replay.

The checked-in recipe registry is the selector. Recorded instances carry their
resolved profile and inputs; replay never consults a live campaign config or
plans another demonstration. Unselected classes keep their existing path.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, is_dataclass

from scripts.campaign_20261110.contracts import (
    ObjectRef, Perturbations, ScaleState, Template, digest, require,
)
from ispatialgym.sampling import campaign_class_recipes as recipes

SCHEMA = "isg-persistent-class-route-v1"


def resolve(config):
    profile = recipes.profile_for_config(config)
    if profile is None:
        return None
    data = dict(config["template"])
    data["objects"] = tuple(ObjectRef(**obj) for obj in data["objects"])
    template = Template(**data)
    p = config["perturbations"]
    perturbations = Perturbations({k: ScaleState(**v) for k, v in p["scales"].items()},
                                 tuple(p.get("removed_objects", [])), p.get("lighting", {}))
    require(template.fingerprint == config["validation_key"]["template_sha256"],
            "class route decoded template fingerprint changed")
    require(perturbations.scale_fingerprint == config["validation_key"]["scale_sha256"],
            "class route decoded scale fingerprint changed")
    return profile


def rank(config, plan, order, candidates, *, written=(), trace):
    """Return None only for an unselected class, never an incompatible profile."""
    profile = resolve(config)
    if profile is None:
        return None
    route_id = profile["route_id"]
    if route_id == "attachment03_retained_source_stance_v1":
        return recipes.attachment03_order_stances(route_id, config, plan, order, candidates, trace=trace)
    if route_id == "under_small03_retained_source_stance_v1":
        return recipes.under_small03_order_stances(route_id, config, plan, order, candidates,
                                                  written=written, trace=trace)
    raise ValueError("required class route has no implementation")


def record(config, plan, trace):
    profile = resolve(config)
    if profile is None:
        return None
    require(plan is not None, "required class route has no instance goal plan")
    events = [copy.deepcopy(e) for e in trace if e.get("plan_sha256") == getattr(plan, "sha256", None)]
    require(any(e.get("action") == "source_stance_ranked" and e.get("route_id") == profile["route_id"]
                for e in events), "selected class strategy did not execute for this instance")
    plan_data = asdict(plan) if is_dataclass(plan) else copy.deepcopy(vars(plan))
    value = {"schema": SCHEMA, "route_id": profile["route_id"], "version": profile["version"],
             "profile": profile, "profile_sha256": digest(profile),
             "configuration": copy.deepcopy(config), "configuration_sha256": digest(config),
             "plan": plan_data, "plan_record_sha256": digest(plan_data), "trace": events}
    value["record_sha256"] = digest(value)
    return value


def verify_recorded(meta, *, phase="replay", physical=None):
    """Verify source routing without changing legacy provenance or actions."""
    recorded = meta.get("runtime", {}).get("class_route")
    if recorded is None:
        return None  # Legacy source remains legacy, even on a newer runtime.
    require(recorded.get("schema") == SCHEMA, "unknown source class route schema")
    payload = {k: v for k, v in recorded.items() if k != "record_sha256"}
    require(digest(payload) == recorded.get("record_sha256"), "source class route record changed")
    config = recorded["configuration"]
    profile = resolve(config)
    require(profile is not None and profile == recorded["profile"] and
            digest(profile) == recorded["profile_sha256"] and
            profile["route_id"] == recorded["route_id"] and profile["version"] == recorded["version"],
            "source route profile differs from release registry")
    require(digest(config) == recorded["configuration_sha256"] and
            digest(recorded["plan"]) == recorded["plan_record_sha256"], "source route inputs changed")
    require(config["validation_key"] == meta["validation_key"] and
            config["perturbations"] == meta["perturbations"] and
            config["template"]["scene_setup"] == meta["scene_setup"] and
            config["template"]["class_id"] == meta["class_id"] and config["split"] == meta["split"],
            "source route is bound to another instance configuration")
    require(recorded["plan"]["group_id"] == meta["goal_group_id"], "source route goal group changed")
    require(any(e.get("action") == "source_stance_ranked" and
                e.get("route_id") == profile["route_id"] and
                e.get("configuration_sha256") == recorded["configuration_sha256"] and
                e.get("plan_sha256") == recorded["plan"]["sha256"] for e in recorded["trace"]),
            "source route lacks its sampling execution trace")
    if physical is not None:
        for obj in config["template"]["objects"]:
            live = physical.get(obj["instance"])
            require(live is not None and (live.category, live.model) == (obj["category"], obj["model"]),
                    "source route live model binding changed")
    trace = []
    recipes.validate_recipe_inputs(profile["route_id"], config, recorded["plan"], phase=phase, trace=trace)
    return {"route_id": profile["route_id"], "version": profile["version"],
            "source_record_sha256": recorded["record_sha256"], "trace": trace}


def execution_policy(config):
    profile = resolve(config)
    if profile is None:
        return None
    return {"route_id": profile["route_id"], "version": profile["version"],
            "max_attempts": profile["historical_max_draws"], "profile_sha256": digest(profile)}


def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    policy = execution_policy(json.loads(Path(args.config).read_text()))
    Path(args.output).write_text(json.dumps(policy, sort_keys=True) + "\n")
    if policy is not None:
        print(int(policy["max_attempts"]))


if __name__ == "__main__":
    main()
