"""Opt-in stationary-obstacle goals for the fresh collection campaign.

The goal gates success; the measure remains camera-only. This module has no
simulator imports so the BDDL/TRO contract can be checked before a GPU launch.
"""

from __future__ import annotations

import copy
import os


def enabled() -> bool:
    return os.environ.get("ISPATIALGYM_CAMPAIGN_OBSTACLE_GOAL") == "1"


def with_stationary_goals(text: str, obstacle_names: list[str]) -> str:
    from bddl.parsing import scan_tokens

    tree = scan_tokens(string=text)
    names = list(dict.fromkeys(obstacle_names))
    if not names:
        raise ValueError("An obstructed task must bind at least one obstacle")
    sections = {section[0]: section for section in tree[1:] if isinstance(section, list) and section}
    if ":goal" not in sections or ":objects" not in sections:
        raise ValueError("Missing BDDL objects or goal section")
    declared = sections[":objects"][1:]
    if any(name not in declared for name in names):
        raise ValueError("Every stationary obstacle must be a declared physical object")
    goal = sections[":goal"]
    if len(goal) != 2:
        raise ValueError("Expected one BDDL goal expression")
    if goal[1][0] != "and":
        goal[1] = ["and", goal[1]]
    for name in names:
        atom = ["pose_reached", "?" + name]
        if atom not in goal[1][1:]:
            goal[1].append(atom)
    # Legacy camera problems without :measure inherit it from :goal. Materialize
    # the original camera-only measure before adding obstacle goal atoms.
    if ":measure" not in sections:
        measure = [":measure", ["pose_reached", "?agent.n.01_1"]]
        tree.append(measure)
    else:
        measure = sections[":measure"]
    if any("?" + name in repr(measure) for name in names):
        raise ValueError("Stationary obstacles must not appear in :measure")

    def render(node):
        return "(" + " ".join(render(x) for x in node) + ")" if isinstance(node, list) else node

    return "(define\n" + "\n".join("    " + render(s) for s in tree[1:]) + "\n)\n"


def initial_pose_targets(poses: dict, names: list[str], *, position_tolerance: float, orientation_tolerance: float) -> dict:
    targets = {}
    for name in names:
        position, orientation = poses[name]
        # Lists as well as tensor stand-ins are accepted; no dependency on torch.
        targets[name] = {
            "position": [float(v) for v in position],
            "orientation": [float(v) for v in orientation],
            "position_metric": "full",
            "orientation_metric": "full",
            "position_tolerance": float(position_tolerance),
            "orientation_tolerance": float(orientation_tolerance),
        }
    return copy.deepcopy(targets)
