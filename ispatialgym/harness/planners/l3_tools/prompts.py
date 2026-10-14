"""System message and Goal turn of the L3 agent (RoboProbe Inspect EEF, adapted to R1Pro)."""

from __future__ import annotations

from typing import Optional

from ispatialgym.harness.config import HarnessConfig


def system_message(cfg: HarnessConfig, embodiment_docs: str) -> str:
    return (
        "You are controlling a real robot embodiment named 'ispatialgym-r1pro'. You receive RGB camera images, a goal image of the head "
        "camera's view once the task is done, the current base-frame grasp-point state of both arms in the same 14 dimensions move_eef "
        "takes, the base and torso state, arm joint angles as context you cannot command, and a task instruction. Move the arms with "
        "move_eef by naming only the base-frame dimensions you want to change, and the base and the torso with move_base and move_torso. "
        "Cartesian targets must be estimated from RGB; no depth or world-coordinate query is available. Respond with exactly one tool "
        "call per turn. After each motion the next observation reports how far the grasp point, the base or the torso ended up from what "
        "you asked for, so check it before assuming a motion landed. Two budgets run down at once and whichever empties first ends the "
        f"episode: {cfg.max_calls} LLM calls, one per turn, and the environment's own step limit, reported with each observation as the "
        "env steps remaining. A motion spends env steps in proportion to how far it travels, so a small correction is cheap and a long "
        "reach is not."
        f"\n\nEmbodiment notes:\n{embodiment_docs.strip()}"
    )


def goal_text(instruction: str, recipe: Optional[str] = None) -> str:
    goal = f"Goal: {instruction or ''}"
    return f"{goal}\n\nTASK RECIPE:\n{recipe.rstrip()}" if recipe else goal
