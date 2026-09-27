"""Episode driver and result summary for ``ISpatialGymTask`` environments."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

from ispatialgym import single_env as SE

ISG_INFO_KEY = "ispatialgym"


def isg_info(step_info: Dict[str, Any]) -> Dict[str, Any]:
    """The ``ispatialgym`` block of an ``env.step`` info dict (the task writes it under ``info["done"]``)."""
    if ISG_INFO_KEY in step_info:
        return step_info[ISG_INFO_KEY]
    return step_info["done"][ISG_INFO_KEY]


def summarize_episode(last_info: Dict[str, Any], steps: int, terminated: bool, truncated: bool) -> Dict[str, Any]:
    """Two-level episode result from the last ``env.step`` info dict.

    Keys:
        ``success`` -- official success (binary success, or settled success when ``require_pose_settled``).
        ``binary_success`` / ``first_success_step`` -- level 1.
        ``settled_success`` / ``first_settled_success_step`` / ``settle_latency_steps`` / ``settle_latency_seconds``
        -- level 2 (``-1`` / ``None`` when the goal never settled after the first binary success).
        ``position_error`` / ``orientation_error`` -- max over the pose targets, taken at the first binary success
        or, if it never happened, at the terminal step. ``layout_position_error`` / ``layout_orientation_error`` --
        the same for layout targets (group relations, articulation joints). ``errors`` holds the per-target detail.
    """
    isg = isg_info(last_info)
    return {
        "activity": isg["activity"],
        "activity_definition_id": isg["activity_definition_id"],
        "task_variant": isg["task_variant"],
        "success": bool(isg["official_success"]),
        "require_pose_settled": isg["require_pose_settled"],
        "binary_success": bool(isg["binary_success_ever"]),
        "first_success_step": isg["first_success_step"],
        "settled_success": isg["first_settled_success_step"] >= 0,
        "first_settled_success_step": isg["first_settled_success_step"],
        "settle_latency_steps": isg["settle_latency_steps"],
        "settle_latency_seconds": isg["settle_latency_seconds"],
        "position_error": isg["max_position_error"],
        "orientation_error": isg["max_orientation_error"],
        "layout_position_error": isg["max_layout_position_error"],
        "layout_orientation_error": isg["max_layout_orientation_error"],
        "errors": isg["final_errors"],
        "errors_at_termination": isg["errors"],
        "conditions": isg["conditions"],
        "measured": isg["measured"],
        "group_assignments": isg["group_assignments"],
        "steps": steps,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
    }


#: BEHAVIOR observation convention (evaluation profile ``isg-v1``): one render per control step, so the frames the
#: policy sees lag the physics state by two steps. Demos, goal images and evaluation all use this value; see
#: "Observation timing" in ``docs/task_definition_and_evaluation.md``.
N_RENDER_ITERATIONS = 1


def run_episode(
    env,
    policy: Callable[[Any, Optional[Dict[str, Any]]], Any],
    max_steps: Optional[int] = None,
    reset: bool = True,
    on_step: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    n_render_iterations: int = N_RENDER_ITERATIONS,
) -> Dict[str, Any]:
    """Roll out ``policy(obs, info) -> action`` until the task terminates / truncates (or ``max_steps``).

    Returns :func:`summarize_episode` of the final step. ``on_step(step_index, info)`` is called after every step.
    ``n_render_iterations`` is forwarded to ``env.step`` and defaults to the BEHAVIOR convention
    (:data:`N_RENDER_ITERATIONS`).
    """
    obs = SE.reset(env)[0] if reset else SE.get_obs(env)[0]
    info: Optional[Dict[str, Any]] = None
    steps = 0
    terminated = truncated = False
    while True:
        action = policy(obs, info)
        obs, _, terminated, truncated, info = SE.step(env, action, n_render_iterations=n_render_iterations)
        steps += 1
        if on_step is not None:
            on_step(steps, info)
        if terminated or truncated or (max_steps is not None and steps >= max_steps):
            break
    return summarize_episode(info, steps, terminated, truncated)
