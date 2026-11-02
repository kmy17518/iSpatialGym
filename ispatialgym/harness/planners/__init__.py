"""Planners: ``l3_tools`` (RoboDojo-style L3 tool agent). ``capx`` (a code agent on ``core.session.RobotAPI``) is the
planned second planner; see the package README."""

from __future__ import annotations

from ispatialgym.harness.planners.base import EpisodeContext, Planner  # noqa: F401


def make_planner(name: str, **kw) -> Planner:
    if name == "l3_tools":
        from ispatialgym.harness.planners.l3_tools.agent import L3ToolAgent

        return L3ToolAgent(**kw)
    if name == "capx":
        raise NotImplementedError("the CaP-X code agent is not implemented yet (core.session.RobotAPI / ThreadBridge are its robot interface)")
    raise ValueError(f"unknown planner {name!r}")
