"""Exact-scope start filtering for the held classroom model's bound floor.

The corridor has four distinct floor objects. Room segmentation alone does not
establish the existing ``ontop(agent, bound_floor)`` initial predicate. This
opt-in rejects mismatched proposals before the unchanged collision/settle tests.
"""
import os

ENV = "ISPATIALGYM_HELD_BOUND_FLOOR_START"


def enabled(*, task, category, model, scene, rooms, floor_name, value=None):
    value = os.environ.get(ENV, "0") if value is None else value
    if value in ("", "0"):
        return False
    if value != "1":
        raise ValueError("held bound-floor start switch must be 0 or 1")
    if (task, category, model, scene, tuple(rooms), floor_name) != (
            "mental_rotation", "classroom_mock_up", "isgm10",
            "office_cubicles_right", ("corridor_0",), "floors_egqrip_0"):
        raise ValueError("held bound-floor start recovery is outside its audited native scope")
    return True


def matching_points(points, *, expected_floor, floor_under):
    """Preserve proposal order and require the exact existing floor identity."""
    if expected_floor is None:
        raise ValueError("held bound-floor start requires the bound floor object")
    return [xy for xy in points if floor_under(xy) is expected_floor]
