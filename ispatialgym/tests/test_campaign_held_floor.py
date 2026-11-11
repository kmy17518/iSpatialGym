from types import SimpleNamespace
import pytest
from ispatialgym.sampling import campaign_held_floor as F


SCOPE = dict(task="mental_rotation", category="classroom_mock_up", model="isgm10",
             scene="office_cubicles_right", rooms=["corridor_0"], floor_name="floors_egqrip_0")


def test_disabled_preserves_unrelated_scope():
    assert not F.enabled(**{**SCOPE, "task": "object_scaling"}, value="0")
    assert not F.enabled(**SCOPE, value="")


def test_exact_native_scope_enabled():
    assert F.enabled(**SCOPE, value="1")


@pytest.mark.parametrize("field,value", [("task", "object_scaling"), ("category", "toy_figure"),
    ("model", "other"), ("scene", "house_single_floor"), ("rooms", ["conference_hall_0"]),
    ("floor_name", "floors_fbwhmr_0")])
def test_wrong_scope_fails_closed(field, value):
    with pytest.raises(ValueError, match="audited native scope"):
        F.enabled(**{**SCOPE, field: value}, value="1")


def test_invalid_switch_fails_closed():
    with pytest.raises(ValueError, match="0 or 1"):
        F.enabled(**SCOPE, value="true")


def test_exact_floor_identity_preserves_order_and_rejects_missing_ray():
    expected = SimpleNamespace(name="floor")
    other = SimpleNamespace(name="floor")
    points = [(0, 0), (1, 0), (2, 0), (3, 0)]
    floors = [other, expected, None, expected]
    assert F.matching_points(points, expected_floor=expected,
                             floor_under=lambda xy: floors[xy[0]]) == [points[1], points[3]]


def test_missing_bound_floor_fails_closed():
    with pytest.raises(ValueError, match="bound floor"):
        F.matching_points([], expected_floor=None, floor_under=lambda xy: None)
