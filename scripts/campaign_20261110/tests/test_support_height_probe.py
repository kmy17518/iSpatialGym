import math

import pytest

from scripts.campaign_20261110.contracts import ContractError
from scripts.campaign_20261110.support_height_probe import corrected_root_height, overlap_depth


def test_bottom_correction_uses_actual_noncentral_geometry():
    points = [[-1, -1, -0.1], [1, 1, 0.9]]
    assert corrected_root_height(points, [0, 0, 0, 1], 0.1, 1.2) == pytest.approx(0.12)
    assert corrected_root_height(points, [0, 0, 0, 1], 0.1, 0.8) == pytest.approx(0.08)
    # Local-z scaling changes no world-z extent for a support rotated 90 degrees about X.
    q = [math.sqrt(0.5), 0, 0, math.sqrt(0.5)]
    assert corrected_root_height(points, q, 1.0, 1.2) == pytest.approx(1.0)


def test_overlapping_boxes_need_overlap_on_every_axis():
    a = [[0, 0, 0], [1, 1, 1]]
    assert overlap_depth(a, [[0.9, 0.2, 0.2], [2, 0.8, 0.8]]) == pytest.approx(0.1)
    assert overlap_depth(a, [[1, 0.2, 0.2], [2, 0.8, 0.8]]) == 0
    assert overlap_depth(a, [[2, 0.2, 0.2], [3, 0.8, 0.8]]) == -1


def test_invalid_scale_and_rotation_cannot_author_scene_variant():
    with pytest.raises(ContractError, match="range"):
        corrected_root_height([[0, 0, 0]], [0, 0, 0, 1], 1, 1.5)
    with pytest.raises(ContractError, match="normalized"):
        corrected_root_height([[0, 0, 0]], [0, 0, 0, 2], 1, 1.1)
