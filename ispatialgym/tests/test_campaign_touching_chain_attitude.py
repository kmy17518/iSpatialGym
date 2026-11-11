"""A fixed drop must screen contact at its release attitude, before yaw snapping."""
from types import SimpleNamespace

import pytest
import torch

from ispatialgym.sampling.v2.tasks.iors_large_touching import TouchingMode
from ispatialgym.sampling import geometry as G


class NativeChain:
    def _push_under_chain(self, place, quat, drop_xy, goal_xy, direction):
        self.native_checks.append((quat.clone(), goal_xy.clone()))
        return {'native_passed': True} if self.native_feasible else None


class Candidate(TouchingMode, NativeChain):
    def __init__(self, active=True):
        self.hull = torch.tensor([[x, y, z] for x in [-.2, .2] for y in [-.1, .1] for z in [0., .1]])
        self.gp = SimpleNamespace(gen=lambda key, default=None: default)
        self.seg_arm = 'right'
        self.place = {}
        self.native_feasible = True
        self.native_checks = []
        self.fixed = G.quat_from_yaw(torch.pi/2)
        self.face = torch.tensor([1., 0.])
        self.goal = self.touching_goal_at(self.face, [1., 0.], self.fixed)
        self.drop = self.goal - torch.tensor([.3, 0.])
        self._campaign_pair_active = SimpleNamespace(placement_mode='drop_push',
            release=([*self.goal, 0.], self.fixed), place=([*self.drop, .15], self.fixed),
            goal_cell=self.goal) if active else None

    @property
    def touching_gap(self):
        return .002

    @property
    def blade_window(self):
        return (.4, .85)

    def blade_span_at(self, *args):
        return None


def stance(candidate):
    return SimpleNamespace(extras={'face_point':candidate.face}, base_pose2d=(0., 0., 0.),
                           translate=.95, tilt=-1.2, idle_posture='untucked')


def test_fixed_drop_screens_native_contact_at_audited_attitude_before_draw_snapping():
    c = Candidate()
    # This coarse feasibility-grid yaw is snapped by the later native draw.
    provisional = G.quat_from_yaw(.15)
    result = c._push_under_chain(stance(c), c._push_chain_screen_quat(provisional), c.drop, c.goal, [1., 0.])
    assert result['native_passed']
    assert torch.allclose(c.native_checks[0][0], c.fixed)
    assert torch.allclose(c.native_checks[0][1], c.goal)


def test_uncampaigned_native_chain_keeps_its_input_attitude():
    c = Candidate(active=False)
    provisional = G.quat_from_yaw(.15)
    result = c._push_under_chain(stance(c), c._push_chain_screen_quat(provisional), c.drop, c.goal, [1., 0.])
    assert result['native_passed']
    assert torch.equal(c.native_checks[0][0], provisional)


@pytest.mark.parametrize('failure', ['native_ik', 'drop', 'endpoint', 'release_attitude'])
def test_fixed_drop_retains_native_ik_and_exact_geometry_rejections(failure):
    c = Candidate()
    if failure == 'native_ik': c.native_feasible = False
    if failure == 'drop': c.drop = c.drop + torch.tensor([.006, 0.])
    if failure == 'endpoint': c._campaign_pair_active.goal_cell = c.goal + torch.tensor([.002, 0.])
    if failure == 'release_attitude': c._campaign_pair_active.release = ([*c.goal, 0.], G.quat_from_yaw(0.))
    assert c._push_under_chain(stance(c), c._push_chain_screen_quat(G.quat_from_yaw(.15)), c.drop, c.goal, [1., 0.]) is None


def test_final_native_draw_keeps_strict_fixed_attitude_gate():
    c = Candidate()
    assert c._push_under_chain(stance(c), G.quat_from_yaw(.15), c.drop, c.goal, [1., 0.]) is None
    assert c.native_checks == []
