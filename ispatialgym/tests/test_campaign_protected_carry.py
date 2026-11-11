"""The recovery planner must prove whole paths before executing any motion."""
from types import SimpleNamespace as NS

import torch as th

from ispatialgym.oracle import campaign_carry as CC


def setup(monkeypatch, results, *, protected=()):
    calls = []
    q = th.zeros(28)
    start = (th.tensor([0., 0., .9]), th.tensor([0., 0., 0., 1.]))
    end = (th.tensor([.3, 0., .9]), start[1])
    def solve(*args, **kw):
        calls.append(kw)
        return next(results)
    backend = NS(q_full=lambda: q, update_world=lambda **kw: None, solve_ik_path=solve,
                 robot_spheres_world=lambda *args: (None, None))
    cfg = NS(protect_above_m=.12, diff_ik_pos_tol=.005, diff_ik_rot_tol=.05,
             diff_ik_max_iters=30, diff_ik_soft_pos_tol_free=.01, diff_ik_soft_rot_tol=.03)
    motion = NS(backend=backend, cfg=cfg, _protect={'objects': ['partner'], 'inflation': .02},
                eef_pose=lambda arm: start, _densify=lambda start, path: path)
    camera = NS(width=128, height=128, project=lambda *args: (64., 64., 1.))
    monitor = NS(camera=camera, cfg=NS(fov_margin=.05), head_camera_pose=lambda: start)
    monkeypatch.setattr(CC.PROT, 'check_trajectory', lambda *args: list(protected))
    return motion, monitor, end, calls


def result(ok=True):
    return NS(success=ok, q_traj=th.zeros((4, 28)), status='ok' if ok else 'IK stall')


def test_failed_high_rise_is_not_executed_before_checked_lower_path(monkeypatch):
    motions, monitor, end, calls = setup(monkeypatch, iter([result(False), result()]))
    path, facts = CC.choose(motions, 'right', end, ignore=['table'], monitor=monitor)
    assert len(facts) == 2 and facts[-1]['accepted'] and not facts[0]['accepted']
    assert abs(float(path[0][0][2]) - .98) < 1e-6
    assert th.equal(path[-1][0], end[0]) and th.equal(path[-1][1], end[1])
    assert len(calls) == 2 and all(c['check_collisions'] and not c['sync_world'] for c in calls)


def test_near_pass_is_never_promoted_to_candidate(monkeypatch):
    motions, monitor, end, calls = setup(monkeypatch, iter([result() for _ in range(9)]),
                                         protected=[{'object': 'partner', 'clearance_m': .0119}])
    path, facts = CC.choose(motions, 'right', end, ignore=[], monitor=monitor)
    assert path is None and len(calls) == 9
    assert all(f['reason'] == 'protected_object_clearance' for f in facts)


def test_out_of_frame_and_missing_proof_fail_closed(monkeypatch):
    motions, monitor, end, calls = setup(monkeypatch, iter([]))
    monitor.camera.project = lambda *args: (127., 64., 1.)
    assert CC.choose(motions, 'right', end, ignore=[], monitor=monitor)[0] is None
    assert not calls
    motions.backend.robot_spheres_world = None
    path, facts = CC.choose(motions, 'right', end, ignore=[], monitor=monitor)
    assert path is None and 'unavailable' in facts[0]['reason']


def test_protection_errors_do_not_count_as_clear(monkeypatch):
    motions, monitor, end, calls = setup(monkeypatch, iter([result() for _ in range(9)]))
    def broken(*args):
        raise ValueError('no geometry')
    monkeypatch.setattr(CC.PROT, 'check_trajectory', broken)
    path, facts = CC.choose(motions, 'right', end, ignore=[], monitor=monitor)
    assert path is None and len(calls) == 9
    assert all('validation_unavailable' in f['reason'] for f in facts)
