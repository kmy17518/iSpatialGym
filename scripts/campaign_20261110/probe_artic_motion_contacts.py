"""Bounded failure diagnosis: observe existing native trials, alter no physics."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def ref(path):
    return {'path': str(path), 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest()}


def run(request):
    data = json.loads(request.read_text())
    assert data['campaign_id'] == 'campaign_20261110' and data['benchmark_credit'] == 0
    for proof in data['inputs']:
        assert ref(proof['path']) == proof
    unit = json.loads(Path(data['wave']['path']).read_text())['units'][data['unit_index']]
    assert unit['id'] == data['unit_id'] and unit['native_runtime'] == data['native_runtime']
    sys.path.insert(0, data['native_runtime'])
    from ispatialgym.grasping import validate_artic as VA, validate_tabletop as VT
    out = Path(data['output']); out.mkdir(exist_ok=False)
    # Every observed step and plan is the original native call. Extra reads
    # record contact identities and IK outcomes; they do not advance physics.
    original_read, original_plan, original_trial = VT.ForceProbe.read, VT.TabletopWorld.plan, VA.MotionWorld._trial
    active = {}
    completed = []

    def contact_read(probe):
        original_read(probe)
        if not active:
            return
        try:
            force = probe.view.get_contact_force_matrix(dt=probe.og.sim.get_physics_dt()).detach().cpu().float()
            active['contact_samples'] += 1
            for row in probe.finger_rows:
                finger = probe.robot_links[row][0]
                for j, (link, _) in enumerate(probe.obj_links):
                    value = float(force[row, probe.obj_col0+j].norm())
                    key = finger + ' -> ' + link
                    previous = active['finger_object_forces'].setdefault(key, {'peak_N': 0., 'positive_samples': 0})
                    previous['peak_N'] = max(previous['peak_N'], value)
                    previous['positive_samples'] += int(value > 1e-5)
        except Exception as exc:
            active['contact_read_error_count'] += 1
            if len(active['contact_read_errors']) < 8:
                active['contact_read_errors'].append(repr(exc))

    def plan(world, obj, geom, candidates, pose):
        result = original_plan(world, obj, geom, candidates, pose)
        if active:
            active['plans'].append({'candidates': candidates,
                                    'object_pose': [v.tolist() for v in pose],
                                    'support_center': list(world.support.centre),
                                    'support_half': list(world.support.half),
                                    'base_yaws': world.cfg.base_yaws,
                                    'base_offsets': world.cfg.base_offsets, 'trunks': world.cfg.trunks,
                                    'native_ik_poses': world.timing.get('plan_ik_poses'),
                                    'results': {k: {n: v for n, v in value.items() if n != 'q'} for k, value in result.items()}})
        return result

    def trial(world, obj, motion, variant, state, rest_sim, geom, from_pre=True):
        active.clear()
        active.update(variant=variant['name'], joint=motion['joint'], q_start=motion['q_start'],
                      contact_samples=0, contact_read_error_count=0, contact_read_errors=[], finger_object_forces={}, plans=[])
        result = original_trial(world, obj, motion, variant, state, rest_sim, geom, from_pre)
        active['native_stop'] = result.get('stop')
        active['native_start'] = result.get('start')
        active['final_joint_positions'] = obj.get_joint_positions().detach().cpu().tolist()
        active['joint_names'] = list(obj.joints)
        active['ag_target_held'] = world.w.robot._ag_obj_in_hand.get(world.w.cfg.arm) is obj
        completed.append(json.loads(json.dumps(active)))
        (out / 'trial_observations.json').write_text(json.dumps({'campaign_id': data['campaign_id'],
            'request': ref(request), 'observed_at': time.time(), 'trials': completed,
            'benchmark_credit': 0, 'production_authorization': False}, indent=2)+'\n')
        active.clear()
        return result

    VT.ForceProbe.read, VT.TabletopWorld.plan, VA.MotionWorld._trial = contact_read, plan, trial
    VA.main(['motions', '--aliases', unit['alias'], '--customs', unit['customs']['path'],
             '--root', str(out / 'diagnostic_libraries'), '--arm', unit['identity']['arm'],
             '--max-trials', '2', '--q-starts', 'motion'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--request', type=Path, required=True)
    run(parser.parse_args().request)
