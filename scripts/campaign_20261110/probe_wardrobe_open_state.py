"""Validate a state-only private wardrobe scene revision with native Open checks."""
import argparse
import copy
import json
from pathlib import Path
import sys
import traceback

from scripts.campaign_20261110.contracts import file_digest, write_json_atomic


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scene-probe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    proof = json.loads(args.scene_probe.read_text())
    source_path = Path(proof['native_scene_path'])
    assert proof['status'] == 'private_context_geometry_and_reload_passed'
    assert file_digest(source_path) == proof['native_scene_sha256']
    row = next(x for x in proof['objects'] if x['model_key'] == 'wardrobe/brimns')
    source = json.loads(source_path.read_text())
    name = row['name']
    from ispatialgym.sampling.knowledge import abilities_of_category
    abilities = abilities_of_category('wardrobe')
    assert 'openable' not in abilities, 'The expected missing taxonomy state has changed'
    updated = copy.deepcopy(source)
    init = updated['objects_info']['init_info'][name]['args']
    assert 'abilities' not in init and init['category'] == 'wardrobe' and init['model'] == 'brimns'
    init['abilities'] = {**abilities, 'openable': {}}
    check = copy.deepcopy(updated)
    del check['objects_info']['init_info'][name]['args']['abilities']
    assert check == source, 'Only the explicit Open-state ability may change'
    args.output.mkdir(parents=True, exist_ok=False)
    stable = args.output / 'stable_scenes' / source_path.name
    write_json_atomic(stable, updated)
    report = dict(schema='campaign-wardrobe-open-state-v1', status='started',
        source_probe=str(args.scene_probe), source_probe_sha256=file_digest(args.scene_probe),
        source_scene_sha256=file_digest(source_path), native_scene_path=str(stable),
        native_scene_sha256=file_digest(stable), object=name, abilities=init['abilities'],
        only_change='explicit openable ability; all geometry, poses, joint states and physics preserved',
        production_released=False, checks=[])
    __import__('ispatialgym.sampling.robot_utils')
    import omnigibson as og
    import torch as th
    from omnigibson.object_states import Open
    from ispatialgym.sampling.sim_utils import SimWorld
    try:
        world = SimWorld(proof['scene'], [proof['room']], scene_variant={'path': str(stable)}, cameras=False)
        obj = world.obj(name)
        assert Open in obj.states
        assert max(abs(float(x)-y) for x,y in zip(obj.scale,row['default_scale'])) < 1e-6
        position, orientation = obj.get_position_orientation()
        assert float(th.linalg.norm(position-th.tensor(row['position']))) < .004
        assert abs(float(th.dot(orientation, th.tensor(row['orientation'])))) > .99999
        joints = list(obj.joints.values())
        assert {j.joint_name for j in joints} == {'j_leftdoor', 'j_rightdoor'}
        closed = th.tensor([j.lower_limit for j in joints])
        for target in [None, *range(len(joints))]:
            values = closed.clone()
            if target is not None:
                values[target] = joints[target].lower_limit + .5*(joints[target].upper_limit-joints[target].lower_limit)
            obj.set_joint_positions(values); obj.keep_still()
            for _ in range(3):
                world.step()
                world.keep_still(world.scene.objects)
            actual = obj.get_joint_positions()
            assert float(th.max(th.abs(actual-values))) < .01
            observed = bool(obj.states[Open].get_value())
            assert observed == (target is not None), 'Native Open state disagrees with measured door joints'
            report['checks'].append(dict(joint=None if target is None else joints[target].joint_name,
                expected_open=target is not None, observed_open=observed, joint_positions=actual.tolist()))
        obj.set_joint_positions(closed); obj.keep_still()
        assert file_digest(source_path) == proof['native_scene_sha256']
        report['status'] = 'native_open_state_passed'
        new_proof = copy.deepcopy(proof)
        new_proof.update(native_scene_path=str(stable),native_scene_sha256=file_digest(stable),
            state_only_revision=report.copy(), prior_scene_probe_sha256=file_digest(args.scene_probe))
        write_json_atomic(args.output/'context_scene_probe.json',new_proof)
    except BaseException as exc:
        report.update(status='failed', error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        write_json_atomic(args.output/'open_state_probe.json', report)
        sys.stdout.flush(); sys.stderr.flush(); og.shutdown()


if __name__ == '__main__':
    main()
