"""Generate a private selected-actor scene and feed the normal generation join.

One explicitly requested native construction, no allocation, preparation work,
new grasp libraries, map computation or production authorization.
"""
import argparse
import copy
from dataclasses import replace
import json
from pathlib import Path
import sys
import traceback

from ispatialgym.sampling.prepared_articulation import checked
from ispatialgym.sampling.selected_context import construction_plan, construct_and_place, merge_native_actor
from scripts.campaign_20261110.contracts import ObjectRef, Perturbations, ScaleState, Template, digest, require
from scripts.campaign_20261110.group_lane import check_reservations
from scripts.campaign_20261110.reservation_policy import active_policy
from scripts.campaign_20261110.training_selection_join import join_generation, reference


def inputs(request_ref):
    request = checked(request_ref)
    require(request.get('schema') == 'isg-selected-context-generation-request-v1'
            and request.get('production_released') is False,
            'selected construction requires its exact unreleased generation request')
    jobs = checked(request['source_jobs'])['jobs']
    templates = checked(request['source_templates'])['templates']
    jobs = [j for j in jobs if j['candidate_id'] == request['candidate_id']]
    templates = [t for t in templates if t['template_id'] == request['candidate_id']]
    require(len(jobs) == len(templates) == 1, 'selected construction source candidate missing or ambiguous')
    job = jobs[0]; raw = dict(templates[0]); fingerprint = raw.pop('fingerprint', None)
    raw['objects'] = tuple(ObjectRef(**obj) for obj in raw['objects'])
    template = Template(**raw)
    require(template.split == 'train' and job['production_released'] is False
            and (fingerprint is None or fingerprint == template.fingerprint),
            'selected construction template changed or is not training')
    root = Path(request['campaign_root']).resolve()
    policy_path = root / 'manifests/active_reservations.json'
    require(request['reservation_policy'] == reference(policy_path),
            'selected construction active reservation pointer changed')
    active = active_policy(root)
    require(active is not None, 'selected construction has no active train reservation policy')
    check_reservations(template, active[1], active[2])
    plan = construction_plan(metadata_ref=request['metadata'], selection_ref=request['selection'],
        configuration_id=request['configuration_id'], variant=job['alias'], scene_ref=request['source_scene'],
        numeric_ref=request['numeric_geometry'], prototype_name=request['prototype_name'],
        floor_name=request['floor_name'], room=request['placement_room'],
        yaw_radians=request['yaw_radians'], max_candidates=request['max_candidates'])
    targets = [obj for obj in template.objects if obj.role == 'target']
    require(len(targets) == 1 and targets[0].model_key == plan['category'] + '/' + plan['model']
            and targets[0].asset_sha256 == plan['asset_sha256'] and targets[0].joint == plan['joint']
            and tuple(targets[0].default_scale) == tuple(plan['original_default_scale'])
            and template.scene == job['scene'] == plan['scene']
            and template.class_id == job['class_id'] == plan['selection_binding']['class_id']
            and set(template.scene_setup.get('task_rooms', [template.room])) == set(plan['rooms']),
            'selected construction job/template identity differs')
    raw_scales = job.get('training_scale_plan')
    states = ({name: ScaleState(**value) for name, value in raw_scales['scales'].items()}
              if raw_scales is not None else
              {obj.instance: ScaleState(obj.default_scale, obj.default_scale, 'fixed') for obj in template.objects})
    perturbations = Perturbations(states, tuple((raw_scales or {}).get('removed_objects', [])),
                                   (raw_scales or {}).get('lighting', {}))
    perturbations.validate('train', [obj.instance for obj in template.objects])
    require(set(states) == {obj.instance for obj in template.objects}
            and all(states[obj.instance].default == obj.default_scale for obj in template.objects)
            and list(states[targets[0].instance].applied) == plan['applied_scale'],
            'selected construction and normal generation scale plans differ')
    argv = job['sample_argv']
    require(argv[:3] == ['python', '-m', 'ispatialgym.sampling.sample_isg']
            and all(argv.count(flag) == 1 for flag in ('--customs', '--scene', '--out'))
            and argv[argv.index('--scene') + 1] == plan['scene']
            and Path(argv[argv.index('--customs') + 1]).resolve() == Path(request['source_customs']['path']).resolve(),
            'selected construction source sampler command differs')
    customs = checked(request['source_customs'])[job['alias']]
    require(customs['roles']['target']['category'] == plan['category']
            and set(customs['scenes'][plan['scene']]['rooms']) == set(plan['rooms']),
            'selected construction source declaration differs')
    return request, template, job, plan


def join_constructed(request, template, job, plan, context_ref, placement_ref, output):
    """Normal sample_isg remains the resulting job's native entry point."""
    output = Path(output)
    placement = checked(placement_ref)
    require(placement['plan'] == plan and placement['native_scene'] == context_ref
            and placement['original_scene_entries_unchanged'] is True,
            'selected native context lacks matching construction evidence')
    argv = list(job['sample_argv'])
    require(argv[:3] == ['python', '-m', 'ispatialgym.sampling.sample_isg']
            and argv.count('--customs') == 1, 'selected construction requires normal generation input')
    original_customs = reference(argv[argv.index('--customs') + 1])
    require(original_customs == request['source_customs'], 'selected construction source customs changed')
    customs = checked(original_customs)
    entry = customs[job['alias']]
    require(entry['roles']['target']['category'] == plan['category']
            and set(entry['scenes'][plan['scene']]['rooms']) == set(plan['rooms']),
            'construction cannot change the selected generation rooms')
    entry['scenes'][plan['scene']].setdefault('bindings', {})['target'] = [plan['actor_name']]
    private_customs = output / 'constructed_customs.json'
    with private_customs.open('x') as stream:
        stream.write(json.dumps(customs, indent=2) + '\n')
    argv[argv.index('--customs') + 1] = str(private_customs)
    target = next(obj for obj in template.objects if obj.role == 'target')
    setup = copy.deepcopy(template.scene_setup)
    setup.setdefault('inst_to_name', {})[target.instance] = plan['actor_name']
    setup['native_scene_sha256'] = context_ref['sha256']
    setup['selected_context_generation'] = {'plan_sha256': digest(plan), 'placement': placement_ref,
        'source_scene': request['source_scene'], 'source_customs': original_customs,
        'source_template_fingerprint': template.fingerprint, 'source_job_sha256': digest(job),
        'motion_readiness_verified': False}
    constructed = replace(template, scene_setup=setup)
    return join_generation(constructed, {**job, 'sample_argv': argv},
        metadata_ref=request['metadata'], selection_ref=request['selection'],
        configuration_id=request['configuration_id'], native_scene_ref=context_ref, output=output / 'generation')


def run(request_ref, output):
    request, template, job, plan = inputs(request_ref)
    output = Path(output).resolve()
    require(not output.exists(), 'selected context generation output already exists')
    output.mkdir(parents=True)
    __import__('ispatialgym.sampling.robot_utils')
    import omnigibson as og
    from omnigibson.utils.config_utils import TorchEncoder
    from ispatialgym.sampling.sim_utils import SimWorld

    try:
        world = SimWorld(plan['scene'], plan['rooms'],
                         scene_variant={'path': request['source_scene']['path']}, cameras=False)
        obj, metadata, placement = construct_and_place(world, plan)
        require(obj.name == plan['actor_name'], 'native construction returned a different actor')
        live = json.loads(json.dumps(world.scene.save(as_dict=True), cls=TorchEncoder))
        source = checked(request['source_scene'])
        context = merge_native_actor(source, live, plan, metadata)
        stable = output / 'stable_scenes' / (plan['scene'] + '_stable.json')
        stable.parent.mkdir()
        with stable.open('x') as stream:
            stream.write(json.dumps(context, indent=2, allow_nan=False) + '\n')
        proof = output / 'placement.json'
        with proof.open('x') as stream:
            stream.write(json.dumps({'schema': 'isg-selected-context-native-placement-v1',
                'request': request_ref, 'plan': plan, 'native_scene': reference(stable),
                'placement': placement, 'original_scene_entries_unchanged': True,
                'native_reload_verified': False, 'motion_readiness_verified': False,
                'production_released': False}, indent=2, allow_nan=False) + '\n')
        return join_constructed(request, template, job, plan, reference(stable), reference(proof), output)[2]
    except BaseException as error:
        with (output / 'construction_failure.json').open('x') as stream:
            stream.write(json.dumps({'request': request_ref, 'plan': plan,
                'error': repr(error), 'traceback': traceback.format_exc(),
                'placement_attempts': getattr(error, 'placement_attempts', []),
                'production_released': False}, indent=2, allow_nan=False) + '\n')
        raise
    finally:
        og.clear()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('request', 'request-sha256', 'output'):
        parser.add_argument('--' + key, required=True)
    parser.add_argument('--preflight', action='store_true')
    args = parser.parse_args()
    try:
        reference_value = {'path': args.request, 'sha256': args.request_sha256}
        result = ({'plan': inputs(reference_value)[3], 'native_execution': False}
                  if args.preflight else run(reference_value, args.output))
        print(json.dumps(result), flush=True)
    finally:
        native = sys.modules.get('omnigibson')
        if native is not None:
            native.shutdown()
