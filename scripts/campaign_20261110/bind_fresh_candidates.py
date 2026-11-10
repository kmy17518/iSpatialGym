#!/usr/bin/env python3
"""Bind successful consumer templates to exact candidate, asset, and split proofs.

Each invocation writes a new immutable batch. Geometry failures are reported
separately; neither a positive grasp library nor a written TRO releases a task.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.campaign_20261110.contracts import ObjectRef, Template, digest, require
from scripts.campaign_20261110.group_lane import check_reservations
from scripts.campaign_20261110.reservation_policy import active_policy
from scripts.campaign_20261110.binding_pools import load_pool_for_template


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contract(row):
    data = dict(row)
    require(not data.pop('blockers', []), 'candidate has unresolved blockers')
    fingerprint = data.pop('fingerprint')
    data['objects'] = tuple(ObjectRef(**o) for o in data['objects'])
    template = Template(**data)
    require(template.fingerprint == fingerprint, 'candidate fingerprint mismatch')
    return template


def role_for(binding):
    return {'target': 'target', 'obstacles': 'target', 'partner': 'partner', 'distractors': 'distractor'}.get(binding['role'], 'support')


def asset_path(code, init):
    category, model = init['category'], init['model']
    dataset = init.get('dataset_name') or 'behavior-1k-assets'
    root = code / 'ispatialgym/assets/datasets' / dataset if dataset == 'ispatialgym-assets' else code / 'Behavior-1K/datasets' / dataset
    return root / 'objects' / category / model / 'usd' / (model + ('.encrypted.usd' if dataset == 'behavior-1k-assets' else '.usd'))


def bind_one(template, job, metadata_path, code, reservations, hashes, policy_pointer):
    # This also verifies immutable maximum-pool bytes and exact active binding
    # names before native metadata can replace any candidate fields.
    load_pool_for_template(template)
    meta = json.loads(metadata_path.read_text())
    directory = metadata_path.parent
    full = directory / 'template.json'
    require(meta.get('n_written', 0) > 0, 'no successful native sample')
    require(sha(full) == meta['template_sha256'], 'native template SHA mismatch')
    require(meta['scene'] == template.scene and set(meta['task_rooms']) == set(template.scene_setup.get('task_rooms', [template.room])), 'native scene/rooms differ from candidate')
    require(meta['variant'] == job['alias'], 'native variant differs from candidate')
    require(meta['bddl_sha256'] == template.scene_setup['runtime_bddl_sha256'], 'native BDDL semantics or definition changed')
    matches = [p for p in (directory / 'activity_definitions').rglob('*.bddl') if sha(p) == meta['bddl_sha256']]
    require(len(matches) == 1, 'exact native BDDL bytes absent or ambiguous')
    tros = sorted((directory / 'instances').glob('*-tro_state.json'))
    require(bool(tros), 'metadata reports success without a fresh TRO')
    state = json.loads(full.read_text())
    init_info = state['objects_info']['init_info']
    stable = code / 'Behavior-1K/datasets/2026-challenge-task-instances/scenes' / template.scene / 'json' / (template.scene + '_stable.json')
    stable_info = json.loads(stable.read_text())['objects_info']['init_info']
    expected = {(o.role, o.category, o.model): o for o in template.objects}
    refs, names, actual_init, seen, resolutions = [], {}, {}, set(), []
    for binding in meta['bindings']:
        name = binding['name']
        if name in seen:
            continue
        seen.add(name)
        init = init_info[binding['scene_object_name']]['args']
        key = (role_for(binding), init['category'], init['model'])
        require(key in expected, 'native role/model is not in approved candidate: ' + str(key))
        obj = expected[key]
        scale = init.get('scale')
        require(isinstance(scale, list) and len(scale) == 3, 'actual initialization scale missing')
        if max(abs(float(x) - y) for x, y in zip(scale, obj.default_scale)) > 1e-6:
            # Only structural scene floors/walls may have been inventoried from
            # an older contextual export. Targets/supports retain exact defaults.
            frozen = stable_info.get(binding['scene_object_name'], {}).get('args', {})
            require(obj.semantic_kind in ('floor', 'wall') and frozen.get('category') == obj.category and frozen.get('model') == obj.model
                    and frozen.get('scale') and max(abs(float(x) - y) for x, y in zip(scale, frozen['scale'])) <= 1e-6,
                    'actual scale differs from candidate/native structural default')
            resolutions.append(dict(instance=name, candidate_scale=list(obj.default_scale), actual_scale=scale, source=str(stable), source_sha256=sha(stable)))
        actual_asset = asset_path(code, init)
        require(actual_asset.is_file() and sha(actual_asset) == obj.asset_sha256, 'actual runtime USD bytes differ from ready candidate')
        refs.append(ObjectRef(**{**asdict(obj), 'instance': name, 'default_scale': tuple(scale)}))
        names[name] = binding['scene_object_name']; actual_init[name] = init
    require(Counter((o.role, o.category, o.model) for o in refs) == Counter((o.role, o.category, o.model) for o in template.objects), 'native role multiplicities differ')
    dependency = job.get('dependency_library')
    if dependency:
        library = Path(job['dependency_release']) / 'glib/models' / dependency['id'] / 'library.json'
        require(library.is_file() and sha(library) == dependency['library_sha256'], 'admitted dependency library bytes changed')
        require(dependency['positive_grasps'] > 0, 'dependency has no validated grasps')
    setup = {**template.scene_setup, 'source_template': str(directory), 'source_template_sha256': sha(full),
        'runtime_template_id': meta['template_id'], 'runtime_bddl_sha256': meta['bddl_sha256'], 'source_bddl_sha256': meta['bddl_sha256'],
        'source_alias': meta['variant'], 'inst_to_name': names, 'task_rooms': meta['task_rooms'], 'scene_variant': meta.get('scene_variant', {}),
        'fresh_geometry': 'native sampler wrote a fresh TRO', 'fresh_demo': 'pending', 'fresh_replay': 'pending',
        'fresh_metadata_sha256': sha(metadata_path), 'actual_init_args': actual_init, 'default_scale_resolution': resolutions,
        'candidate_fingerprint': template.fingerprint, 'source_candidate_manifest': job['source_manifest'],
        'source_candidate_manifest_sha256': sha(Path(job['source_manifest'])), 'admitted_dependency': dependency,
        'reservation_policy_sha256': policy_pointer['policy_sha256'], 'reservation_asset_hash_manifest_sha256': policy_pointer['asset_hashes']['sha256'],
        'fresh_geometry_tros': [{'path': str(path), 'sha256': sha(path)} for path in tros], 'production_released': False}
    result = Template(template_id=template.template_id, task=template.task, class_id=template.class_id, split=template.split,
        scene=template.scene, room=template.room, objects=tuple(refs),
        definition_sha256=digest(dict(candidate_definition_sha256=template.definition_sha256, fresh_template_sha256=sha(full),
                                    bddl_sha256=meta['bddl_sha256'], actual_init_args=actual_init)), scene_setup=setup)
    check_reservations(result, reservations, hashes)
    return result


def finished_geometry_failure(template, job, runs):
    """Return a finished zero-output native run; an absent report stays pending."""
    argv = job['sample_argv']
    require(argv.count('--out') == 1, 'consumer must name exactly one native output directory')
    directory = Path(argv[argv.index('--out') + 1].replace('${CAMPAIGN_OUT}', str(runs)))
    require(directory.resolve().is_relative_to(runs.resolve()), 'native output directory outside supplied runs')
    path = directory / 'summary.json'
    if not path.is_file():
        return None
    summary = json.loads(path.read_text())
    require(summary.get('args', {}).get('variant') == job['alias'], 'native summary alias differs from consumer')
    require(summary.get('args', {}).get('scene') == template.scene, 'native summary scene differs from consumer')
    if not summary.get('finished') or summary.get('n_written', 0) > 0:
        return None
    records = summary.get('problems', {}).get(job['alias'], {})
    reasons = Counter()
    examples = {}
    for value in records.values():
        reasons.update(value.get('rejections', {}))
        for reason, rows in value.get('rejection_examples', {}).items():
            examples.setdefault(reason, []).extend(rows[:2])
    return dict(candidate_id=template.template_id, class_id=template.class_id,
        summary=str(path), summary_sha256=sha(path), finished=summary['finished'],
        rejections=dict(reasons), rejection_examples=examples,
        errors=summary.get('errors', []), variants=summary.get('variants', {}))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('campaign-root', 'jobs', 'runs', 'code', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    require(not a.output.exists() and a.output.resolve().is_relative_to(a.campaign_root.resolve()), 'new private output required')
    current = active_policy(a.campaign_root)
    require(current is not None, 'active reservation policy required')
    pointer, reservations, hashes = current
    jobs = json.loads(a.jobs.read_text())['jobs']
    candidates, index, source_cache = [], {}, {}
    for job in jobs:
        path = Path(job['source_manifest']).parent / 'contract_candidates.json'
        if path not in source_cache:
            source_cache[path] = {r['fingerprint']: r for r in json.loads(path.read_text())['templates']}
        require(job['template_fingerprint'] in source_cache[path], 'consumer references absent candidate fingerprint')
        template = load_contract(source_cache[path][job['template_fingerprint']])
        require(template.template_id == job['candidate_id'], 'consumer candidate id mismatch')
        check_reservations(template, reservations, hashes)
        candidates.append((template, job))
        key = (template.scene, template.room, job['alias'], template.scene_setup['runtime_bddl_sha256'])
        index.setdefault(key, []).append((template, job))
    output, failures, pending, geometry_failures = [], [], [], []
    for path in sorted(a.runs.rglob('template_meta.json')):
        try:
            meta = json.loads(path.read_text())
            if not meta.get('n_written', 0):
                pending.append(dict(path=str(path), reason='native template has no written TRO'))
                continue
            key = (meta['scene'], meta['task_rooms'][0], meta['variant'], meta['bddl_sha256'])
            matches = index.get(key, [])
            require(len(matches) == 1, 'successful native template has ambiguous/unapproved candidate identity')
            template, job = matches[0]
            result = bind_one(template, job, path, a.code, reservations, hashes, pointer)
            output.append({**asdict(result), 'fingerprint': result.fingerprint})
        except (ValueError, KeyError, OSError) as error:
            failures.append(dict(path=str(path), error=str(error)))
    completed = {t['template_id'] for t in output}
    for template, job in candidates:
        if template.template_id not in completed:
            try:
                failed = finished_geometry_failure(template, job, a.runs)
            except (ValueError, KeyError, OSError) as error:
                failures.append(dict(candidate_id=template.template_id, error=str(error)))
                continue
            if failed:
                geometry_failures.append(failed)
            else:
                pending.append(dict(candidate_id=template.template_id, class_id=template.class_id, reason='no successfully bound fresh geometry yet'))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(dict(templates=output, failures=failures, geometry_failures=geometry_failures, pending=pending, production_released=0,
        jobs_manifest_sha256=sha(a.jobs), active_policy_sha256=pointer['policy_sha256']), indent=2, sort_keys=True) + '\n')
    print(json.dumps(dict(bound=len(output), failures=failures, geometry_failures=len(geometry_failures),
        pending_candidates=len(candidates)-len(completed)-len(geometry_failures), output=str(a.output), sha256=sha(a.output))))


if __name__ == '__main__':
    main()
