"""Exact current-runtime regression authorization; never authorizes training."""
from __future__ import annotations
import datetime as dt
import json
from pathlib import Path
from .contracts import file_digest, require


def checked(root, reference):
    path = Path(reference['path']).resolve()
    require(path.is_relative_to(Path(root).resolve()), 'suite artifact outside campaign')
    require(file_digest(path) == reference['sha256'], 'suite artifact bytes changed: '+str(path))
    return json.loads(path.read_text())


def timestamp(value):
    return float(value) if isinstance(value, (int, float)) else dt.datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp()


def active(root):
    root = Path(root).resolve()
    pointer = json.loads((root/'runtime_validation/current.json').read_text())
    require(pointer.get('schema') == 'isg-current-runtime-suite-pointer-v1', 'unknown current suite pointer')
    release = checked(root, pointer['release'])
    manifest = checked(root, pointer['manifest'])
    reservation = checked(root, pointer['index_reservation'])
    require(release.get('schema') == 'isg-current-runtime-release-v1' and
            release.get('campaign_id') == 'campaign_20261110' and release.get('required_classes') == 100,
            'suite release identity differs')
    classes = [r['class_id'] for r in manifest['classes']]
    require(len(classes) == len(set(classes)) == 100, 'suite needs exactly100 distinct classes')
    base = Path(pointer['release']['path']).resolve().parent
    require(base.is_relative_to(root/'runtime_validation') and
            Path(pointer['index_reservation']['path']).resolve() == base/'index_reservation.json',
            'suite index reservation belongs to another run')
    runtime = release['frozen_runtime']; native = Path(runtime['path']).resolve()
    require(native.is_relative_to(root/'code') and
            file_digest(native/'.snapshot_commit') == runtime['snapshot_sha256'] and
            (native/'.snapshot_commit').read_text().strip() == runtime['commit'], 'suite frozen runtime changed')
    return pointer, release, manifest, reservation, base


def registered_rows(root, pointer, base, class_id):
    initial = base/'rows'/(class_id+'.json')
    rows = [{'path':str(initial), 'sha256':file_digest(initial)}]
    if pointer.get('attempt_registry'):
        registry = checked(root,pointer['attempt_registry'])
        require(registry.get('schema')=='isg-runtime-suite-attempt-registry-v1' and
                registry['release']==pointer['release'], 'suite retry registry differs')
        for entry in registry['attempts'].get(class_id,[]):
            from .runtime_suite_retry import verify_registered
            verify_registered(root,entry,rows[-1],len(rows),pointer,base)
            rows.append(entry['row'])
    return rows


def authorize(root, environment, config_path, config, *, start, count, kind='collection'):
    """Require the registered immutable run, class row, goal and disjoint range."""
    pointer, release, manifest, reservation, base = active(root)
    row_ref = {'path': environment['ISPATIALGYM_RUNTIME_SUITE_ROW'],
               'sha256': environment['ISPATIALGYM_RUNTIME_SUITE_ROW_SHA256']}
    row = checked(root, row_ref); template = config['template']
    require(environment['ISPATIALGYM_RUNTIME_SUITE_RUN'] == row['run_id'] == release['run_id'] and
            row['run_started_at'] == release['run_started_at'] and row['release'] == pointer['release'] and
            row['frozen_runtime'] == release['frozen_runtime'], 'suite run/runtime differs')
    require(template['split'] == 'eval_standard' and config.get('purpose') == 'validation' and
            row['class_id'] == template['class_id'] and row['class_id'] in {r['class_id'] for r in manifest['classes']},
            'suite authorization is standard regression only')
    approved=next(entry for entry in manifest['classes'] if entry['class_id']==row['class_id'])
    require(template == approved['exact_configuration'], 'suite template differs from approved immutable configuration')
    rows=registered_rows(root,pointer,base,row['class_id'])
    require(row_ref in rows, 'suite row is not canonical or a registered retry')
    require(kind in ('collection','source_validation'), 'unsupported suite operation')
    require(kind!='collection' or row_ref==rows[-1], 'suite attempt retired from dispatch')
    require(Path(config_path).resolve() == Path(row['campaign_config']['path']).resolve() and
            file_digest(config_path) == row['campaign_config']['sha256'], 'suite configuration differs')
    require(config['validation_key']['code_sha256'] == release['frozen_runtime']['snapshot_sha256'],
            'suite config runtime differs')
    require(type(start) is int and type(count) is int and 1 <= count <= 3 and
            (start, count) == (row['fresh_start'], row['fresh_count']), 'suite exact range differs')
    reserved = reservation['proof']; lo = int(reserved['start']); size = int(reserved['reserved_count'])
    require(size == 300, 'original suite reservation changed')
    if row_ref==rows[0]:
        require(lo <= start and start+count <= lo+size and (start-lo)%3 == 0,
                'suite range is outside its disjoint reservation')
        ordinal = (start-lo)//3
        require(manifest['classes'][ordinal]['class_id'] == row['class_id'], 'suite range belongs to another class')
    camera = template['task'].startswith('camera_relocalization')
    require(row['camera_exempt'] is camera, 'camera exemption differs')
    if camera:
        require(row['goal_plan'] is None and row['group_id'] is None and count == 1,
                'camera suite row must use the explicit one-instance exception')
    else:
        plan = checked(root, row['goal_plan'])
        require(count == 3 and row['group_id'] and plan.get('group_id') == row['group_id'] and
                Path(environment['ISPATIALGYM_CAMPAIGN_GOAL_PLAN']).resolve() == Path(row['goal_plan']['path']).resolve(),
                'suite goal group differs')
    if environment.get('ISGE_CODE'):
        require(Path(environment['ISGE_CODE']).resolve() == Path(release['frozen_runtime']['path']).resolve(),
                'suite execution runtime differs')
    return {'row': row_ref, 'release': pointer['release'], 'run_id': row['run_id'],
            'class_id': row['class_id'], 'start': start, 'count': count,
            'production_authorization': False, 'quota_claim_required': True}


def current_training_ready(root):
    """Re-audit all100 exact results; an aggregate label is never a release."""
    try:
        _, release, manifest, _, base = active(root)
        from .runtime_suite_acceptance import verify_result
        for entry in manifest['classes']:
            path = base/'acceptance'/(entry['class_id']+'.json')
            result = verify_result(root, {'path':str(path), 'sha256':file_digest(path)})
            require(result['run_id'] == release['run_id'] and result['class_id'] == entry['class_id'] and result['validation_passed'] is True,
                    'current-runtime class not validated')
        return True
    except (OSError, ValueError, KeyError, TypeError, ImportError):
        return False
