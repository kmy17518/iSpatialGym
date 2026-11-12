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


def validate_pointer(root, pointer):
    root = Path(root).resolve()
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


def run_registry(root):
    path = Path(root)/'runtime_validation/authorized_runs.json'
    if not path.exists():
        return None
    value = json.loads(path.read_text())
    require(value.get('schema') == 'isg-runtime-suite-run-registry-v1' and
            value.get('campaign_id') == 'campaign_20261110' and isinstance(value.get('runs'), dict),
            'unknown registered suite runs')
    return value


def active(root, *, run_dir=None, run_id=None):
    """Resolve only the current run or an explicitly registered immutable run.

    Legacy owners keep their current run. Concurrent owners name their own run
    directory; a row's run ID selects its exact authorization without changing
    the default pointer or borrowing another run's evidence.
    """
    root = Path(root).resolve()
    current = validate_pointer(root, json.loads((root/'runtime_validation/current.json').read_text()))
    if run_dir is None and run_id is None:
        return current
    base = Path(run_dir).resolve() if run_dir is not None else None
    if ((base is None or base == current[4]) and
            (run_id is None or run_id == current[1]['run_id'])):
        return current
    registry = run_registry(root)
    require(registry is not None, 'suite run is not registered')
    matches = []
    for name, entry in registry['runs'].items():
        if run_id is not None and name != run_id:
            continue
        value = validate_pointer(root, checked(root, entry['pointer']))
        require(value[1]['run_id'] == name, 'registered suite run ID differs')
        approval = checked(root, entry['approval'])
        _run_approval(approval, entry['pointer'])
        if (base is None or base == value[4]) and (run_id is None or run_id == name):
            matches.append(value)
    require(len(matches) == 1, 'suite run is absent or ambiguous in registered runs')
    return matches[0]


def _run_approval(approval, pointer_ref):
    require(approval.get('schema') == 'isg-runtime-suite-run-approval-v1' and
            approval.get('owner') == '/root' and approval.get('approved') is True and
            approval.get('pointer') == pointer_ref and approval.get('production_authorization') is False,
            'suite run lacks exact owner approval')


def _shared_control_closure(root, references):
    """Refuse a registration which the campaign's secondary account cannot read."""
    import stat
    for reference in references:
        path = Path(reference['path']).resolve()
        require(path.is_relative_to(root) and file_digest(path) == reference['sha256'],
                'registered suite control bytes differ')
        require(stat.S_IMODE(path.stat().st_mode) & 0o044,
                'registered suite control is not shared-readable: '+str(path))
        parent = path.parent
        while parent != root:
            require(stat.S_IMODE(parent.stat().st_mode) & 0o011,
                    'registered suite control directory is not shared-traversable: '+str(parent))
            parent = parent.parent


def register_run(root, pointer_ref, approval_ref, *, required_validation_run=False):
    """Register an approved immutable run; no dispatch, acceptance or quota write."""
    from .contracts import digest
    from .storage import atomic_directory_lock
    from .runtime_suite_acceptance import immutable
    from .runtime_suite_retry import _replace_pointer
    root = Path(root).resolve()
    with atomic_directory_lock(root/'runtime_validation/run_registration.lockdir'):
        approval = checked(root, approval_ref); _run_approval(approval, pointer_ref)
        require(approval.get('required_validation_run') is required_validation_run,
                'required validation run decision differs from approval')
        value = validate_pointer(root, checked(root, pointer_ref))
        _shared_control_closure(root, [pointer_ref, approval_ref, value[0]['release'],
            value[0]['manifest'], value[0]['index_reservation']])
        current = active(root); name = value[1]['run_id']
        require(name != current[1]['run_id'] and value[4] != current[4],
                'concurrent registration cannot replace the current run')
        require([r['class_id'] for r in value[2]['classes']] ==
                [r['class_id'] for r in current[2]['classes']], 'new suite catalog class order differs')
        registry = run_registry(root) or {'schema':'isg-runtime-suite-run-registry-v1',
            'campaign_id':'campaign_20261110','runs':{}}
        entry = {'pointer':pointer_ref,'approval':approval_ref}
        require(name not in registry['runs'] or registry['runs'][name] == entry,
                'immutable registered suite identity changed')
        lo = int(value[3]['proof']['start']); size = int(value[3]['proof']['reserved_count'])
        require(size == 300, 'new suite requires its own 300-index reservation')
        others = [current] + [validate_pointer(root,checked(root,e['pointer']))
            for n,e in registry['runs'].items() if n != name]
        for other in others:
            oldlo = int(other[3]['proof']['start']); oldsize = int(other[3]['proof']['reserved_count'])
            require(lo+size <= oldlo or oldlo+oldsize <= lo, 'suite reservations overlap')
        registry['runs'][name] = entry
        if required_validation_run:
            previous_target = registry.get('required_validation_run')
            if previous_target and previous_target['run_id'] != name:
                previous_run = active(root, run_id=previous_target['run_id'])
                require(timestamp(value[1]['run_started_at']) > timestamp(previous_run[1]['run_started_at']),
                        'required validation run cannot move back to an older release')
            registry['required_validation_run'] = {'run_id':name,'pointer':pointer_ref}
        path = root/'runtime_validation/authorized_runs.json'
        if path.exists():
            old = json.loads(path.read_text())
            immutable(root/'runtime_validation/run_registry_history'/(digest(old)+'.json'),old)
        immutable(root/'runtime_validation/run_registry_history'/(digest(registry)+'.json'),registry)
        if path.exists():
            _replace_pointer(path,registry)
        else:
            import os
            import tempfile
            from .contracts import canonical_json
            previous = (root/'runtime_validation/current.json').stat()
            temporary = None
            try:
                with tempfile.NamedTemporaryFile('w',dir=path.parent,prefix=path.name+'.',delete=False) as stream:
                    temporary = Path(stream.name)
                    if os.fstat(stream.fileno()).st_gid != previous.st_gid:
                        os.fchown(stream.fileno(),-1,previous.st_gid)
                    os.fchmod(stream.fileno(),0o640)
                    stream.write(canonical_json(registry)+'\n');stream.flush();os.fsync(stream.fileno())
                os.replace(temporary,path);temporary = None
            finally:
                if temporary is not None:temporary.unlink(missing_ok=True)
        return {'run_id':name,'pointer':pointer_ref,'registry':{'path':str(path),'sha256':file_digest(path)},
                'required_validation_run':registry.get('required_validation_run'),
                'current_pointer_unchanged':True,'dispatch_performed':False,'production_authorization':False}


def required_run(root):
    registry = run_registry(root)
    if registry is None:
        return active(root)
    if not registry.get('required_validation_run'):
        for entry in registry['runs'].values():
            approval = checked(root, entry['approval']); _run_approval(approval, entry['pointer'])
            require(approval.get('required_validation_run') is False,
                    'required validation run target is missing')
        return active(root)
    target = registry['required_validation_run']
    require(target.get('run_id') in registry['runs'] and
            registry['runs'][target['run_id']]['pointer'] == target.get('pointer'),
            'required validation run is not the exact registered run')
    approval = checked(root, registry['runs'][target['run_id']]['approval'])
    require(approval.get('required_validation_run') is True,
            'required validation run lacks the exact owner decision')
    value = active(root, run_id=target['run_id'])
    require(value[0] == checked(root,target['pointer']), 'required validation pointer differs')
    return value


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
    row_ref = {'path': environment['ISPATIALGYM_RUNTIME_SUITE_ROW'],
               'sha256': environment['ISPATIALGYM_RUNTIME_SUITE_ROW_SHA256']}
    row = checked(root, row_ref); template = config['template']
    pointer, release, manifest, reservation, base = active(root, run_id=row['run_id'])
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
        _, release, manifest, _, base = required_run(root)
        from .runtime_suite_acceptance import verify_result
        for entry in manifest['classes']:
            path = base/'acceptance'/(entry['class_id']+'.json')
            result = verify_result(root, {'path':str(path), 'sha256':file_digest(path)})
            require(result['run_id'] == release['run_id'] and result['class_id'] == entry['class_id'] and result['validation_passed'] is True,
                    'current-runtime class not validated')
        return True
    except (OSError, ValueError, KeyError, TypeError, ImportError):
        return False
