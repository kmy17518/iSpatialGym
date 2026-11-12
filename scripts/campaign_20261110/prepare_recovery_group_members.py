"""Project completed bounded native groups into exact singleton collectors.

The generation request/runtime/configuration and original bytes stay explicit.
Member reviews are engineering projections, never new native executions or
quota authorization. Compatible with the existing sole replay owner.
"""
import argparse
import json
import shlex
import shutil
import stat
import time
from pathlib import Path

from .contracts import file_digest, require, write_json_atomic
from .storage import atomic_directory_lock


def reference(path):
    return {'path': str(path), 'sha256': file_digest(path)}


def checked(root, ref):
    p = Path(ref['path']).resolve()
    require(p.is_relative_to(root) and file_digest(p) == ref['sha256'], 'group evidence changed')
    return p


def parent_has_complete_members(v):
    """A timed-out parent may contain complete members; it remains timed out."""
    if v.get('returncode') == 0 and v.get('native_artifacts_complete') is True:
        return True
    members = v.get('members', [])
    return (v.get('returncode') == 124 and v.get('native_artifacts_complete') is False and
            v.get('requested_count') == 3 and type(v.get('written_count')) is int and
            0 < v['written_count'] <= 3 and v['written_count'] == len(members) and
            len(v.get('artifacts', [])) == 3 * len(members))


def checked_member_source(root, request_ref, review_path, index):
    """Verify existing bytes independently of the parent's completion status."""
    qpath = checked(root, request_ref)
    q = json.loads(qpath.read_text()); review_ref = reference(review_path)
    v = json.loads(checked(root, review_ref).read_text())
    require(q['schema'] == 'isg-bounded-canonical-group-request-v1' and
            q['campaign_id'] == 'campaign_20261110' and q['purpose'] == 'canonical_recovery_bounded_canary' and
            q['count'] == 3 and q['benchmark_credit'] == q['quota_credit'] == q.get('current100_r1_credit', 0) == 0 and
            q['production_released'] is False, 'not an authorized bounded engineering group')
    require(v['request'] == request_ref and parent_has_complete_members(v) and
            v['native_runtime'] == q['native_runtime'] and v['benchmark_credit'] == 0 and
            v['production_released'] is False, 'native group review incomplete')
    require(type(index) is int and q['start_index'] <= index < q['start_index'] + q['count'], 'member outside reserved group')
    for path, digest in q['inputs'].items(): checked(root, {'path': path, 'sha256': digest})
    artifacts = {str(checked(root, r)): r for r in v['artifacts']}
    candidates = [Path(p) for p in artifacts if p.endswith('-tro_state.json') and
                  int(json.loads(Path(p).read_text())['ispatialgym']['instance_id'].rsplit('__', 1)[-1]) == index]
    require(len(candidates) == 1, 'group member is absent or ambiguous')
    tro = candidates[0]; raw = json.loads(tro.read_text())['ispatialgym']; iid = raw['instance_id']
    config = json.loads(Path(q['config']).read_text())
    require(raw['campaign']['validation_key'] == config['validation_key'] and
            raw['campaign']['split'] == config['split'] == 'eval_standard', 'member native identity differs')
    selected = [tro, tro.with_name(iid + '.oracle.json'), tro.with_name(iid + '.summary.json')]
    require(all(str(p) in artifacts for p in selected) and
            json.loads(selected[1].read_text())['tro_sha256'] == file_digest(tro), 'member sidecar is not bound')
    members = [m for m in v.get('members', []) if m['instance_id'] == iid and m['index'] == index]
    require(len(members) == 1, 'member omitted or ambiguous in native review')
    return q, v, review_ref, artifacts, selected, members


def member_projection(root, request_ref, review_path, index, output):
    """Bind one existing member without changing its native identity."""
    q, v, review_ref, artifacts, selected, members = checked_member_source(root, request_ref, review_path, index)
    member = {**q, 'schema': 'isg-bounded-canonical-member-projection-v2',
              'start_index': index, 'count': 1,
              # Some independently authored bounded requests omit the retired
              # R1 accounting label. The guard above still requires explicit
              # zero benchmark/quota credit and no production authorization.
              # Make the zero explicit in the projection, preserving parent
              # bytes and their bound identity for every consumer.
              'current100_r1_credit': 0,
              'native_generation_request': request_ref, 'native_generation_review': review_ref,
              'projection_only': True, 'new_native_draws': 0,
              'projection_meaning': 'Exact existing member of parent group; no new native execution or identity change.'}
    output.mkdir(parents=True, exist_ok=False)
    write_json_atomic(output/'request.json', member)
    mv = {**v, 'schema': 'isg-bounded-native-member-review-v2', 'request': reference(output/'request.json'),
          'artifacts': [artifacts[str(p)] for p in selected], 'members': members,
          'requested_count': 1, 'written_count': 1, 'parent_requested_count': q['count'],
          'native_generation_request': request_ref, 'native_generation_review': review_ref,
          'projection_only': True, 'new_native_draws': 0, 'member_native_artifacts_complete': True,
          'parent_returncode': v['returncode'], 'parent_native_artifacts_complete': v['native_artifacts_complete']}
    write_json_atomic(output/'native_review.json', mv)
    return member, selected[0]


def validate_timed_out_member(root, request, review):
    """Strict consumer verification; never turn parent rc124 into success."""
    require(request.get('schema') == 'isg-bounded-canonical-member-projection-v2' and
            review.get('schema') == 'isg-bounded-native-member-review-v2' and
            request.get('projection_only') is review.get('projection_only') is True and
            request.get('new_native_draws') == review.get('new_native_draws') == 0 and
            review.get('member_native_artifacts_complete') is True and
            review.get('returncode') == review.get('parent_returncode') == 124 and
            review.get('native_artifacts_complete') is review.get('parent_native_artifacts_complete') is False,
            'invalid timed-out member projection')
    require(request['native_generation_request'] == review['native_generation_request'] and
            request['native_generation_review'] == review['native_generation_review'], 'member parent differs')
    parent_review = checked(root, request['native_generation_review'])
    q, v, _, artifacts, selected, members = checked_member_source(root, request['native_generation_request'], parent_review, request['start_index'])
    require(v['returncode'] == 124 and request['count'] == review['requested_count'] == review['written_count'] == 1 and
            review['parent_requested_count'] == q['count'] and review['members'] == members and
            review['artifacts'] == [artifacts[str(p)] for p in selected], 'incomplete projected native member')
    require(all(request.get(k) == value for k, value in q.items() if k not in ('schema', 'count', 'start_index')),
            'projected native identity changed')
    return True


def prepare(root, parent_ref, index):
    parent = checked(root, parent_ref); q = json.loads(parent.read_text())
    label = q['class_id'] + '_g' + str(index)
    out = root/'infra/evidence_acceptance/recovery_group_members_r1'/label
    prepared = out/'prepared.json'
    if prepared.exists():
        saved = json.loads(prepared.read_text())
        require(saved['native_generation_request'] == parent_ref, 'existing member belongs to another group')
        for name in ('source_tro', 'copied_tro', 'source_request', 'source_review', 'job_environment', 'copy_proof', 'native_generation_review'):
            checked(root, saved[name])
        return prepared
    member, tro = member_projection(root, parent_ref, parent.parent/'native_review.json', index, out)
    native = Path(q['native_runtime']); cfg = json.loads(Path(q['config']).read_text())
    require(native.is_relative_to(root/'code') and file_digest(native/'.snapshot_commit') == cfg['validation_key']['code_sha256'],
            'member runtime differs from source config')
    iid = json.loads(tro.read_text())['ispatialgym']['instance_id']
    template, variant = tro.parent.parent, tro.parent.parent.parent
    job = root/'out/recovery_group_members_r1'/q['class_id']/('g'+str(index))
    require(not job.exists(), 'unreviewed member job already exists')
    target_variant = job/'inst'/variant.name
    files = {}
    for path in variant.rglob('*'):
        if not path.is_file() or (path.parent == tro.parent and not path.name.startswith(iid)):
            continue
        rel = path.relative_to(variant); dest = target_variant/rel
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, dest)
        files[str(rel)] = file_digest(path)
        require(file_digest(dest) == files[str(rel)], 'member copy changed')
    target = target_variant/template.name; copied = target/'instances'/tro.name
    require(list((job/'inst').rglob('*-tro_state.json')) == [copied], 'member copy contains another TRO')
    (job/'claims').mkdir(); (job/'sampling.done').touch()
    source_env = {}
    for line in (parent.parent/'exact.env').read_text().splitlines():
        require(line.startswith('export ') and '=' in line, 'invalid native environment')
        key, value = line[7:].split('=', 1); values = shlex.split(value)
        require(len(values) == 1, 'invalid native environment value'); source_env[key] = values[0]
    env = {'ALIAS_KEY': q['class_id'], 'ALIAS': variant.name, 'FAMILY': cfg['template']['task'].split('-')[0],
           'START': str(index), 'N': '1', 'JOB_TAG': 'g'+str(index), 'ISGF_ROOT': str(root), 'ISGE_CODE': str(native),
           'ISGF_SCRIPTS': str(native/'scripts/trainfast'), 'ISGF_JB_NFS': str(root/'jobs'),
           'KIT_TPL_NFS_ROOT': str(root/'kit_templates'), 'KIT_TEX_STORE': str(root/'kit_textures'),
           'ISGF_EXP_PATH': str(root/'isaacsim_overlay/apps'), 'KCOLLECT': '1', 'TIME_LIMIT_S': '4800', 'DEMO_LIMIT_S': '1800',
           'TEMPLATE': str(target), 'RECOLLECT_OF': str(template),
           'ISGF_SAMPLING_CACHE': str(root/'cache'/('recovery_group_'+str(index))),
           'ISGF_COLLECT_EXTRA': '--no-goal-override', 'ISGF_GRASPS': 'v2',
           'ISGF_GOAL_OCCLUSION': 'on', 'ISPATIALGYM_GOAL_OCCLUSION': 'on',
           'ISPATIALGYM_CAMPAIGN_CONFIG': q['config'], 'ISG_VEC_ASSET_CACHE_DIR': str(root/'asset_cache'), **source_env}
    require(all(env.get(k) == v for k, v in q.get('required_collection_environment', {}).items()), 'required option differs')
    (job/'job.env').write_text(''.join(k+'='+shlex.quote(v)+'\n' for k, v in sorted(env.items())))
    write_json_atomic(job/'recollect.json', {'schema': 'isg-existing-native-collector-copy-v1', 'source': str(template),
        'source_review': reference(out/'native_review.json'), 'source_native_files': files,
        'copied_tro': reference(copied), 'original_tro': reference(tro), 'byte_identical': True, 'new_native_draws': 0,
        'native_generation_request': parent_ref, 'preserved_campaign_configuration': reference(q['config'])})
    script = native/'scripts/trainfast/job_recollect.sh'
    command = ['sbatch', '--parsable', '--nodes=1', '--ntasks=1', '--nice=0', '-A', 'vision', '-p', 'svl,viscam',
               '--exclude=svl9', '--gres=gpu:titanrtx:1', '-c', '8', '--mem=32G', '--time=01:20:00',
               '-J', 'isg261110_group_collect', '-o', str(root/'slurm/%x_%j.log'), str(script), str(job)]
    write_json_atomic(prepared, {'campaign_id': 'campaign_20261110', 'observed_at': time.time(),
        'candidate_id': 'recovery_group_'+label, 'start': index, 'configuration': reference(q['config']),
        'source_request': reference(out/'request.json'), 'source_review': reference(out/'native_review.json'),
        'native_generation_request': parent_ref, 'native_generation_review': reference(parent.parent/'native_review.json'),
        'source_native_template': str(template), 'runtime': str(native), 'job_dir': str(job),
        'copied_tro': reference(copied), 'source_tro': reference(tro), 'copy_proof': reference(job/'recollect.json'),
        'submission_script': reference(script), 'command': command, 'job_environment': reference(job/'job.env'),
        'source_environment': reference(parent.parent/'exact.env'), 'new_native_draws': 0, 'benchmark_credit': 0,
        'gate': {'allowed': False, 'reason': 'awaiting_exact_prospective_registration'}})
    return prepared


def register(root, prepared_paths):
    from .production_gate import read_environment, check_job
    from .prospective_source_validation import check_review, require_no_quota
    from .training_policy_history import replace_policy
    entries = []
    for prepared in prepared_paths:
        p = json.loads(prepared.read_text()); iid = json.loads(Path(p['copied_tro']['path']).read_text())['ispatialgym']['instance_id']
        manifest = prepared.with_name('prospective_manifest.json')
        value = {'schema': 'isg-prospective-source-review-v1', 'campaign_id': 'campaign_20261110', 'job_dir': p['job_dir'],
                 'instance_id': iid, 'preparation': reference(prepared), 'new_native_draws': 0, 'benchmark_credit': 0,
                 'quota_credit': 0, 'current100_r1_credit': 0, 'production_released': False}
        if manifest.exists(): require(json.loads(manifest.read_text()) == value, 'member manifest changed')
        else: write_json_atomic(manifest, value)
        entries.append({'job_dir': p['job_dir'], 'instance_id': iid, 'manifest': reference(manifest)})
    path = root/'controllers/training_production_gate.json'
    with atomic_directory_lock(path.with_suffix('.lockdir')):
        old = json.loads(path.read_text()); require(old['standalone_training_allowed'] is False, 'bulk hold changed')
        updated = {**old}; rows = list(old.get('allowed_prospective_source_reviews', []))
        for entry in entries:
            prior = [r for r in rows if r['job_dir'] == entry['job_dir']]
            if prior: require(prior == [entry], 'existing member authorization differs')
            else: rows.append(entry)
        updated['allowed_prospective_source_reviews'] = rows
        for prepared in prepared_paths:
            p = json.loads(prepared.read_text())
            require(check_review(root, p['job_dir'], read_environment(Path(p['job_dir'])/'job.env'), updated,
                                 kind='collection', submission_script=p['submission_script']['path'])['allowed'], 'member review denied')
        if updated != old:
            mode = stat.S_IMODE(path.stat().st_mode)
            replace_policy(root, updated, expected_previous_sha256=file_digest(path)); path.chmod(mode | 0o040)
    for prepared in prepared_paths:
        p = json.loads(prepared.read_text())
        require(check_job(root, p['job_dir'], submission_script=p['submission_script']['path'])['allowed'], 'registered member denied')
        require(check_job(root, p['job_dir'], kind='source_validation')['allowed'], 'member replay source denied')
    return entries


def run(root, batch_ref, *, register_members=False):
    root = Path(root).resolve(); batch = json.loads(checked(root, batch_ref).read_text()); prepared = []; pending = []
    require(batch['schema'] == 'isg-canonical-six-group-batch-v1' and batch['benchmark_credit'] == 0, 'unapproved parent batch')
    for row in batch['requests']:
        parent = checked(root, row['request']); review = parent.parent/'native_review.json'
        if not review.exists(): pending.append(row['class_id']); continue
        v = json.loads(review.read_text())
        if not parent_has_complete_members(v): pending.append(row['class_id']); continue
        for member in v['members']:
            prepared.append(prepare(root, row['request'], int(member['index'])))
    entries = register(root, prepared) if register_members and prepared else []
    return {'schema': 'isg-recovery-group-member-preparation-v1', 'observed_at': time.time(),
            'parent_batch': batch_ref, 'prepared': [reference(p) for p in prepared], 'registered': entries,
            'pending_classes': pending, 'benchmark_credit': 0, 'submitted_jobs': []}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--root', required=True); p.add_argument('--batch', required=True)
    p.add_argument('--batch-sha256', required=True); p.add_argument('--register', action='store_true'); args = p.parse_args()
    print(json.dumps(run(args.root, {'path': args.batch, 'sha256': args.batch_sha256}, register_members=args.register)))


if __name__ == '__main__': main()
