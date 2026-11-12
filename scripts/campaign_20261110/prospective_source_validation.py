"""Exact preserved-input engineering collection; never quota or suite credit."""
import json
from pathlib import Path
from .contracts import file_digest, require, validate_campaign_block


def check_review(root, job, environment, policy, *, kind, submission_script=None):
    root, job = Path(root).resolve(), Path(job).resolve()
    entries = [e for e in policy.get('allowed_prospective_source_reviews', [])
               if Path(e['job_dir']).resolve() == job]
    if not entries:
        return None
    require(len(entries) == 1, 'ambiguous prospective source authorization')
    entry = entries[0]
    def checked(reference):
        path = Path(reference['path']).resolve()
        require(path.is_relative_to(root) and file_digest(path) == reference['sha256'],
                'prospective evidence changed')
        return path
    manifest = json.loads(checked(entry['manifest']).read_text())
    require(manifest['schema'] == 'isg-prospective-source-review-v1' and
            manifest['campaign_id'] == 'campaign_20261110' and
            manifest['job_dir'] == str(job) and manifest['instance_id'] == entry['instance_id'] and
            all(manifest[k] == 0 for k in ('new_native_draws', 'benchmark_credit', 'quota_credit', 'current100_r1_credit')),
            'prospective scope cannot grant production or quota')
    prepared = json.loads(checked(manifest['preparation']).read_text())
    config_path = checked(prepared['configuration']); config = json.loads(config_path.read_text())
    request = json.loads(checked(prepared['source_request']).read_text())
    review = json.loads(checked(prepared['source_review']).read_text())
    require(config['purpose'] == 'validation' and config['split'] == config['template']['split'] == 'eval_standard' and
            Path(environment['ISPATIALGYM_CAMPAIGN_CONFIG']).resolve() == config_path and
            request['campaign_id'] == 'campaign_20261110' and request['purpose'] == 'canonical_recovery_bounded_canary' and
            request['benchmark_credit'] == request['current100_r1_credit'] == 0 and request['production_released'] is False,
            'prospective exact evaluation configuration differs')
    require(Path(request['config']).resolve() == config_path and request['class_id'] == config['template']['class_id'] and
            int(environment['START']) == prepared['start'] == request['start_index'] and
            int(environment['N']) == request['count'] == 1, 'prospective source range differs')
    require(review['request'] == prepared['source_request'] and review['returncode'] == 0 and
            review['native_artifacts_complete'] is True and review['benchmark_credit'] == 0 and
            review['production_released'] is False, 'prospective native evidence incomplete')
    native = Path(prepared['runtime']).resolve(); script = checked(prepared['submission_script'])
    require(native.is_relative_to(root/'code') and request['native_runtime'] == review['native_runtime'] == str(native) and
            file_digest(native/'.snapshot_commit') == config['validation_key']['code_sha256'] and
            script == native/'scripts/trainfast/job_recollect.sh', 'prospective runtime differs')
    require(kind == 'source_validation' or (kind == 'collection' and submission_script is not None and
            Path(submission_script).resolve() == script), 'prospective authorization is collector-only')
    require(checked(prepared['job_environment']) == job/'job.env' and
            checked(prepared['copy_proof']) == job/'recollect.json' and (job/'sampling.done').is_file(),
            'prospective preserved-input preparation differs')
    checked(prepared['source_environment'])
    require(all(environment.get(k) == v for k, v in request.get('required_collection_environment', {}).items()),
            'prospective required execution option differs')
    source = checked(prepared['source_tro']); tro = checked(prepared['copied_tro'])
    require(tro.is_relative_to(job/'inst') and file_digest(source) == file_digest(tro) and
            {p.resolve() for p in (job/'inst').rglob('*-tro_state.json')} == {tro}, 'prospective native inventory differs')
    raw = json.loads(tro.read_text())['ispatialgym']; block = raw['campaign']
    validate_campaign_block(block, object_bindings=raw['object_bindings'])
    require(raw['instance_id'] == manifest['instance_id'] and int(raw['instance_id'].rsplit('__', 1)[-1]) == request['start_index'] and
            block['validation_key'] == config['validation_key'] and block['split'] == 'eval_standard',
            'prospective native campaign identity differs')
    source_files = {str(checked(r)): r['sha256'] for r in review['artifacts']}
    require(str(source) in source_files, 'native review omitted exact TRO')
    for suffix in ('.oracle.json', '.summary.json'):
        original = source.with_name(raw['instance_id'] + suffix); copied = tro.with_name(original.name)
        require(str(original) in source_files and file_digest(copied) == source_files[str(original)],
                'prospective native sidecar differs')
    require(json.loads(tro.with_name(raw['instance_id']+'.oracle.json').read_text())['tro_sha256'] == file_digest(tro),
            'prospective oracle sidecar is unbound')
    return {'allowed': True, 'category': 'prospective_source_validation',
            'reason': 'exact preserved engineering source; collection and own replay only',
            'authorization': entry, 'config_path': str(config_path), 'config_sha256': file_digest(config_path),
            'class_id': config['template']['class_id'], 'split': 'eval_standard',
            'new_native_draws': 0, 'quota_credit': 0, 'benchmark_credit': 0, 'current100_r1_credit': 0}


def require_no_quota(root, source_job, instance_ids):
    """Registered engineering instances remain ineligible even if quotas later open."""
    from .production_gate import read_policy
    if not (Path(root)/'controllers/training_production_gate.json').exists():
        return
    policy, _ = read_policy(root)
    wanted = set(instance_ids)
    require(not any(Path(e['job_dir']).resolve() == Path(source_job).resolve() or e['instance_id'] in wanted
                    for e in policy.get('allowed_prospective_source_reviews', [])),
            'prospective engineering evidence cannot consume benchmark or training quota')
