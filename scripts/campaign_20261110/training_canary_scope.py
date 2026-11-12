"""Bounded validation may execute an immutable future production configuration.

Authorization belongs to an exact zero-credit index range. The configuration's
physical identity is unchanged and cannot itself authorize production.
"""
from .contracts import require


def validate_scope(config, authorization):
    require(config['split'] == config['template']['split'] == 'train', 'not campaign training validation')
    require(authorization.get('purpose') == 'training_pipeline_canary', 'explicit canary purpose required')
    purpose = config.get('purpose')
    require(purpose in ('training_pipeline_canary', 'training_production'), 'unsupported training configuration purpose')
    if purpose == 'training_production':
        require(authorization.get('validation_scope') == 'exact_production_configuration'
                and authorization.get('production_released') is False
                and type(authorization.get('benchmark_credit')) is int and authorization['benchmark_credit'] == 0
                and type(authorization.get('count')) is int and type(authorization.get('max_instances')) is int
                and 0 < authorization['count'] <= authorization['max_instances'] <= 24,
                'production configuration needs explicit bounded zero-credit validation')
    return {'configuration_purpose': purpose, 'authorization_purpose': 'training_pipeline_canary',
            'production_released': False, 'benchmark_credit': 0}


def restoration_execution(meta, tro_path, native_runtime, *, environment=None):
    """Recheck a production-purpose oracle input before its first state mutation.

    The exact job can be a bounded test of the eventual configuration or a
    separately released production queue. A purpose string grants neither.
    """
    if meta.get('runtime', {}).get('purpose') != 'training_production':
        return None
    import json
    import os
    from pathlib import Path
    from .contracts import digest, file_digest
    from .production_gate import check_job, read_environment
    environment = os.environ if environment is None else environment
    root = Path(environment['ISGF_ROOT']).resolve(); tro = Path(tro_path).resolve()
    require(tro.is_relative_to(root/'out'), 'production restoration TRO outside campaign output')
    parents = [p.parent for p in tro.parents if p.name == 'inst' and (p.parent/'job.env').is_file()]
    require(len(parents) == 1, 'production restoration exact source job missing or ambiguous')
    job = parents[0]; job_env = read_environment(job/'job.env')
    config_path = Path(job_env['ISPATIALGYM_CAMPAIGN_CONFIG']).resolve()
    require(config_path.is_relative_to(root), 'production restoration config outside campaign')
    config = json.loads(config_path.read_text()); raw = json.loads(tro.read_text())['ispatialgym']
    require(raw['campaign'] == meta and config.get('purpose') == 'training_production' and
            config['validation_key'] == meta['validation_key'] and
            config['template']['scene_setup'] == meta['scene_setup'] and
            config['perturbations'] == meta['perturbations'], 'production restoration physical input differs')
    runtime = Path(native_runtime).resolve()
    require(runtime.is_relative_to(root/'code'), 'production restoration native runtime differs')
    runtime_compatibility = None
    if file_digest(runtime/'.snapshot_commit') != config['validation_key']['code_sha256']:
        from .training_batch_probe import inputs, verify_runtime_compatibility
        request_path = Path(environment.get('ISPATIALGYM_TRAINING_OBSERVATION_REQUEST', '')).resolve()
        require(request_path.is_relative_to(root) and request_path.is_file() and
                file_digest(request_path) == environment.get('ISPATIALGYM_TRAINING_OBSERVATION_REQUEST_SHA256'),
                'production restoration native runtime differs without exact observation proof')
        request, records = inputs(root, request_path)
        require(not request.get('physical_candidate') and
                request['probe_runtime_sha256'] == file_digest(runtime/'.snapshot_commit') and
                any(row['instance_id'] == raw['instance_id'] and
                    row['artifacts']['tro']['sha256'] == file_digest(tro) for row in records),
                'production observation runtime or exact input differs')
        runtime_compatibility = verify_runtime_compatibility(root, request, records, runtime)
    start, count = int(job_env['START']), int(job_env['N'])
    index = int(raw['instance_id'].rsplit('__', 1)[-1])
    require(count > 0 and start <= index < start+count, 'production restoration instance outside authorized job range')
    decision = check_job(root, job, kind='source_validation')
    require(decision.get('allowed') is True and decision.get('config_sha256') == file_digest(config_path) and
            decision.get('category') in ('validation_canary', 'released_training'),
            'production restoration needs current exact bounded or released authorization; decision=' +
            json.dumps(decision,sort_keys=True,default=str))
    if decision['category'] == 'validation_canary':
        authorization = decision['authorization']; path = Path(authorization['authorization_path']).resolve()
        require(path.is_relative_to(root) and file_digest(path) == authorization['authorization_sha256'],
                'production restoration canary authorization changed')
        validate_scope(config, json.loads(path.read_text()))
    return {'schema':'isg-training-restoration-execution-v1', 'metadata_sha256':digest(meta),
            'config_sha256':file_digest(config_path), 'source_tro_sha256':file_digest(tro),
            'instance_id':raw['instance_id'], 'start':start, 'count':count,
            'category':decision['category'], 'authorization_checked':True,
            'source_job':str(job), 'native_runtime':str(runtime),
            'observational_runtime_compatibility':runtime_compatibility}
