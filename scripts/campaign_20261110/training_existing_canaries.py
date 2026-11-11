"""Permit only exact collector-only reviews of previously authorized native canaries."""
from datetime import datetime
import json
from pathlib import Path
from scripts.campaign_20261110.contracts import file_digest,require,validate_campaign_block


def check_existing_review(root,job,environment,policy,*,kind,submission_script=None):
    root,job=Path(root).resolve(),Path(job).resolve()
    entries=[e for e in policy.get('allowed_existing_canary_reviews',[]) if Path(e['job_dir']).resolve()==job]
    if not entries:return None
    require(len(entries)==1,'ambiguous existing-canary review authorization')
    entry=entries[0]
    def checked(ref):
        path=Path(ref['path']).resolve()
        require(path.is_relative_to(root) and file_digest(path)==ref['sha256'],'existing-canary review evidence changed')
        return path
    manifest=json.loads(checked(entry['review_manifest']).read_text())
    require(manifest['schema']=='isg-existing-training-canary-review-v1' and manifest['campaign_id']=='campaign_20261110'
        and Path(manifest['job_dir']).resolve()==job and manifest['new_native_instances']==0,'invalid existing-canary scope')
    config_path=checked(manifest['configuration']);config=json.loads(config_path.read_text())
    require(Path(environment['ISPATIALGYM_CAMPAIGN_CONFIG']).resolve()==config_path
        and config['split']==config['template']['split']=='train','existing-canary config differs')
    require(entry['config_sha256']==file_digest(config_path) and entry['purpose']=='training_pipeline_canary',
            'existing-canary authorization config differs')
    require((int(environment['START']),int(environment['N']))==(entry['start'],entry['count'])
        and 0<entry['count']<=entry['max_instances']<=24,'existing-canary index range differs')
    require(checked(manifest['job_environment'])==job/'job.env','collector environment differs')
    require(checked(manifest['recollect_receipt'])==job/'recollect.json' and (job/'sampling.done').is_file(),
            'collector-only preparation incomplete')
    script=checked(manifest['submission_script'])
    native=Path(manifest['native_runtime']).resolve()
    require(script==native/'scripts/trainfast/job_recollect.sh'
        and file_digest(native/'.snapshot_commit')==config['validation_key']['code_sha256'],
            'collector-only runtime or script differs')
    require(kind=='source_validation' or (kind=='collection' and submission_script is not None
        and Path(submission_script).resolve()==script),'existing canary cannot authorize a sampler script')
    original=checked({'path':entry['authorization_path'],'sha256':entry['authorization_sha256']})
    auth=json.loads(original.read_text())
    require(all(auth.get(k)==entry[k]for k in ('purpose','config_sha256','start','count')),'original canary authorization differs')
    expiry=datetime.fromisoformat(entry['expires_at'].replace('Z','+00:00')).timestamp()
    copied=set()
    for item in manifest['native_files']:
        source=checked(item['source_tro']);tro=checked(item['copied_tro'])
        require(tro.is_relative_to(job/'inst') and file_digest(source)==file_digest(tro),'existing native TRO changed')
        raw=json.loads(tro.read_text())['ispatialgym'];block=raw['campaign']
        validate_campaign_block(block,object_bindings=raw['object_bindings'])
        require(block['campaign_id']=='campaign_20261110' and block['split']=='train'
            and block['validation_key']==config['validation_key'],'native campaign identity differs')
        require(original.stat().st_mtime<=block['created_at']<expiry,'native creation was outside original authorization')
        index=int(raw['instance_id'].rsplit('__',1)[-1])
        require(entry['start']<=index<entry['start']+entry['count'],'native index was not authorized')
        sidecar=checked(item['copied_sidecar']);source_sidecar=checked(item['source_sidecar'])
        require(sidecar==tro.with_name(raw['instance_id']+'.oracle.json') and file_digest(sidecar)==file_digest(source_sidecar)
            and json.loads(sidecar.read_text())['tro_sha256']==file_digest(tro),'existing native sidecar changed')
        copied.add(tro)
    require(len(copied)==entry['count'] and copied=={p.resolve()for p in (job/'inst').rglob('*-tro_state.json')},
            'collector-only native inventory differs')
    return {'allowed':True,'category':'validation_canary','reason':'exact existing native canary; collector-only, no new sampling',
        'authorization':entry,'config_path':str(config_path),'config_sha256':file_digest(config_path),
        'class_id':config['template']['class_id'],'task':config['template']['task'],'split':'train','new_native_instances':0}
