"""Audit authorized training canaries without allocating benchmark quota.

Canary exports exercise the same artifact normalizer as the final export, but
remain in a separate immutable ledger. A successful trajectory is not evidence
that the complete training distribution or pipeline is eligible for release.
"""
from __future__ import annotations
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import hashlib
from pathlib import Path

from scripts.campaign_20261110.acceptance import artifact, read
from scripts.campaign_20261110.contracts import Perturbations, ScaleState, digest, file_digest, require, write_json_atomic
from scripts.campaign_20261110.controller_admission import source_proof
from scripts.campaign_20261110.export_release import ArtifactAudit, normalize_record, template_from
from scripts.campaign_20261110.production_gate import read_environment, read_policy
from scripts.campaign_20261110.reservation_policy import active_policy
from scripts.campaign_20261110.storage import atomic_directory_lock
from scripts.campaign_20261110.training_eligibility import coverage_facts


def scales_match(actual, expected):
    return isinstance(actual, (list, tuple)) and len(actual) == 3 and all(
        abs(float(a)-float(b)) <= 1e-6*max(1.,abs(float(b))) for a,b in zip(actual,expected))


def verify_scale_provenance(template, perturbations, raw, reports):
    """Original asset defaults, applied scales and live observations stay distinct."""
    plan=Perturbations({k:ScaleState(**v) for k,v in perturbations['scales'].items()},
                      tuple(perturbations['removed_objects']),perturbations['lighting'])
    plan.validate('train', [o.instance for o in template.objects])
    require(set(plan.scales)=={o.instance for o in template.objects},'canary scale object set differs')
    require(digest(raw['campaign']['perturbations'])==digest(asdict(plan)),'canary native perturbations differ from configuration')
    observed=raw['campaign']['runtime'].get('observed_scales',{})
    result={}
    for obj in template.objects:
        scale=plan.scales[obj.instance]
        require(tuple(scale.default)==tuple(obj.default_scale),'canary lost original asset default scale')
        require(scales_match(observed.get(obj.instance),scale.applied),'canary native applied scale differs')
        for report in reports:
            restored=report.get('loader',{}).get('campaign_restoration',{})
            require(restored.get('scale_verified') is True and
                    scales_match(restored.get('observed_scales',{}).get(obj.instance),scale.applied),
                    'canary report lacks exact observed scale restoration')
            require(restored.get('scene_augmentation') is bool(plan.removed_objects),
                    'canary scene-removal restoration does not match request')
            require(restored.get('lighting_augmentation') is bool(plan.lighting),
                    'canary lighting restoration does not match request')
        result[obj.instance]={'default':list(obj.default_scale),'applied':list(scale.applied),
                              'native_observed':observed[obj.instance],'mode':scale.mode}
    return result


def audit_canary(job, gate_path, *, root, started_at):
    root,job=Path(root).resolve(),Path(job).resolve()
    policy,policy_sha=read_policy(root)
    return _audit_canary(job,gate_path,root=root,started_at=started_at,
                         policy=policy,policy_sha=policy_sha)


def audit_historical_canary(job, gate_path, *, root, started_at, policy_reference):
    """Re-audit completed evidence under its exact original policy bytes.

    This function cannot publish, authorize execution or release production.
    The ordinary publication path always reads the current policy.
    """
    root=Path(root).resolve();path=Path(policy_reference['path']).resolve()
    history=root/'infra/training_production_gate'
    require(path.is_relative_to(history) and path.is_file(),
            'historical canary policy bytes missing or changed')
    payload=path.read_bytes()
    require(hashlib.sha256(payload).hexdigest()==policy_reference['sha256'],
            'historical canary policy bytes missing or changed')
    policy=json.loads(payload)
    require(policy.get('schema')=='isg-training-production-gate-v1'
            and policy.get('campaign_id')=='campaign_20261110',
            'historical canary policy identity differs')
    return _audit_canary(Path(job).resolve(),gate_path,root=root,started_at=started_at,
                         policy=policy,policy_sha=policy_reference['sha256'])


def _audit_canary(job, gate_path, *, root, started_at, policy, policy_sha):
    env=read_environment(artifact(job/'job.env',root,started_at))
    config_path=artifact(env['ISPATIALGYM_CAMPAIGN_CONFIG'],root,started_at)
    config=read(config_path);config_sha=file_digest(config_path)
    require(config['campaign_id']==policy['campaign_id'] and config['split']=='train'
            and config['template']['split']=='train','canary is not campaign training validation')
    start,count=int(env['START']),int(env['N'])
    # A copied native instance retains its original canary authorization. Its
    # exact collector-only scope is stronger than the still-active sampler
    # entry and must be verified without counting the same approval twice.
    existing=[a for a in policy.get('allowed_existing_canary_reviews',[])
        if Path(a['job_dir']).resolve()==job]
    authorizations=existing if existing else policy.get('allowed_canaries',[])
    matches=[a for a in authorizations if a.get('purpose')=='training_pipeline_canary'
             and a.get('config_sha256')==config_sha and (a.get('start'),a.get('count'))==(start,count)]
    require(len(matches)==1,'canary lacks one exact explicit authorization')
    authorization=matches[0]
    if authorization.get('review_manifest'):
        from scripts.campaign_20261110.training_existing_canaries import check_existing_review
        require(check_existing_review(root,job,env,policy,kind='source_validation') is not None,
                'existing native canary review not authorized')
    require(0<count<=min(24,int(authorization['max_instances'])),'canary exceeds authorized bound')
    authorization_path=artifact(authorization['authorization_path'],root,started_at)
    require(file_digest(authorization_path)==authorization['authorization_sha256'],'canary authorization changed')
    declared=read(authorization_path)
    from .training_canary_scope import validate_scope
    if config.get('purpose') == 'training_production':
        validate_scope(config, declared)
    require(all(declared.get(k)==authorization[k] for k in ('purpose','config_sha256','start','count')),
            'canary authorization identity differs')
    # This is an explicitly authorized canary audit, not generic queue
    # admission. Native held-object tasks use their own region author, while
    # retaining the same exact demo/replay, split and artifact checks here.
    proof=source_proof(gate_path,root=root,started_at=started_at,campaign_id=policy['campaign_id'],
                       permitted_split='train')
    require(proof['source_job']==str(job) and proof['config']['sha256']==config_sha,
            'canary needs its own exact-job physics replay')
    gate=read(gate_path);tro_path=Path(gate['instance']);raw=read(tro_path)['ispatialgym']
    block=raw['campaign'];iid=raw['instance_id']
    expiry=datetime.fromisoformat(authorization['expires_at'].replace('Z','+00:00')).timestamp()
    require(authorization_path.stat().st_mtime <= block['created_at'] < expiry,
            'canary native execution falls outside its authorization window')
    require(start<=int(iid.rsplit('__',1)[-1])<start+count,'canary instance index outside authorized range')
    template=template_from(config['template'])
    reports=[read(e['report_path']) for e in proof['evidence']]
    scales=verify_scale_provenance(template,config['perturbations'],raw,reports)
    sidecar=artifact(tro_path.with_name(iid+'.oracle.json'),root,started_at)
    demo=Path(gate['demo']);report=demo.parent/'meta.json'
    paths={'tro':tro_path,'sidecar':sidecar,'demo':demo,'report':report}
    row={'instance_id':iid,'task':template.task,'class_id':template.class_id,'split':'train','sampling_split':'train',
         'contract':asdict(template),'group_id':block.get('goal_group_id'),'validation_key':config['validation_key'],
         'artifacts':{k:{'path':str(p),'sha256':file_digest(p)} for k,p in paths.items()},
         'gate':{'path':str(Path(gate_path).resolve()),'sha256':file_digest(gate_path)},
         'evidence':proof['evidence'],'allocation_source':'training_canary_validation_only'}
    active=active_policy(root);require(active is not None,'canary export needs active reservation policy')
    exported,_=normalize_record(row,root=root,started_at=started_at,templates={},reservations=active[1],
        reservation_hashes=active[2],auditor=ArtifactAudit(root,started_at))
    exported.update(dataset_eligible=False,accepted=False,validation_category='training_pipeline_canary')
    coverage=coverage_facts({'room':f'{template.scene}/{template.room}','validation_key':config['validation_key']},raw,asdict(template))
    return {'schema':'isg-training-canary-review-v1','campaign_id':policy['campaign_id'],
        'instance_id':iid,'task':template.task,'class_id':template.class_id,'validation_passed':True,
        'accepted':False,'eligible_for_benchmark':False,'quota_consumed':0,
        'reason':'bounded pipeline validation; complete pipeline and exact production-template eligibility remain required',
        'source_job':str(job),'authorization':authorization,'gate_policy_sha256':policy_sha,
        'source_proof':proof,'scale_provenance':scales,'coverage':coverage,'export_record':exported,
        'unverified_pipeline_checks':['batched replay/render contamination','complete distribution coverage','production release']}


def committed_equivalent(root, review):
    """Keep the immutable review after a full current-policy re-audit.

    A policy observation hash alone is not new physical or authorization
    evidence. Any other changed field still needs ordinary publication.
    """
    directory=Path(root)/'validation/training_canaries';ledger=directory/'ledger.json'
    if not ledger.exists():return None
    previous=read(ledger).get('records',{}).get(review['instance_id'])
    if not previous:return None
    path=Path(previous['review']).resolve()
    require(path.is_relative_to((directory/'reviews').resolve()) and
            file_digest(path)==previous['sha256'],'committed canary review changed')
    saved=read(path)
    comparable=lambda value:{k:v for k,v in value.items() if k!='gate_policy_sha256'}
    if digest(comparable(saved))!=digest(comparable(review)):return None
    require(previous=={'review':str(path),'sha256':file_digest(path),'instance_id':review['instance_id'],
        'task':review['task'],'class_id':review['class_id'],'validation_passed':True,
        'eligible_for_benchmark':False,'quota_consumed':0},'committed canary index changed')
    return previous


def publish_canary(job, gate, *, root, started_at):
    root=Path(root).resolve();review=audit_canary(job,gate,root=root,started_at=started_at)
    previous=committed_equivalent(root,review)
    if previous:return previous
    key=digest(review);directory=root/'validation/training_canaries';directory.mkdir(parents=True,exist_ok=True)
    path=directory/'reviews'/(key+'.json')
    ledger_path=directory/'ledger.json'
    # Re-observation has already re-audited physical evidence above. Returning
    # an identical committed record needs no writer lock or ledger rewrite.
    if path.exists() and ledger_path.exists():
        require(digest(read(path))==key,'immutable canary review changed')
        previous=read(ledger_path).get('records',{}).get(review['instance_id'])
        expected={'review':str(path),'sha256':file_digest(path),'instance_id':review['instance_id'],
                  'task':review['task'],'class_id':review['class_id'],'validation_passed':True,
                  'eligible_for_benchmark':False,'quota_consumed':0}
        if previous==expected:return expected
    with atomic_directory_lock(directory/'ledger.lockdir'):
        if path.exists():require(digest(read(path))==key,'immutable canary review changed')
        else:write_json_atomic(path,review)
        ledger_path=directory/'ledger.json'
        ledger=read(ledger_path) if ledger_path.exists() else {'schema':'isg-training-canary-ledger-v1',
            'campaign_id':review['campaign_id'],'records':{},'accepted':0,'eligible_accepted':0,'quota_consumed':0}
        record={'review':str(path),'sha256':file_digest(path),'instance_id':review['instance_id'],
                'task':review['task'],'class_id':review['class_id'],'validation_passed':True,
                'eligible_for_benchmark':False,'quota_consumed':0}
        previous=ledger['records'].get(review['instance_id'])
        if previous and previous!=record:ledger.setdefault('history',[]).append(previous)
        ledger['records'][review['instance_id']]=record;ledger['validated_instances']=len(ledger['records'])
        ledger['observed_at']=datetime.now(timezone.utc).isoformat();write_json_atomic(ledger_path,ledger)
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('root','job','gate'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--started-at',type=float,required=True);a=p.parse_args()
    print(json.dumps(publish_canary(a.job,a.gate,root=a.root,started_at=a.started_at)))
