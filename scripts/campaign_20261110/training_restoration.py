"""Verify actual batched restoration observations; this module cannot release production.

Shared machinery, the applicable task path and the exact configuration have
separate proofs. Other tasks/templates may remain unvalidated. A diversity
ticket is an additional dispatch requirement, not supplied by this module.
"""
from __future__ import annotations
import math
from pathlib import Path
import json
from scripts.campaign_20261110.acceptance import audit_job,report_passed
from scripts.campaign_20261110.contracts import digest,file_digest,require
from scripts.campaign_20261110.training_canaries import audit_canary


def checked(root,reference):
    path=Path(reference['path']).resolve()
    require(path.is_relative_to(Path(root).resolve()) and file_digest(path)==reference['sha256'],
            'training release evidence missing or changed')
    if 'jsonl_index' in reference:
        index=reference['jsonl_index'];require(isinstance(index,int) and index>=0,'invalid evidence row')
        return json.loads(path.read_text().splitlines()[index])
    return json.loads(path.read_text())


def validated_canary(root,reference,started_at):
    value=checked(root,reference)
    require(value['schema']=='isg-training-canary-review-v1' and value['validation_passed'] is True
            and value['accepted'] is False and value['eligible_for_benchmark'] is False,
            'release needs a separately audited canary, not a label')
    actual=audit_canary(value['source_job'],value['source_proof']['source_gate']['path'],root=root,started_at=started_at)
    require(digest(actual['export_record'])==digest(value['export_record']) and
            digest(actual['scale_provenance'])==digest(value['scale_provenance']),
            'canary physical or export evidence changed')
    return actual


def same_loaded_state(first,last):
    """Compare actual native reads after returning to A; no success label used."""
    require(first.keys()==last.keys(),'restoration observation fields differ')
    require(first['objects'].keys()==last['objects'].keys(),'restoration changed object inventory')
    for name,a in first['objects'].items():
        b=last['objects'][name]
        require(all(a[k]==b[k] for k in ('category','model','fixed_base')),'restoration object identity changed')
        require(all(abs(float(x)-float(y))<=1e-6 for x,y in zip(a['scale'],b['scale'])),'restoration scale contaminated')
        require(math.dist(a['position'],b['position'])<=.002,'restoration object pose contaminated')
        qa,qb=a['orientation'],b['orientation']
        dot=abs(sum(float(x)*float(y) for x,y in zip(qa,qb)))
        norms=math.sqrt(sum(float(x)**2 for x in qa)*sum(float(x)**2 for x in qb))
        require(norms>0 and 2*math.acos(min(1.,dot/norms))<=1e-4,'restoration object orientation contaminated')
        for key,tolerance in [('joint_positions',1e-5),('joint_velocities',1e-4),('linear_velocity',1e-4),('angular_velocity',1e-4)]:
            x,y=a[key],b[key]
            if x is None or y is None:
                require(x is y,'restoration motion observation missing')
            else:
                require(len(x)==len(y) and all(abs(float(u)-float(v))<=tolerance for u,v in zip(x,y)),
                        'restoration '+key+' contaminated')
    require(first['lights'].keys()==last['lights'].keys(),'restoration fixture inventory changed')
    for name,a in first['lights'].items():
        require(abs(float(a)-float(last['lights'][name]))<=1e-5*max(1.,abs(float(a))),'restoration fixture intensity contaminated')
    a,b=first['skybox'],last['skybox']
    require(abs(a['intensity']-b['intensity'])<=1e-5*max(1.,abs(a['intensity'])) and
            all(abs(x-y)<=1e-5 for x,y in zip(a['color'],b['color'])),'restoration skybox contaminated')


def verify_batch(root,reference,started_at,kind):
    probe=checked(root,reference)
    require(probe['schema']=='isg-training-restoration-probe-v1' and probe['kind']==kind,
            'missing real production-path batch probe')
    entries=probe['entries'];require(len(entries)==3,'batch probe must execute A/B/A')
    observations=[];instances=[];tasks=set()
    for entry in entries:
        source=validated_canary(root,entry['canary_review'],started_at)
        exported=source['export_record'];report=checked(root,entry['report']);tasks.add(source['task'])
        if kind=='physics_action_open_loop':
            report_passed(report,'replay')
            require(report['instance_id']==source['instance_id'] and
                    Path(report['replay']).resolve()==Path(exported['artifacts']['demo']['path']).resolve(),
                    'batch replay did not execute this exact demo')
            restoration=report['loader']['campaign_restoration']
        else:
            require(kind=='kinematic_observation_rendering' and report.get('tag')=='REPLAY_DEMO'
                    and report.get('kept') is True and isinstance(report.get('episode_index'),int)
                    and Path(report['demo']).name==source['instance_id'],'batch render episode failed')
            restoration=report['campaign_restoration']
            require(entry.get('outputs'),'render probe has no actual output artifact')
            for output in entry['outputs']:
                path=Path(output['path']).resolve()
                require(path.is_relative_to(Path(root).resolve()) and path.stat().st_size>0 and
                        file_digest(path)==output['sha256'],'render probe output missing or changed')
        require(restoration.get('scale_verified') is True,'batch scale restoration failed')
        observation=restoration.get('native_observation',{})
        require(observation.get('schema')=='isg-native-restoration-observation-v1' and
                observation.get('simulator_module','').startswith('omnigibson.') and observation.get('scheduler_job_id'),
                'batch evidence is not a native scheduled simulator observation')
        require(observation['validation_key']==exported['validation_key'] and
                observation['perturbations_sha256']==digest(exported['perturbations']),
                'batch observation configuration differs')
        require(observation['loaded_state_sha256']==digest(observation['loaded_state']),'native state observation changed')
        observations.append(observation);instances.append(source['instance_id'])
    require(instances[0]==instances[2] and instances[0]!=instances[1],'batch did not execute distinct A/B/A inputs')
    require(len({o['process_session'] for o in observations})==1 and
            len({o['sim_runtime_id'] for o in observations})==1 and
            len({o['scene_runtime_id'] for o in observations})==1,'batch did not exercise one reused native world')
    require(observations[0]['sequence']<observations[1]['sequence']<observations[2]['sequence'],'batch observation order differs')
    same_loaded_state(observations[0]['loaded_state'],observations[2]['loaded_state'])
    return {'reference':reference,'tasks':sorted(tasks),'instances':instances,'process_session':observations[0]['process_session']}

