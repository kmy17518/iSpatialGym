"""Recheck stage-specific original dependencies; an execution ledger cannot authorize work."""
from datetime import datetime,timezone
import json
from pathlib import Path
from scripts.campaign_20261110.contracts import file_digest,require


def verify_execution_dependencies(root,configuration_sha256,stage,*,start=None,count=None):
    require(stage in ('native_generation','collection','physics_replay'),'unknown execution prerequisite stage')
    root=Path(root).resolve();path=root/'infra/execution_ledger.json'
    if not path.exists():return {'status':'not_indexed','authorization_granted':False,'original_gate_required':True}
    value=json.loads(path.read_text())
    require(value.get('schema')=='isg-campaign-execution-ledger-v1'and value.get('campaign_id')=='campaign_20261110',
            'execution dependency ledger identity differs')
    rows=[]
    for row in value.get('rows',[]):
        if row.get('configuration_sha256')!=configuration_sha256:continue
        if row.get('kind')=='pipeline_canary'and (row.get('start'),row.get('count'))!=(start,count):continue
        rows.append(row)
    # Resolve the compact indexes only by exact materialized configuration hash.
    # Null proposal hashes never match an executable configuration.
    indexes=value.get('indexed_sources',{})
    if not rows and indexes.get('configurations',{}).get('sha256'):
        from scripts.campaign_20261110.training_eligibility import immutable_evidence_json
        def indexed(ref):
            target=Path(ref['path']).resolve()
            require(target.is_relative_to(root),'execution source index outside campaign')
            return immutable_evidence_json(target,ref['sha256'])
        configurations=indexed(indexes['configurations'])
        values=configurations.get('configurations',configurations.get('rows',[]))
        if isinstance(values,dict):values=list(values.values())
        matches=[row for row in values if row.get('configuration_sha256')==configuration_sha256]
        if matches:
            dependency_source=indexed(indexes['dependencies'])
            values=dependency_source.get('artifact_requests',dependency_source.get('rows',[]))
            if isinstance(values,dict):values=list(values.values())
            by_id={dependency['id']:dependency for dependency in values}
            for original in matches:
                dependencies=[]
                for key in original.get('dependency_ids',original.get('dependencies',[])):
                    key=key.get('id')if isinstance(key,dict)else key
                    require(key in by_id,'exact configuration dependency is missing from index')
                    dependency=by_id[key]
                    dependencies.append({**dependency,'state':dependency.get('state',dependency.get('status')),
                        'reason':dependency.get('reason',dependency.get('next_action')),
                        'source_ref':indexes['dependencies']})
                require(dependencies,'materialized configuration has no dependency chain')
                rows.append({**original,'dependencies':dependencies})
    # Missing exact materialization cannot replace the existing immutable authorization.
    if not rows:return {'status':'not_indexed','authorization_granted':False,'original_gate_required':True,
                        'ledger':{'path':str(path),'sha256':file_digest(path)}}
    observed=datetime.fromisoformat(value['observed_at'].replace('Z','+00:00'))
    age=(datetime.now(timezone.utc)-observed).total_seconds()
    require(-60<=age<=600,'execution dependency observation stale or future-dated')
    verified=[]
    for row in rows:
        for dependency in row.get('dependencies',[]):
            require(isinstance(dependency,dict),'execution dependency has no explicit stage scope')
            if stage not in dependency.get('required_for',[]):continue
            state=dependency.get('state')
            if state=='not_applicable':
                require(bool(dependency.get('reason')),'not-applicable dependency lacks reason');continue
            require(state in ('verified','positively_verified'),'required execution dependency is not verified: '+str(dependency.get('id')))
            ref=dependency.get('artifact')or{}
            original=Path(ref.get('path','')).resolve()
            require(original.is_relative_to(root)and original.is_file()and file_digest(original)==ref.get('sha256'),
                    'execution dependency original changed: '+str(dependency.get('id')))
            for proof in dependency.get('evidence',[]):
                evidence_path=Path(proof['path']).resolve()
                require(evidence_path.is_relative_to(root)and file_digest(evidence_path)==proof['sha256'],
                        'execution dependency verification evidence changed')
            verified.append({'id':dependency['id'],'artifact':ref,'source_ref':dependency.get('source_ref')})
    return {'status':'exact_originals_checked','stage':stage,'rows':[row['id']for row in rows],
            'dependencies':verified,'ledger':{'path':str(path),'sha256':file_digest(path)},
            'authorization_granted':False,'original_gate_required':True}
