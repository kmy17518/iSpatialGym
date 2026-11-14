import copy
from types import SimpleNamespace
import pytest
from scripts.campaign_20261110 import training_loading_boundary_contract as contract


def request(control='ABA'):
    return {'schema':'isg-retained-loading-boundary-request-v1','campaign_id':'campaign_20261110',
            'boundary_kind':'room','control':control,'production_released':False,'benchmark_credit':0,
            'timeout_seconds':1800,'max_native_episodes':len(contract.ORDERS[control]),
            'sources':[{'path':'A','sha256':'a'*64},{'path':'B','sha256':'b'*64}],
            'compatible_AA_request':{'path':'AA','sha256':'c'*64},'native_runtime':{'path':'runtime','sha256':'d'*64}}


def completed(control='ABA'):
    q=request(control);plan=contract.segment_plan(q);children=[];segments=[]
    for ordinal,spec in enumerate(plan):
        children.append({'request':spec['request'],'status':'native_mechanism_controls_completed',
                         'native_runtime':q['native_runtime'],'world_construction':{'attempted':1,'completed':1},
                         'entries':[{'returncode':0,'result':{'success':True},'expected_abort':False}for _ in range(spec['episodes'])]})
        boundary={'required_returncode':75,'reuse_allowed':False,'physical_steps_added':0,'world_mutations':0,
                  'signatures':['A','B'],'template_compatible':False} if ordinal+1<len(plan)else None
        segments.append({'ordinal':ordinal,'source':spec['source'],'process_pid':100+ordinal,
                         'returncode':75 if boundary else 0,'boundary':boundary})
    observed={'schema':'isg-retained-loading-boundary-observation-v1','status':'completed',
              'native_runtime':q['native_runtime'],'benchmark_credit':0,'production_released':False,
              'segments':segments,'world_loads':len(plan),'native_episodes':q['max_native_episodes']}
    return q,observed,children


def test_three_room_processes_are_never_reported_as_one_world_reuse():
    q,observation,children=completed()
    audit=contract.validate_sequence(q,observation,children)
    assert audit['world_loads']==3 and audit['room_changes']==2 and audit['one_world_reuse']is False
    q,observation,children=completed('AA')
    audit=contract.validate_sequence(q,observation,children)
    assert audit['world_loads']==1 and audit['native_episodes']==2 and audit['one_world_reuse']is True


@pytest.mark.parametrize('mutation',[
    lambda q,o,c:o['segments'][1].update(process_pid=100),
    lambda q,o,c:o.update(world_loads=1),
    lambda q,o,c:c[1]['entries'][0]['result'].update(success=False),
    lambda q,o,c:c[1].update(native_runtime={'path':'old','sha256':'e'*64}),
    lambda q,o,c:c[1].update(request=q['sources'][0]),
    lambda q,o,c:o['segments'][0]['boundary'].update(reuse_allowed=True),
    lambda q,o,c:o['segments'][0]['boundary'].update(signatures=['A','A'],template_compatible=True),
    lambda q,o,c:c[0].update(world_construction={'attempted':2,'completed':2}),
    lambda q,o,c:q.update(max_native_episodes=2),
    lambda q,o,c:q.update(timeout_seconds=1801),
    lambda q,o,c:q.update(production_released=True),
])
def test_incomplete_or_inferred_boundary_evidence_fails(mutation):
    q,o,c=completed();mutation(q,o,c)
    with pytest.raises(ValueError):contract.validate_sequence(q,o,c)


def test_native_compatibility_decision_uses_both_actual_apis():
    calls=[]
    runner=SimpleNamespace(EXIT_QUEUE_RESTART=75,
       instance_signature=lambda p:('same' if p!='other_room'else'other'),
       resolve_instance=lambda p:(None,'directory_'+p),
       template_compatible=lambda *a:(calls.append(a)or(True,'native comparison')))
    assert contract.native_boundary(runner,'A','B')['reuse_allowed']is True
    result=contract.native_boundary(runner,'A','other_room')
    assert result['reuse_allowed']is False and result['required_returncode']==75 and len(calls)==2
    runner.template_compatible=lambda *a:(False,'different saved scene')
    assert contract.native_boundary(runner,'A','B')['required_returncode']==75


def test_execution_requires_exact_clean_scheduler_row(tmp_path):
    import json
    submission=tmp_path/'submission.json';submission.write_text(json.dumps({'job_id':1234}))
    scheduler=tmp_path/'sacct.txt';scheduler.write_text('1234|COMPLETED|0:0|800\n1234.batch|COMPLETED|0:0|800\n')
    evidence={'submission':contract.ref(submission),'scheduler':contract.ref(scheduler)}
    assert contract.executed_job(tmp_path,{'scheduler_job_id':'1234'},evidence)=='1234'
    scheduler.write_text('1234|TIMEOUT|0:15|1800\n');evidence['scheduler']=contract.ref(scheduler)
    with pytest.raises(ValueError,match='complete cleanly'):contract.executed_job(tmp_path,{'scheduler_job_id':'1234'},evidence)
    with pytest.raises(ValueError,match='allocation identity'):contract.executed_job(tmp_path,{'scheduler_job_id':'999'},evidence)


def test_input_binding_keeps_sources_runtime_and_actual_room_metadata(tmp_path):
    import json
    runtime=tmp_path/'code/candidate';runtime.mkdir(parents=True)
    (runtime/'.snapshot_commit').write_text('candidate\n');identity=contract.ref(runtime/'.snapshot_commit')
    q=request('AA');q['native_runtime']=identity;sources={};references=[]
    for ordinal in range(2):
        template=tmp_path/f'template{ordinal}';(template/'instances').mkdir(parents=True)
        (template/'template_meta.json').write_text(json.dumps({'scene':f'room{ordinal}','load_room_instances':['room_0']}))
        tro=template/'instances/source.json';tro.write_text('{}')
        path=tmp_path/f'source{ordinal}.json';path.write_text('{}');references.append(contract.ref(path))
        sources[str(path)]=({'control':'FRESH_A','native_runtime':identity,'physical_delta':{'declared':True}},
                          {'instance_id':f'instance{ordinal}','source_tro':contract.ref(tro)}, {})
    aa=tmp_path/'AA.json';aa.write_text('{}');q.update(sources=references,compatible_AA_request=contract.ref(aa))
    sources[str(aa)]=({**sources[str(tmp_path/'source0.json')][0],'control':'AA'},sources[str(tmp_path/'source0.json')][1],{})
    path=tmp_path/'request.json';path.write_text(json.dumps(q))
    reader=lambda root,path,**kwargs:sources[str(path)]
    value,rows,plan=contract.inputs(tmp_path,path,runtime=runtime,source_reader=reader)
    assert len(plan)==1 and rows[0]['room']['scene']=='room0' and rows[1]['room']['scene']=='room1'
    (tmp_path/'template1/template_meta.json').write_text((tmp_path/'template0/template_meta.json').read_text())
    with pytest.raises(ValueError,match='distinct retained worlds'):
        contract.inputs(tmp_path,path,runtime=runtime,source_reader=reader)


def test_retained_cold_runtime_bridge_rejects_physics_or_unreviewed_payloads(tmp_path):
    import json
    old=tmp_path/'old';new=tmp_path/'new'
    helper='scripts/campaign_20261110/training_loading_boundary_probe.py'
    inventories=[]
    for directory,body in [(old,'old'),(new,'exit-repair')]:
        (directory/'ispatialgym').mkdir(parents=True);(directory/'ispatialgym/physics.py').write_text('unchanged')
        path=directory/helper;path.parent.mkdir(parents=True);path.write_text(body)
        (directory/'.snapshot_commit').write_text(body)
        inventories.append({str(p.relative_to(directory)):contract.file_digest(p)for p in directory.rglob('*.py')})
    q={'native_runtime':contract.ref(new/'.snapshot_commit')};cold={'native_runtime':contract.ref(old/'.snapshot_commit')}
    proof={'schema':'isg-loading-supervision-only-runtime-v1','cold_runtime':cold['native_runtime'],
           'candidate_runtime':q['native_runtime'],'physics_changes':[],'benchmark_credit':0,
           'cold_python_payloads':inventories[0],'candidate_python_payloads':inventories[1],
           'reviewed_changed_payloads':{helper:{'cold':inventories[0][helper],'candidate':inventories[1][helper]}}}
    path=tmp_path/'bridge.json';path.write_text(json.dumps(proof));q['cold_runtime_compatibility']=contract.ref(path)
    assert contract.cold_runtime(tmp_path,q,cold)==old
    (new/'ispatialgym/physics.py').write_text('changed')
    with pytest.raises(ValueError,match='inventory changed'):contract.cold_runtime(tmp_path,q,cold)
    proof['candidate_python_payloads']['ispatialgym/physics.py']=contract.file_digest(new/'ispatialgym/physics.py')
    proof['reviewed_changed_payloads']['ispatialgym/physics.py']={'cold':inventories[0]['ispatialgym/physics.py'],
        'candidate':proof['candidate_python_payloads']['ispatialgym/physics.py']}
    path.write_text(json.dumps(proof));q['cold_runtime_compatibility']=contract.ref(path)
    with pytest.raises(ValueError,match='physical payload change'):contract.cold_runtime(tmp_path,q,cold)
