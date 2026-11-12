import json
from copy import deepcopy
import pytest
from scripts.campaign_20261110 import training_mass_restart_verification as m
from scripts.campaign_20261110.contracts import file_digest,digest,ContractError
from scripts.campaign_20261110.tests.test_training_sequential_augmentation import observed


def fixture(root,monkeypatch,fault=None):
 def put(name,value):
  p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return {'path':str(p),'sha256':file_digest(p)}
 stamp=put('runtime/.snapshot_commit','exact');sources=[];refs=[]
 for member in (0,1):
  sources.append({'instance_id':f'i{member}','export_record':{'validation_key':{'exact':member},'perturbations':{},'artifacts':{'demo':{'path':str(root/f'demo{member}')}}}})
  refs.append(put(f'source{member}.json',sources[-1]))
 compatibility=put('compatibility.json',{'exact':True});controls={}
 for name,control,order in [('fresh_B','FRESH_B',[1]),('ABA',None,[0,1,0])]:
  q={'sequential_mass_restart':True,'sequential_anchor_rebase':True,'sequential_augmentation':True,'diagnostic_control':control,'production_released':False,'benchmark_credit':0,'canaries':refs,'probe_runtime_sha256':stamp['sha256']}
  if fault=='request_option'and name=='ABA':q['unreviewed_option']=True
  request=put(name+'/request.json',q);entries=[]
  for ordinal,member in enumerate(order):
   obs=observed(ordinal+1);obs.update(process_session=name,simulator_module='omnigibson.sim',scheduler_job_id='123',validation_key={'exact':member},perturbations_sha256=digest({}))
   step={'native_control_calls_observed':1,'physics_steps_added':0,'before':deepcopy(obs),'after':deepcopy(obs)}
   restore={'scale_verified':True,'next_native_control_step_observation':step}
   for key in ('native_reset_boundary_observation','native_reset_step_completed_observation','native_final_pre_action_observation'):restore[key]={'observation':deepcopy(obs)}
   r={'mode':'replay','status':'success','success':True,'qa':{'enabled':True,'hard_triggered':[]},'instance_id':f'i{member}','replay':str(root/f'demo{member}'),'loader':{'reused_world':ordinal>0,'campaign_restoration':restore}}
   if name=='ABA'and ordinal==1:
    if fault=='qa':r['qa']['hard_triggered']=['goal_occlusion']
    if fault=='extra_step':step['physics_steps_added']=1
    if fault=='wrong_source':r['replay']=str(root/'other_demo')
    if fault=='world_reload':r['loader']['reused_world']=False
    if fault=='boundary_mass':
     ctx=restore['native_reset_step_completed_observation']['observation'];ctx['native_dynamics_context']['dynamic_roots']['target']['mass']=3;ctx['native_dynamics_context_sha256']=digest(ctx['native_dynamics_context'])
   report=put(f'{name}/report{ordinal}.json',r);entries.append({'report':report,'canary_review':refs[member],'returncode':0})
  probe={'request':request,'probe_runtime':stamp,'runtime_compatibility':compatibility,'kind':'physics_action_open_loop','entries':entries,'world_construction':{'method':'omnigibson.Environment','attempted':1,'completed':1}}
  if fault=='missing_episode'and name=='ABA':probe['entries'].pop()
  if fault=='two_worlds'and name=='ABA':probe['world_construction']['completed']=2
  controls[name]={'probe':put(name+'/probe.json',probe),'execution':put(name+'/execution.json',{})}
 monkeypatch.setattr(m,'inputs',lambda root,path:(json.loads(__import__('pathlib').Path(path).read_text()),[]))
 monkeypatch.setattr(m,'verify_probe_execution',lambda *a:{'strict_terminal':True})
 monkeypatch.setattr(m,'verify_runtime_compatibility',lambda *a:compatibility)
 monkeypatch.setattr(m,'validated_canary',lambda root,ref,start:sources[refs.index(ref)])
 cold=[]
 monkeypatch.setattr(m,'verify_original_cold',lambda root,item,ref,current,start:cold.append(ref)or{'exact':True})
 monkeypatch.setattr(m,'observed_augmentation_change',lambda *a:{'actual_change_verified':True})
 inp=put('input.json',{'schema':'isg-sequential-mass-restart-verification-input-v1','controls':controls,'original_cold':{'fresh_A':{},'fresh_B':{}}})
 return inp,cold


def test_targeted_pass_preserves_cold_checks_and_does_not_grant_full_matrix(tmp_path,monkeypatch):
 inp,cold=fixture(tmp_path,monkeypatch);r=m.verify(tmp_path,inp,0)
 assert r['status']=='verified'and len(cold)==2
 assert not r['full_matrix_verified']and not r['cleanup_controls_verified']and not r['production_released']
 assert all(v['world_load_count']==1 for v in r['controls'].values())


@pytest.mark.parametrize('fault',['request_option','qa','extra_step','wrong_source','world_reload','boundary_mass','missing_episode','two_worlds'])
def test_targeted_still_requires_exact_sources_native_qa_and_physical_boundaries(tmp_path,monkeypatch,fault):
 inp,_=fixture(tmp_path,monkeypatch,fault)
 with pytest.raises(ContractError):m.verify(tmp_path,inp,0)
