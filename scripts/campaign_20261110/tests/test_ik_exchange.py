import copy
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import pytest
from scripts.campaign_20261110.ik_compatibility import CAMPAIGN,SCHEMA,digest,semantic
from scripts.campaign_20261110 import ik_exchange as X
from ispatialgym.sampling import exact_exchange as E


def manifest():
 identity={'campaign_id':CAMPAIGN,'cache_kind':'exact_ik_v3','branch_limit':6,'branch_min_rad':.2}
 return {'schema':SCHEMA,'campaign_id':CAMPAIGN,'compatibility_sha256':digest(identity),'compatible_identity':identity}

def entry(m,branch=0):
 query={'scene':'room','arm':'right','poses':[[0.,0.,0.,0.,0.,0.,1.]]};key=E.query_key(query)
 value=E.seal({'ok':True,'q':[branch,0.],'qs':[[branch,0.]],'_exchange':{'schema':E.ENTRY_SCHEMA,'campaign_id':CAMPAIGN,
  'compatibility_sha256':m['compatibility_sha256'],'query':query,'key':key,'arm_indices':[0],'dof':2,'publishers':[str(branch)]}})
 return key,value

def roots(tmp_path):
 root=tmp_path/CAMPAIGN
 return root/'minyeong',root/'exchange',root/'shiyuc'

def test_incremental_roundtrip_and_new_branches_for_existing_key(tmp_path):
 m=manifest();a,exchange,b=roots(tmp_path);key,value=entry(m)
 E.install(X.directory(a,m),value,key,m)
 first=X.publish(a,exchange,m);assert first['entries']==1 and first['branches']==1 and first['complete']
 second=X.publish(a,exchange,m,cursor=first['cursor']);assert second['entries']==0
 imported=X.import_entries(exchange,b,m);assert imported['branches']==1
 _,more=entry(m,1.);E.install(X.directory(a,m),more,key,m)
 update=X.publish(a,exchange,m,cursor=first['cursor']);assert update['entries']==1 and update['branches']==2
 enriched=X.import_entries(exchange,b,m,cursor=imported['cursor']);assert enriched['branches']==2
 assert len(E.read_entry(X.directory(b,m)/(key+'.json'),key,m)['alts'])==1
 assert E.read_entry(X.directory(a,m)/(key+'.json'),key,m)['qs']==value['qs']

def test_concurrent_publication_merges_branches_and_retains_primary(tmp_path):
 m=manifest();a,exchange,b=roots(tmp_path);key,base=entry(m)
 E.install(X.directory(exchange,m),base,key,m)
 with ThreadPoolExecutor(max_workers=4)as pool:
  results=list(pool.map(lambda n:E.install(X.directory(exchange,m),entry(m,float(n))[1],key,m),range(1,6)))
 value=E.read_entry(X.directory(exchange,m)/(key+'.json'),key,m)
 assert value['qs']==base['qs'] and len(value['alts'])==5
 assert {c[-1][0]for c in [value['qs']]+value['alts']}==set(range(6))
 assert not list(X.directory(exchange,m).glob('*.lock'))

def test_branch_limit_and_same_branch_do_not_overwrite_richer_entry(tmp_path):
 m=manifest();key,value=entry(m)
 for n in range(1,10):value=E.merge_entries(value,entry(m,float(n))[1],key,m)
 assert len(value['alts'])==5 and value['qs']==[[0,0.]]
 previous=copy.deepcopy(value);same=E.merge_entries(value,entry(m,.1)[1],key,m)
 assert same['qs']==previous['qs'] and same['alts']==previous['alts']

def test_incompatible_malformed_negative_and_partial_entries_rejected(tmp_path):
 m=manifest();key,value=entry(m)
 for mutate in [lambda v:v.update(ok=False),lambda v:v.update(qs=[[float('nan'),0.]]),
                lambda v:v.update(qs=[]),lambda v:v['_exchange'].update(campaign_id='older'),
                lambda v:v['_exchange'].update(compatibility_sha256='0'*64),
                lambda v:v['_exchange'].update(query={'poses':[None],'arm':'right'})]:
  bad=copy.deepcopy(value);mutate(bad)
  with pytest.raises((ValueError,TypeError)):E.validate_entry(bad,key,m)
 assert E.validate_entry(value,key,m)

def test_no_unsafe_paths_or_foreign_files_exported(tmp_path):
 m=manifest();a,exchange,b=roots(tmp_path);base=X.directory(a,m);base.mkdir(parents=True)
 key,value=entry(m);E.atomic(base/(key+'.json'),value)
 (base/'feasible_sets_v2-any.json').write_text('{}');(base/(key+'.json.tmp')).write_text('{')
 (base/'exact_ik_v3-invalid.json').write_text('{}')
 packed=X.pack_delta(a,m,tmp_path/'bundle.json');assert packed['entry_count']==1 and len(packed['failures'])==1
 doc=json.loads((tmp_path/'bundle.json').read_text());doc['entries'][0]['key']='../../escape'
 doc['bundle_sha256']=digest({k:v for k,v in doc.items()if k!='bundle_sha256'});E.atomic(tmp_path/'bad.json',doc)
 with pytest.raises(ValueError,match='unsafe'):X.ingest_bundle(tmp_path/'bad.json',exchange,m)
 with pytest.raises(ValueError,match='namespace'):X.import_entries(exchange,tmp_path/'other',m)


def test_corrupted_transport_and_existing_entry_preserved(tmp_path):
 m=manifest();a,exchange,b=roots(tmp_path);key,value=entry(m);E.install(X.directory(a,m),value,key,m)
 packed=X.pack_delta(a,m,tmp_path/'bundle.json')
 with pytest.raises(ValueError,match='transport checksum'):X.ingest_bundle(packed['bundle_path'],exchange,m,expected_sha256='0'*64)
 target=X.directory(exchange,m);target.mkdir(parents=True);(target/(key+'.json')).write_text('{')
 result=X.ingest_bundle(packed['bundle_path'],exchange,m);assert not result['complete']and result['errors']
 assert (target/(key+'.json')).read_text()=='{'


def test_semantic_hash_ignores_comments_docs_not_behavior(tmp_path):
 p=tmp_path/'one.py';p.write_text('"""old docs"""\nx=1 #comment\n');first=semantic(p)[0]
 p.write_text('"""new docs"""\n#more comments\nx=1\n');assert semantic(p)[0]==first
 p.write_text('x=2\n');assert semantic(p)[0]!=first


def test_cold_cache_and_unchanged_nonexact_cache(monkeypatch,tmp_path):
 from ispatialgym.sampling import cache
 monkeypatch.setenv('ISPATIALGYM_SAMPLING_CACHE',str(tmp_path/'ordinary'))
 monkeypatch.delenv('ISPATIALGYM_EXACT_IK_MANIFEST',raising=False)
 cache.cache_put('feasible_sets_v2-local',{'ok':False});assert cache.cache_get('feasible_sets_v2-local')=={'ok':False}
 m=manifest();key,value=entry(m);cache.cache_put(key,{'ok':True,'q':[1,2],'qs':[[1,2]]})
 assert cache.cache_get(key)['ok']is True
 monkeypatch.setenv('ISPATIALGYM_SAMPLING_CACHE','off');assert cache.cache_get(key)is None


def test_lock_timeout_does_not_steal_unproven_owner(tmp_path):
 p=tmp_path/'entry.lock';p.mkdir();(p/'owner.json').write_text('{"host":"other-host","pid":7}')
 with pytest.raises(TimeoutError):
  with E.file_lock(p,timeout=.02):pass
 assert p.exists()


def compatibility_inputs(tmp_path):
 from scripts.campaign_20261110 import ik_compatibility as C
 code=tmp_path/'runtime';code.mkdir()
 for name in C.MODULES:
  p=code/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('value=1\n')
 refs={}
 for name in ('robot_urdf','robot_curobo_yaml','robot_meshes/arm','curobo_implementation/solver'):
  p=tmp_path/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(name)
  refs[name]={'path':str(p),'sha256':C.file_hash(p)}
 source=tmp_path/'scene';source.mkdir();(source/'template.json').write_text('{}')
 scales={'target':{'default':[1.,1.,1.],'applied':[1.,1.,1.],'mode':'fixed'}}
 cfg={'split':'train','template':{'objects':[{'instance':'target','category':'cup','model':'model',
  'ready':True,'asset_sha256':'a'*64,'default_scale':[1.,1.,1.]}],'scene':'house_single_floor','room':'kitchen',
  'scene_setup':{'source_template':str(source),'source_template_sha256':C.file_hash(source/'template.json')}},
  'perturbations':{'scales':scales,'removed_objects':[],'lighting':{}},'validation_key':{'scale_sha256':digest(scales)}}
 config=tmp_path/'config.json';config.write_text(json.dumps(cfg))
 return C,code,config,refs,cfg


def test_compatibility_binds_applied_scale_and_scene_removals(tmp_path):
 C,code,config,refs,cfg=compatibility_inputs(tmp_path)
 baseline=C.build_manifest(code,config,{},refs)
 scale=cfg['perturbations']['scales']['target'];scale.update(applied=[1.,1.,1.1],mode='height')
 cfg['validation_key']['scale_sha256']=digest(cfg['perturbations']['scales']);config.write_text(json.dumps(cfg))
 changed=C.build_manifest(code,config,{},refs)
 assert changed['compatibility_sha256']!=baseline['compatibility_sha256']
 assert changed['compatible_identity']['objects'][0]['applied_scale']==(1.,1.,1.1)
 with pytest.raises(ValueError,match='incompatible'):C.verify_running(baseline,code,config,{})
 cfg['perturbations']['removed_objects']=['background_chair'];config.write_text(json.dumps(cfg))
 assert C.build_manifest(code,config,{},refs)['compatibility_sha256']!=changed['compatibility_sha256']


def test_compatibility_rejects_incomplete_scale_and_changed_solver(tmp_path):
 C,code,config,refs,cfg=compatibility_inputs(tmp_path)
 value=C.build_manifest(code,config,{},refs)
 Path(refs['robot_urdf']['path']).write_text('changed kinematics')
 with pytest.raises(ValueError,match='hash mismatch'):C.verify_running(value,code,config,{})
 Path(refs['robot_urdf']['path']).write_text('robot_urdf')
 cfg['perturbations']['scales']={};cfg['validation_key']['scale_sha256']=digest({});config.write_text(json.dumps(cfg))
 with pytest.raises(ValueError,match='complete exact'):C.build_manifest(code,config,{},refs)


def test_compatibility_identical_content_survives_path_and_comment_changes(tmp_path):
 C,code,config,refs,cfg=compatibility_inputs(tmp_path)
 value=C.build_manifest(code,config,{},refs)
 renamed=tmp_path/'other_config.json';renamed.write_text(config.read_text())
 (code/C.MODULES[0]).write_text('#different comment\nvalue=1\n')
 assert C.build_manifest(code,renamed,{},refs)['compatibility_sha256']==value['compatibility_sha256']
 (code/C.MODULES[0]).write_text('value=2\n')
 assert C.build_manifest(code,renamed,{},refs)['compatibility_sha256']!=value['compatibility_sha256']


def test_query_registration_uses_native_cache_tensor_and_numpy_serialization(monkeypatch):
 import numpy as np
 import torch
 from ispatialgym.sampling.cache import cache_key
 monkeypatch.setattr(E,'context',lambda:({},None));monkeypatch.setattr(E,'_QUERIES',{})
 payload={'poses':[[0.,0.,0.,0.,0.,0.,1.]],'arm':'right',
          'seed_arms':torch.tensor([[.123456,.234567]]),'scale':np.float32(.123456)}
 key=cache_key('exact_ik_v3',payload)
 E.register_query(key,payload,[0],2)
 assert E.query_key(E._QUERIES[key]['query'])==key
 assert E._QUERIES[key]['query']['seed_arms']==[.1235,.2346]
 assert E._QUERIES[key]['query']['scale']==float(payload['scale'])
 E.register_query(key,{'changed':True},[0],2)
 assert key not in E._QUERIES  # no native-sampling failure or stale registration
