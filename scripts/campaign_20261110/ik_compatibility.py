"""Content-based compatibility for campaign exact-IK candidate witnesses only."""
from __future__ import annotations
import ast
import hashlib
import json
import os
from pathlib import Path

CAMPAIGN = 'campaign_20261110'
SCHEMA = 'isg-exact-ik-compatibility-v1'
MODULES = (
 'ispatialgym/sampling/v2/ik.py','ispatialgym/sampling/robot_utils.py',
 'ispatialgym/oracle/curobo_planner.py','ispatialgym/oracle/r1pro_model.py',
 'ispatialgym/robot/conventions.py','ispatialgym/sampling/v2/support_split.py',
 'ispatialgym/grasping/contact_policy.py','ispatialgym/grasping/hand_geometry.py',
 'ispatialgym/sampling/geometry.py','ispatialgym/sampling/cache.py','ispatialgym/fastflags.py')

def digest(value):
 return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()

def file_hash(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()

class WithoutDocstrings(ast.NodeTransformer):
 def generic_visit(self,node):
  node=super().generic_visit(node)
  if isinstance(node,(ast.Module,ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
   if node.body and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant) and isinstance(node.body[0].value.value,str):node.body=node.body[1:]
  return node

def semantic(path):
 tree=WithoutDocstrings().visit(ast.parse(Path(path).read_text()))
 return hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest(),tree

def settings_keys(trees):
 keys={'ISPATIALGYM_FAST'}
 for tree in trees:
  for n in ast.walk(tree):
   if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('ISPATIALGYM_'):
    if not any(x in n.value for x in ('CACHE','CONFIG','PATH','ROOT','DIR','DIAG','PROFILE','LOG','STATS')):keys.add(n.value)
   if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('fast_on','fast_value') and n.args and isinstance(n.args[0],ast.Constant) and isinstance(n.args[0].value,str):
    keys.add('ISPATIALGYM_FAST_'+n.args[0].value.upper())
 return sorted(keys)

def build_manifest(code,config,env,robot_refs):
 """robot_refs maps logical names to {path,sha256}; include robot and cuRobo.

 Paths and repository revisions are provenance, not compatibility. Scene/query
 state is additionally bound by each original exact_ik_v3 query key; candidates
 retain all current-world checks when consumed.
 """
 code=Path(code).resolve();config=Path(config).resolve();cfg=json.loads(config.read_text());t=cfg['template']
 if cfg.get('split') not in ('eval_standard','eval_model','eval_category','train'):raise ValueError('unknown campaign split')
 required=('robot_urdf','robot_curobo_yaml','robot_meshes','curobo_implementation')
 if any(not any(k==prefix or k.startswith(prefix+'/') for k in robot_refs) for prefix in required):raise ValueError('robot/kinematics/collision/cuRobo evidence incomplete')
 refs={}
 for name,ref in sorted(robot_refs.items()):
  if not isinstance(ref,dict) or not ref.get('sha256') or file_hash(ref['path'])!=ref['sha256']:raise ValueError('robot or solver evidence hash mismatch: '+name)
  refs[name]=ref['sha256']
 modules={};trees=[]
 for name in MODULES:
  h,tree=semantic(code/name);modules[name]=h;trees.append(tree)
 objects=[]
 for o in t['objects']:
  if o.get('ready') is not True or len(o.get('asset_sha256',''))!=64:raise ValueError('asset identity/readiness unavailable')
  objects.append({k:o[k] for k in ('instance','category','model','asset_sha256','default_scale')})
 setup=t.get('scene_setup',{});source=Path(setup['source_template'])/'template.json'
 source_hash=file_hash(source)
 if source_hash!=setup['source_template_sha256']:raise ValueError('scene template identity differs')
 keys=settings_keys(trees)
 identity={'campaign_id':CAMPAIGN,'cache_kind':'exact_ik_v3','cache_schema':1,'modules':modules,
  'robot_solver_assets':refs,'settings':{k:env.get(k) for k in keys},
  'objects':sorted(objects,key=lambda o:o['instance']),
  'scene':{'model':t['scene'],'room':t['room'],'template_sha256':source_hash},
  'branch_limit':int(env.get('ISPATIALGYM_FAST_IK_CACHE_MAX_CHAINS',6)),
  'branch_min_rad':float(env.get('ISPATIALGYM_FAST_IK_BRANCH_MIN_RAD',.2)),
  'candidate_checks':'unchanged FK + current-world collision + support policy + normal downstream validation'}
 if not 1<=identity['branch_limit']<=6 or not identity['branch_min_rad']>=.2:raise ValueError('unsupported branch policy')
 return {'schema':SCHEMA,'campaign_id':CAMPAIGN,'compatibility_sha256':digest(identity),'compatible_identity':identity,
  'provenance':{'code':str(code),'config':str(config),'config_sha256':file_hash(config),'robot_refs':robot_refs,
    'scene_source':str(source),'runtime_commit':(code/'.snapshot_commit').read_text().strip() if (code/'.snapshot_commit').exists() else None,
    'environment':{k:env[k] for k in keys if k in env}}}

def validate_manifest(value):
 if value.get('schema')!=SCHEMA or value.get('campaign_id')!=CAMPAIGN:raise ValueError('foreign compatibility manifest')
 if value.get('compatibility_sha256')!=digest(value.get('compatible_identity')):raise ValueError('compatibility identity checksum differs')
 if value['compatible_identity'].get('campaign_id')!=CAMPAIGN or value['compatible_identity'].get('cache_kind')!='exact_ik_v3':raise ValueError('foreign cache scope')
 return value

def verify_running(value,code,config,env):
 validate_manifest(value)
 rebuilt=build_manifest(code,config,env,value['provenance']['robot_refs'])
 if rebuilt['compatibility_sha256']!=value['compatibility_sha256']:raise ValueError('running inputs are incompatible with cache manifest')
 return value
