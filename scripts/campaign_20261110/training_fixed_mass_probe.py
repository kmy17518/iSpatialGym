"""Bounded fixed-tray overridden-baseline experiment, never production clearance."""
import copy
import json
from pathlib import Path
from .contracts import file_digest, require, digest, write_json_atomic

PARENT_COMMIT = '2bbeebe84aaa4c1f86cb1da5ed7082d198d07314'
DELTA = {'name': 'fixed_tray_mass_inertia_baseline_v1',
         'boundary': 'world_initialized_before_first_episode_restore',
         'cold_physical_operations_changed': True, 'warm_physical_operations_changed': True,
         'additional_physics_steps': 0, 'task_input_files_changed': False,
         'thresholds_changed': False, 'stock_mass_restored_between_episodes': False}
ALLOWED = {'scripts/campaign_20261110/training_fixed_mass_probe.py',
           'scripts/campaign_20261110/tests/test_training_fixed_mass_probe.py',
           'scripts/campaign_20261110/training_batch_probe.py'}


def verify_fixed_mass_runtime(root, request, records, runtime, verify_parent):
    require(request.get('fixed_mass_baseline') is True and request.get('sequential_anchor_rebase') is True
            and request.get('sequential_augmentation') is True and not request.get('sequential_mass_restart')
            and not request.get('physical_candidate') and request.get('production_released') is False
            and request.get('benchmark_credit') == 0
            and request.get('diagnostic_control') in (None,'FRESH_A','FRESH_B','A_A','A_FAILURE_A','A_INTERRUPT_A'),
            'fixed mass baseline requires bounded prospective physics scope')
    reference=request['runtime_compatibility']; path=Path(reference['path']).resolve()
    require(path.is_relative_to(root) and file_digest(path)==reference['sha256'], 'fixed mass proof changed')
    proof=json.loads(path.read_text())
    require(proof.get('schema')=='isg-fixed-mass-baseline-runtime-compatibility-v1'
            and proof.get('physical_delta')==DELTA and proof.get('physical_code_changes')==1
            and proof.get('permitted_execution_kinds')==['physics_action_open_loop'],
            'fixed mass experiment must declare its physical delta')
    parent_ref=proof['parent_runtime'];parent=Path(parent_ref['path']).resolve().parent
    require(parent.is_relative_to(root/'code') and runtime.is_relative_to(root/'code')
            and parent_ref=={'path':str(parent/'.snapshot_commit'),'sha256':file_digest(parent/'.snapshot_commit')}
            and (parent/'.snapshot_commit').read_text().strip()==PARENT_COMMIT
            and proof['candidate_runtime']=={'path':str(runtime/'.snapshot_commit'),'sha256':file_digest(runtime/'.snapshot_commit')},
            'fixed mass runtime identity differs')
    inherited={**request,'runtime_compatibility':proof['parent_compatibility'],
               'probe_runtime_sha256':parent_ref['sha256']};inherited.pop('fixed_mass_baseline')
    verify_parent(root,inherited,records,parent)
    paths={str(p.relative_to(tree))for tree in(parent,runtime)for folder in('ispatialgym','scripts')
           for p in(tree/folder).rglob('*.py')if p.is_file()}
    changed={n for n in paths if not(parent/n).is_file()or not(runtime/n).is_file()or file_digest(parent/n)!=file_digest(runtime/n)}
    require(changed==set(proof['changed_payloads']) and changed<=ALLOWED,'unapproved fixed mass experiment payload')
    require(all((runtime/n).resolve().is_relative_to(runtime) and file_digest(runtime/n)==h
                for n,h in proof['changed_payloads'].items()),'fixed mass experiment bytes changed')
    scopes=[]
    for row in records:
        tro=json.loads(Path(row['artifacts']['tro']['path']).read_text())['ispatialgym']
        mass=[r for r in tro['constraints'] if r['type']=='mass']
        require(len(mass)==2 and {r['object']for r in mass}=={'tray.n.01_1','tray.n.01_2'}
                and all(r['mass_kg']==2. and not r.get('link')for r in mass),'fixed tray mass inputs differ')
        scales=tro['campaign']['perturbations']['scales']
        require(all(scales[r['object']]['mode']=='fixed' and scales[r['object']]['applied']==[1.,1.,1.]
                    for r in mass),'tray scale changes require a different baseline')
        scopes.append({'mass':mass,'bindings':{r['object']:tro['object_bindings'][r['object']]for r in mass}})
    require(scopes and all(s==scopes[0]for s in scopes),'fixed tray model/mass/scale changes across episodes')
    return reference


def principal_inertia(tensor):
    """PhysX inertia is at COM in prim frame; USD uses principal values/axes."""
    import numpy as np
    from scipy.spatial.transform import Rotation
    matrix=np.asarray(tensor,dtype=float).reshape(3,3,order='F')
    require(np.isfinite(matrix).all() and np.allclose(matrix,matrix.T,atol=1e-10,rtol=1e-6),
            'invalid native inertia tensor')
    values, axes=np.linalg.eigh(matrix)
    require((values>0).all(),'nonpositive native inertia')
    if np.linalg.det(axes)<0:axes[:,0]*=-1
    quaternion=Rotation.from_matrix(axes).as_quat()
    require(np.allclose(axes@np.diag(values)@axes.T,matrix,atol=1e-10,rtol=1e-6),
            'principal inertia reconstruction differs')
    return values.tolist(),quaternion.tolist()


def _read(link):
    from ispatialgym.oracle.campaign_restore import _mass_state
    state=_mass_state(link)
    for key,attr in [('diagonal_inertia','physics:diagonalInertia'),('principal_axes','physics:principalAxes')]:
        a=link.prim.GetAttribute(attr);v=a.Get() if a and a.HasAuthoredValueOpinion() else None
        if v is not None:v=([float(x)for x in v.GetImaginary()]+[float(v.GetReal())]) if key=='principal_axes' else list(map(float,v))
        state['usd_'+key]=v
    return state


def _write(link, state):
    import torch
    import omnigibson as og
    from pxr import UsdPhysics,Gf
    vals,quat=principal_inertia(state['inertia'])
    with og.sim.editing_usd():
        api=UsdPhysics.MassAPI.Apply(link.prim)
        api.CreateMassAttr().Set(float(state['mass_kg']))
        api.CreateDiagonalInertiaAttr().Set(Gf.Vec3f(*vals))
        api.CreatePrincipalAxesAttr().Set(Gf.Quatf(quat[3],Gf.Vec3f(*quat[:3])))
    view=link._rigid_prim_view;inertia=view.get_inertias();masses=view.get_masses()
    view.set_masses(torch.as_tensor([state['mass_kg']],dtype=masses.dtype,device=masses.device))
    view.set_inertias(torch.as_tensor(state['inertia'],dtype=inertia.dtype,device=inertia.device))


def matches(state, expected):
    import numpy as np
    from scipy.spatial.transform import Rotation
    if not abs(state['mass_kg']-expected['mass_kg'])<=1e-7*max(1.,expected['mass_kg']):return False
    if not np.allclose(state['inertia'],expected['inertia'],atol=1e-9,rtol=1e-6):return False
    if state['usd_mass'] is None or abs(state['usd_mass']-expected['mass_kg'])>1e-7:return False
    if state['usd_diagonal_inertia'] is None or state['usd_principal_axes'] is None:return False
    rotation=Rotation.from_quat(state['usd_principal_axes']).as_matrix()
    usd=rotation@np.diag(state['usd_diagonal_inertia'])@rotation.T
    return bool(np.allclose(usd,np.asarray(expected['inertia']).reshape(3,3,order='F'),atol=1e-9,rtol=1e-6))


def install_fixed_mass_baseline(runner, request, output):
    """Only this separately verified probe calls the installer; metadata stays exact."""
    from ispatialgym.oracle import campaign_restore as CR
    import omnigibson as og
    import torch
    require(request.get('fixed_mass_baseline') is True and request.get('production_released') is False,
            'fixed mass experiment not requested')
    build=runner.InstanceLoader.build_env; restart=CR.restart_reused_world
    mass=runner.InstanceLoader._apply_mass; restore=runner.InstanceLoader._restore_complete_snapshot
    targets={};events=[]; output=Path(output)
    def record(stage, **extra):
        states={key:_read(link)for key,(link,expected)in targets.items()}
        passed=all(matches(states[key],expected)for key,(link,expected)in targets.items())
        events.append({'stage':stage,'passed':passed,'native':states,**extra})
        write_json_atomic(output/'fixed_mass_audit.json',{'schema':'isg-fixed-tray-mass-inertia-audit-v1',
            'physical_delta':DELTA,'runtime_compatibility':request['runtime_compatibility'],
            'expected':{key:expected for key,(link,expected)in targets.items()},'events':events,
            'physics_steps_added':0,'production_released':False})
        require(passed,'fixed tray mass/inertia persistence failed at '+stage)
    def initialized(loader,*args,**kwargs):
        result=build(loader,*args,**kwargs)
        campaign=loader.tro.meta.raw.get('campaign')or{}
        require(CR.pristine_mass_enabled(campaign,loader._restoration_execution)
                and not targets,'fixed baseline requires one new authorized world')
        objects=[];stock={}
        for row in loader.tro.meta.extras.get('constraints')or[]:
            if row['type']!='mass':continue
            obj=loader.task.object_scope[row['object']]
            require(obj.category=='tray' and not row.get('link') and float(row['mass_kg'])==2.,'unexpected mass target')
            link=obj.root_link; before=_read(link);desired=copy.deepcopy(before)
            desired['mass_kg']=float(row['mass_kg'])
            desired['inertia']=(torch.as_tensor(before['inertia'],dtype=torch.float32)*(desired['mass_kg']/before['mass_kg'])).tolist()
            vals,quat=principal_inertia(desired['inertia'])
            desired.update(usd_mass_authored=True,usd_mass=desired['mass_kg'],
                           usd_diagonal_inertia=vals,usd_principal_axes=quat)
            stock[str(link.prim_path)]=before;_write(link,desired)
            targets[str(link.prim_path)]=(link,desired);objects.append(obj)
        require(len(targets)==2,'fixed baseline tray inventory differs')
        record('world_initialized_after_one_override',stock=stock)
        # Ordinary scene.save does not prove inertial persistence. Keep a required
        # companion baseline with explicit USD/native values and exact source IDs.
        loader._scene_state0=copy.deepcopy(loader.env.scene.dump_state(serialized=False))
        loader._pristine_mass0=CR.capture_pristine_mass(campaign,objects,execution=loader._restoration_execution)
        loader.report['fixed_mass_baseline']={'physical_delta':DELTA,'source_campaign_sha256':digest(campaign),
            'source_tro':{'path':str(loader.tro_path),'sha256':file_digest(loader.tro_path)},
            'intended':{key:expected for key,(link,expected)in targets.items()},
            'required_companion':str(output/'fixed_mass_audit.json'),'ordinary_scene_save_alone_is_sufficient':False}
        write_json_atomic(output/'fixed_mass_baseline_config.json',loader.report['fixed_mass_baseline'])
        return result
    def restarted(*args,**kwargs):
        record('before_existing_stop_play')
        result=restart(*args,**kwargs)
        record('after_existing_stop_play')
        return result
    def applied(loader,row):
        result=mass(loader,row)
        # Equal mass is insufficient: reject a stale tensor or USD inertia even
        # when apply_mass took its ordinary equal-mass early return.
        record('after_native_mass_constraint',object=row['object'])
        return result
    def restored(loader,*args,**kwargs):
        record('before_episode_restore',instance_id=loader.iid)
        result=restore(loader,*args,**kwargs)
        record('after_episode_restore',instance_id=loader.iid)
        return result
    runner.InstanceLoader.build_env=initialized
    CR.restart_reused_world=restarted
    runner.InstanceLoader._apply_mass=applied
    runner.InstanceLoader._restore_complete_snapshot=restored
