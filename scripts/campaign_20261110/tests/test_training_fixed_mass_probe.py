import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from scripts.campaign_20261110.training_fixed_mass_probe import principal_inertia,matches,DELTA
from scripts.campaign_20261110.contracts import ContractError


def state(matrix):
    values,quat=principal_inertia(matrix)
    return {'mass_kg':2.,'inertia':np.asarray(matrix).reshape(1,9,order='F').tolist(),
            'usd_mass':2.,'usd_diagonal_inertia':values,'usd_principal_axes':quat}


def test_non_axis_aligned_inertia_round_trip_and_native_equality():
    r=Rotation.from_euler('xyz',[.3,-.6,1.1]).as_matrix()
    matrix=r@np.diag([.001,.003,.004])@r.T
    expected=state(matrix);actual=state(matrix)
    assert matches(actual,expected)
    actual['usd_diagonal_inertia'][0]*=2
    assert not matches(actual,expected)


def test_equal_mass_never_hides_wrong_native_inertia_or_missing_usd():
    original=state(np.diag([.001,.002,.003]))
    stale=state(np.diag([.0001,.0002,.0003]))
    assert stale['mass_kg']==original['mass_kg'] and not matches(stale,original)
    missing=state(np.diag([.001,.002,.003]));missing['usd_diagonal_inertia']=None
    assert not matches(missing,original)
    missing=state(np.diag([.001,.002,.003]));missing['usd_mass']=.39
    assert not matches(missing,original)


@pytest.mark.parametrize('matrix',[np.zeros((3,3)),np.diag([1.,1.,-1.]),
                                    np.array([[1.,.1,0],[0,1,0],[0,0,1]]),np.full((3,3),np.nan)])
def test_invalid_native_tensor_fails_before_usd_write(matrix):
    with pytest.raises(ContractError):principal_inertia(matrix)


def test_declares_changed_cold_physics_without_threshold_or_source_rewrite():
    assert DELTA['cold_physical_operations_changed'] and DELTA['warm_physical_operations_changed']
    assert DELTA['additional_physics_steps']==0 and not DELTA['thresholds_changed']
    assert not DELTA['task_input_files_changed'] and not DELTA['stock_mass_restored_between_episodes']


def test_installer_captures_overridden_baseline_once_and_detects_restart_inertia_drift(tmp_path,monkeypatch):
    import copy,json,sys
    from types import SimpleNamespace as NS
    import scripts.campaign_20261110.training_fixed_mass_probe as F
    from ispatialgym.oracle import campaign_restore as CR
    memory={};events=[];objects={}
    for i in (1,2):
        link=NS(prim_path='/World/tray'+str(i));obj=NS(category='tray',root_link=link)
        objects['tray.n.01_'+str(i)]=obj
        s=state(np.diag([.001,.002,.003]));s.update(mass_kg=.4,usd_mass=.4)
        memory[link.prim_path]=s
    monkeypatch.setitem(sys.modules,'omnigibson',NS(sim=NS()))
    monkeypatch.setattr(F,'_read',lambda link:copy.deepcopy(memory[link.prim_path]))
    def write(link,s):events.append('override');memory[link.prim_path]=copy.deepcopy(s)
    monkeypatch.setattr(F,'_write',write)
    monkeypatch.setattr(CR,'pristine_mass_enabled',lambda *a:True)
    def capture(*a,**kw):events.append('capture');return copy.deepcopy(memory)
    monkeypatch.setattr(CR,'capture_pristine_mass',capture)
    def restart(*a,**kw):
        events.append('restart')
        if kw.get('while_stopped'): kw['while_stopped']()
        return {'observed_physics_step_calls':[],'physics_steps_requested':0}
    monkeypatch.setattr(CR,'restart_reused_world',restart)
    monkeypatch.setattr(F,'_author_inertia',lambda *a:events.append('author_stopped'))
    source=tmp_path/'source.json';source.write_text('{}\n');campaign={'campaign_id':'campaign_20261110'}
    class Loader:
        def __init__(self):
            self.tro=NS(meta=NS(raw={'campaign':campaign},extras={'constraints':[
                {'type':'mass','object':k,'mass_kg':2.}for k in objects]}))
            self.tro_path=source;self._restoration_execution=None;self.report={};self.iid='A'
            self.task=NS(object_scope=objects);self.env=NS(scene=NS(dump_state=lambda **kw:{'original_positions':'exact'}))
        def build_env(self):events.append('build');return self.env
        def _apply_mass(self,row):return {'equal_mass_return':True}
        def _restore_complete_snapshot(self):
            assert all(s['mass_kg']==2 for s in self._pristine_mass0.values())
            return 'restored'
    request={'fixed_mass_baseline':True,'production_released':False,'runtime_compatibility':{'sha256':'exact'}}
    F.install_fixed_mass_baseline(NS(InstanceLoader=Loader),request,tmp_path)
    loader=Loader();loader.build_env()
    assert events==['build','restart','author_stopped','author_stopped','override','override','capture']
    assert campaign=={'campaign_id':'campaign_20261110'} and source.read_text()=='{}\n'
    assert loader._scene_state0=={'original_positions':'exact'}
    assert loader._restore_complete_snapshot()=='restored'
    loader._apply_mass({'object':'tray.n.01_1'})
    CR.restart_reused_world({},None)
    audit=json.loads((tmp_path/'fixed_mass_audit.json').read_text())
    assert all(e['passed']for e in audit['events'])
    assert events.count('override')==2
    # Equal mass with wrong inertia cannot be silently accepted after a restart.
    memory['/World/tray1']['inertia'][0][0]*=.5
    with pytest.raises(ContractError,match='persistence failed'):
        CR.restart_reused_world({},None)
    audit=json.loads((tmp_path/'fixed_mass_audit.json').read_text())
    assert audit['events'][-1]['passed']is False and not audit['production_released']


def test_all_actors_authored_stopped_before_any_native_tensor_access(monkeypatch):
    import scripts.campaign_20261110.training_fixed_mass_probe as F
    events=[]
    targets={'A':('A',{}),'B':('B',{})}
    monkeypatch.setattr(F,'_author_inertia',lambda link,s:events.append(('author',link)))
    monkeypatch.setattr(F,'_write',lambda link,s:events.append(('native',link)))
    def restart(meta,sim,*,execution,while_stopped):
        events.append('stop');while_stopped();events.append('play')
        return {'observed_physics_step_calls':[],'physics_steps_requested':0}
    F.initialize_inertial_actors(targets,{},None,restart,None)
    assert events==['stop',('author','A'),('author','B'),'play',('native','A'),('native','B')]


def test_initial_reconstruction_rejects_unmeasured_or_stepping_lifecycle(monkeypatch):
    import scripts.campaign_20261110.training_fixed_mass_probe as F
    monkeypatch.setattr(F,'_write',lambda *a:pytest.fail('native tensor write before lifecycle proof'))
    for evidence in [None,{'observed_physics_step_calls':['step'],'physics_steps_requested':0}]:
        with pytest.raises(ContractError,match='no-step lifecycle proof'):
            F.initialize_inertial_actors({'A':('A',{})},{},None,lambda *a,**kw:evidence,None)
