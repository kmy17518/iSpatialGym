from types import SimpleNamespace
from unittest.mock import patch
import sys
import pytest
from ispatialgym.sampling import sample_isg as sampler


def loaded_pool(tmp_path):
    pool=sampler.WorldPool(str(tmp_path/'first'))
    pool.world=SimpleNamespace(spawned={},model_pool=SimpleNamespace(active={}))
    pool.kit=object();pool.n_loads=1;pool.model_pool_key='exact-manifest'
    return pool


def test_borrowed_main_uses_same_world_and_preserves_default_lifecycle(tmp_path,monkeypatch):
    pool=loaded_pool(tmp_path);world=pool.world;calls=[]
    monkeypatch.setitem(sys.modules,'omnigibson',SimpleNamespace(shutdown=lambda:calls.append('shutdown')))
    monkeypatch.setattr(sampler,'configure_logging',lambda _:None)
    def run(args,variant,actual,report,*_):
        assert actual is pool and actual.world is world and actual.require_compatible_reuse
        actual.last_action='reused';report['problems'][variant]={'template':{'n_written':1}}
        calls.append(variant)
    monkeypatch.setattr(sampler,'run_variant',run)
    for i,model in enumerate(('A','B','A')):
        code,report,returned=sampler.main(['--variant',model,'--customs',str(tmp_path/'none'),
            '--out',str(tmp_path/str(i))],shutdown=False,existing_pool=pool)
        assert code==0 and returned is pool and pool.world is world and pool.n_loads==1
        assert report['world']['borrowed_pool'] and report['world']['model_pool_manifest_sha256']=='exact-manifest'
        assert pool.require_compatible_reuse is False
    assert calls==['A','B','A']


def test_borrowed_pool_requires_loaded_clean_template_and_does_not_shutdown(tmp_path):
    pool=loaded_pool(tmp_path)
    with pytest.raises(ValueError,match='shutdown=False'):sampler.main([],existing_pool=pool)
    pool.world.spawned={'active':object()}
    with pytest.raises(ValueError,match='cleanup'):sampler.main([],shutdown=False,existing_pool=pool)
    pool.world.spawned={};pool.world.model_pool.active={'active':'target'}
    with pytest.raises(ValueError,match='cleanup'):sampler.main([],shutdown=False,existing_pool=pool)
    pool.world.model_pool.active={};pool.kit=None
    with pytest.raises(ValueError,match='not loaded'):sampler.main([],shutdown=False,existing_pool=pool)


def test_default_main_still_owns_and_shuts_down_its_new_world(tmp_path,monkeypatch):
    shutdown=[]
    monkeypatch.setitem(sys.modules,'omnigibson',SimpleNamespace(shutdown=lambda:shutdown.append(True)))
    monkeypatch.setattr(sampler,'configure_logging',lambda _:None)
    def run(args,variant,pool,report,*_):
        assert not pool.require_compatible_reuse and pool.world is None
        report['problems'][variant]={'template':{'n_written':1}}
    monkeypatch.setattr(sampler,'run_variant',run)
    with pytest.raises(SystemExit) as exit_result:
        sampler.main(['--variant','A','--customs',str(tmp_path/'none'),'--out',str(tmp_path/'default')])
    assert exit_result.value.code==0 and shutdown==[True]


def test_incompatible_borrowed_world_is_preserved_without_clear_or_rebuild(tmp_path,monkeypatch):
    stable=tmp_path/'stable.json';stable.write_text('{}')
    pool=loaded_pool(tmp_path);world=pool.world;pool.key=('old',('room',),());pool.require_compatible_reuse=True
    clear=[]
    monkeypatch.setitem(sys.modules,'omnigibson',SimpleNamespace(clear=lambda:clear.append(True)))
    monkeypatch.setitem(sys.modules,'ispatialgym.sampling.robot_utils',SimpleNamespace(RobotKit=object))
    monkeypatch.setitem(sys.modules,'ispatialgym.sampling.sim_utils',SimpleNamespace(SimWorld=object,stable_scene_path=lambda _:str(stable)))
    monkeypatch.setitem(sys.modules,'ispatialgym.sampling.scene_crop',SimpleNamespace(request_key=lambda *_:('full',),covers=lambda *_:True))
    monkeypatch.setattr(sampler,'world_key',lambda *_:('new',('room',),()))
    monkeypatch.setattr(sampler,'world_cameras_required',lambda _:True)
    monkeypatch.setattr(sampler,'scene_crop_request',lambda *_:('full',{},None))
    monkeypatch.delenv('ISPATIALGYM_CAMPAIGN_MODEL_POOL_MANIFEST',raising=False)
    with pytest.raises(ValueError,match='incompatible'):pool.get(SimpleNamespace(subtask='task'),'new')
    assert pool.world is world and pool.n_loads==1 and not clear
