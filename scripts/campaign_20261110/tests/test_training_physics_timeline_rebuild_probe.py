from types import SimpleNamespace
import pytest
from scripts.campaign_20261110.contracts import ContractError
from scripts.campaign_20261110.training_physics_timeline_rebuild_probe import reconstruct_stopped

def fixture():
    calls = []
    class API:
        def release_physics_objects(self): calls.append('release')
        def force_load_physics_from_usd(self): calls.append('load')
        def is_running(self): return False
    sim = SimpleNamespace(stage=object(), scenes=[object()], is_stopped=lambda: True, currently_stepping=False)
    return sim, API(), SimpleNamespace(get_attached_stage=lambda: 0), calls

def test_rebuild_keeps_native_timeline_route_and_counts_physics_separately():
    sim, api, manual, calls = fixture(); stage=sim.stage; scene=sim.scenes[0]
    out = reconstruct_stopped(sim, api, manual, 7)
    assert calls == ['release', 'load'] and sim.stage is stage and sim.scenes[0] is scene
    assert out['explicit_sdk_reconstructions'] == 1 and out['world_loads_added'] == 0
    assert out['manual_attached_stage_before'] == out['manual_attached_stage_after'] == 0
    assert out['explicit_physics_steps_requested'] == 0

def test_rejects_manual_simulation_attachment_before_mutation():
    sim, api, manual, calls = fixture(); manual.get_attached_stage=lambda: 7
    with pytest.raises(ContractError, match='disable'): reconstruct_stopped(sim, api, manual, 7)
    assert not calls

def test_rejects_running_physics_before_mutation():
    sim, api, manual, calls = fixture(); sim.is_stopped=lambda: False
    with pytest.raises(ContractError, match='stopped'): reconstruct_stopped(sim, api, manual, 7)
    assert not calls

def test_rejects_changed_world_after_native_load():
    sim, api, manual, calls = fixture()
    def changed(): sim.scenes.append(object())
    api.force_load_physics_from_usd=changed
    with pytest.raises(ContractError, match='changed loaded world'): reconstruct_stopped(sim, api, manual, 7)

def test_api_failure_is_not_silently_counted_successful():
    sim, api, manual, calls = fixture()
    def failed(): raise RuntimeError('native load failed')
    api.force_load_physics_from_usd=failed
    with pytest.raises(RuntimeError, match='native load failed'): reconstruct_stopped(sim, api, manual, 7)
    assert calls == ['release']

def test_installer_does_not_import_kit_before_environment(tmp_path, monkeypatch):
    import builtins,json
    from ispatialgym.oracle import campaign_restore
    from scripts.campaign_20261110.training_physics_timeline_rebuild_probe import install
    native_import=builtins.__import__
    def before_environment(name,*args,**kwargs):
        if name.startswith('omni.physx'): raise ModuleNotFoundError('Kit not loaded')
        return native_import(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',before_environment)
    original=campaign_restore.restart_reused_world;monkeypatch.setattr(campaign_restore,'restart_reused_world',original)
    proof=tmp_path/'proof.json';proof.write_text(json.dumps({}))
    install(None,{'explicit_physics_timeline_rebuild_v1':True,'runtime_compatibility':{'path':str(proof)}},tmp_path)
    assert campaign_restore.restart_reused_world is not original
