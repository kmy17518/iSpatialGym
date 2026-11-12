from types import SimpleNamespace

import pytest

from scripts.campaign_20261110.contracts import ContractError
from scripts.campaign_20261110.training_physics_rebuild_probe import reconstruct_stopped


def fixture():
    calls = []
    class API:
        attached = 7
        def get_attached_stage(self): return self.attached
        def detach_stage(self): calls.append('detach'); self.attached = 0
        def attach_stage(self, stage): calls.append(('attach', stage)); self.attached = stage; return True
    return SimpleNamespace(stage=object(), scenes=[object()], is_stopped=lambda: True,
                           currently_stepping=False), API(), calls


def test_reconstruction_retains_stage_and_world_and_reports_separate_counter():
    sim, api, calls = fixture(); stage = sim.stage; scene = sim.scenes[0]
    result = reconstruct_stopped(sim, api, 7)
    assert calls == ['detach', ('attach', 7)] and sim.stage is stage and sim.scenes[0] is scene
    assert result['explicit_sdk_reconstructions'] == 1 and result['world_loads_added'] == 0
    assert result['explicit_physics_steps_requested'] == 0


def test_rejects_running_physics_before_sdk_mutation():
    sim, api, calls = fixture(); sim.is_stopped = lambda: False
    with pytest.raises(ContractError, match='stopped physics'): reconstruct_stopped(sim, api, 7)
    assert calls == []


def test_failed_detach_cannot_be_called_a_reconstruction():
    sim, api, calls = fixture(); api.detach_stage = lambda: None
    with pytest.raises(ContractError, match='did not detach'): reconstruct_stopped(sim, api, 7)
    assert calls == []


def test_failed_attach_is_preserved():
    sim, api, calls = fixture(); api.attach_stage = lambda stage: False
    with pytest.raises(ContractError, match='rejected'): reconstruct_stopped(sim, api, 7)


def test_replacing_loaded_world_is_rejected():
    sim, api, calls = fixture(); attach = api.attach_stage
    def changed(stage): sim.scenes.append(object()); return attach(stage)
    api.attach_stage = changed
    with pytest.raises(ContractError, match='replaced'): reconstruct_stopped(sim, api, 7)


def test_installer_does_not_require_kit_extension_before_environment(tmp_path, monkeypatch):
    import builtins
    import json
    from ispatialgym.oracle import campaign_restore
    from scripts.campaign_20261110.training_physics_rebuild_probe import install
    native_import = builtins.__import__
    def before_environment(name, *args, **kwargs):
        if name.startswith('omni.physx'):
            raise ModuleNotFoundError('Kit extension is not loaded before Environment')
        return native_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', before_environment)
    original = campaign_restore.restart_reused_world
    monkeypatch.setattr(campaign_restore, 'restart_reused_world', original)
    proof = tmp_path/'proof.json'; proof.write_text(json.dumps({}))
    install(None, {'explicit_physics_sdk_rebuild_v1': True,
                   'runtime_compatibility': {'path': str(proof)}}, tmp_path)
    assert campaign_restore.restart_reused_world is not original
