import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.campaign_20261110.contracts import ContractError, file_digest
from scripts.campaign_20261110 import training_pool_probe as probe
from scripts.campaign_20261110.training_pool_world import (
    scene_contract, isolated_member_scene, verify_pool_isolation)
from scripts.campaign_20261110.tests.test_training_pool_world import scenes


def test_load_only_does_not_claim_unconstructed_planner_isolation():
    documents = scenes()
    contract = scene_contract(documents, ['campaign_pool_A', 'campaign_pool_B', 'campaign_pool_A'], {'sha256': 'pool'})
    _, expected = isolated_member_scene(contract, documents[0], 'campaign_pool_A')
    observed = {name: {'properties': row['properties'], 'position': row['state']['root_link']['pos'],
                      'linear_velocity': [0, 0, 0], 'angular_velocity': [0, 0, 0]}
                for name, row in expected.items()}
    result = verify_pool_isolation(expected, observed, task_bindings={'target': 'campaign_pool_A'}, planner_names=None)
    assert result['properties_verified'] and result['planner_exclusion_verified'] is False
    observed['campaign_pool_B']['linear_velocity'] = [0, 0, 0.1]
    with pytest.raises(ContractError, match='physically or logically active'):
        verify_pool_isolation(expected, observed, task_bindings={'target': 'campaign_pool_A'}, planner_names=None)


def test_runtime_proof_rejects_other_physical_source_changes(tmp_path, monkeypatch):
    base = tmp_path/'code'/'native'; candidate = tmp_path/'code'/'candidate'
    ref = lambda p: {'path': str(p), 'sha256': file_digest(p)}
    for tree in (base, candidate):
        (tree/'ispatialgym').mkdir(parents=True)
        (tree/'scripts').mkdir()
        (tree/'.snapshot_commit').write_text(tree.name)
        (tree/'ispatialgym'/'physical.py').write_text('unchanged = True\n')
    module = 'scripts/campaign_20261110/training_pool_probe.py'
    (candidate/module).parent.mkdir(parents=True)
    (candidate/module).write_text('declared = True\n')
    source = {'native_runtime': ref(base/'.snapshot_commit')}
    monkeypatch.setattr(probe, 'verify_inputs', lambda *a: (source, {}))
    proof = {'schema': probe.SCHEMA, 'physical_delta': probe.DELTA,
             'permitted_execution_kinds': ['native_load_only'], 'native_runtime': source['native_runtime'],
             'candidate_runtime': ref(candidate/'.snapshot_commit'), 'changed_payloads': {module: file_digest(candidate/module)}}
    path = tmp_path/'proof.json'; path.write_text(json.dumps(proof))
    request = {'runtime_compatibility': ref(path)}
    records = [{'validation_key': {'code_sha256': source['native_runtime']['sha256']}}]
    assert probe.verify_runtime(tmp_path, request, records, candidate) == ref(path)
    (candidate/'ispatialgym'/'physical.py').write_text('unchanged = False\n')
    with pytest.raises(ContractError, match='unreviewed native payloads'):
        probe.verify_runtime(tmp_path, request, records, candidate)


@pytest.mark.parametrize('changes', [{'load_only': False}, {'production_released': True},
    {'benchmark_credit': 1}, {'diagnostic_control': 'A_FAILURE_A'}, {'physical_candidate': 'other'}])
def test_rejects_use_outside_bounded_load_only_scope(tmp_path, changes):
    request = {'selected_pool_world': True, 'load_only': True, 'production_released': False,
               'benchmark_credit': 0, 'diagnostic_control': 'ABA', **changes}
    with pytest.raises(ContractError, match='bounded diagnostic scope'):
        probe.verify_inputs(tmp_path, request, [])


def test_spawn_accepts_exact_model_or_generic_robot_record():
    pose = {'position': [1, 2, 3], 'orientation': [0, 0, 0, 1]}
    robot = SimpleNamespace(model='R1Pro')
    for key in ('robot', 'R1Pro'):
        assert probe._incoming_spawn(SimpleNamespace(tro=SimpleNamespace(robot_poses={key: [pose]})), robot) == pose
    with pytest.raises(ContractError, match='ambiguous'):
        probe._incoming_spawn(SimpleNamespace(tro=SimpleNamespace(robot_poses={'robot': [pose, pose]})), robot)
