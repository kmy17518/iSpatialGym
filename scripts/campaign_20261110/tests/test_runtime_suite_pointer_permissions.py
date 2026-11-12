import json
import os
import stat

import pytest

from scripts.campaign_20261110 import runtime_suite_retry as retry
from scripts.campaign_20261110.contracts import ContractError, canonical_json
from scripts.campaign_20261110.tests.test_runtime_suite_retry import proposed, approve


@pytest.mark.parametrize('mode', [0o600, 0o640, 0o644])
def test_registration_preserves_pointer_mode_under_restrictive_umask(tmp_path, mode):
    base, _, _, _, proposal = proposed(tmp_path)
    (base/'attempt_registries').mkdir(mode=0o750)
    prepared = retry.prepare(tmp_path, proposal)
    approval = approve(tmp_path, prepared)
    pointer = tmp_path/'runtime_validation/current.json'
    pointer.chmod(mode)
    before = json.loads(pointer.read_text())
    old_inode = pointer.stat().st_ino
    old_umask = os.umask(0o077)
    try:
        result = retry.register(tmp_path, prepared, approval)
    finally:
        os.umask(old_umask)
    expected = {**before, 'attempt_registry': result['registry']}
    assert pointer.read_text() == canonical_json(expected)+'\n'
    assert stat.S_IMODE(pointer.stat().st_mode) == mode
    assert pointer.stat().st_ino != old_inode
    assert not result['dispatch_performed']


def test_failed_atomic_publication_preserves_original_bytes_and_mode(tmp_path, monkeypatch):
    pointer = tmp_path/'current.json'
    pointer.write_bytes(b'{"original": true}\n')
    pointer.chmod(0o640)
    def fail(*args):
        raise OSError('publication failed')
    monkeypatch.setattr(retry.os, 'replace', fail)
    with pytest.raises(OSError, match='publication failed'):
        retry._replace_pointer(pointer, {'new': True})
    assert pointer.read_bytes() == b'{"original": true}\n'
    assert stat.S_IMODE(pointer.stat().st_mode) == 0o640
    assert sorted(p.name for p in tmp_path.iterdir()) == ['current.json']


def test_pointer_symlink_is_rejected_without_changing_target(tmp_path):
    target = tmp_path/'target.json'
    target.write_bytes(b'{}\n')
    pointer = tmp_path/'current.json'
    pointer.symlink_to(target)
    with pytest.raises(ContractError, match='regular file'):
        retry._replace_pointer(pointer, {'new': True})
    assert target.read_bytes() == b'{}\n'


def test_registration_shares_only_exact_control_dependencies(tmp_path):
    base, _, _, rowref, proposal = proposed(tmp_path)
    prepared = retry.prepare(tmp_path, proposal)
    approval = approve(tmp_path, prepared)
    value = json.loads(open(prepared['path']).read())
    required = [rowref, prepared, approval, value['diagnosis']]
    before = {}
    for ref in required:
        path = retry.Path(ref['path'])
        path.chmod(0o600)
        before[str(path)] = path.read_bytes()
    unrelated = base/'attempts/unregistered_private.json'
    unrelated.write_bytes(b'{"unrelated": true}\n')
    unrelated.chmod(0o600)
    result = retry.register(tmp_path, prepared, approval)
    for ref in required + [result['registry']]:
        path = retry.Path(ref['path'])
        assert stat.S_IMODE(path.stat().st_mode) & stat.S_IRGRP
        if str(path) in before:
            assert path.read_bytes() == before[str(path)]
    assert stat.S_IMODE(unrelated.stat().st_mode) == 0o600
    assert unrelated.read_bytes() == b'{"unrelated": true}\n'
    assert str(unrelated) not in [r['reference']['path'] for r in result['control_access']]


def test_private_control_directory_fails_without_broadening_it(tmp_path):
    base, _, _, rowref, proposal = proposed(tmp_path)
    prepared = retry.prepare(tmp_path, proposal)
    approval = approve(tmp_path, prepared)
    directory = retry.Path(rowref['path']).parent
    directory.chmod(0o700)
    pointer = tmp_path/'runtime_validation/current.json'
    before = pointer.read_bytes()
    with pytest.raises(ContractError, match='group traversable'):
        retry.register(tmp_path, prepared, approval)
    assert pointer.read_bytes() == before
    assert stat.S_IMODE(directory.stat().st_mode) == 0o700


def test_exact_external_group_goal_is_shared_but_other_evidence_is_private(tmp_path):
    tmp_path.chmod(0o750)
    from scripts.campaign_20261110.tests.test_runtime_suite import put
    base, _, _, rowref, proposal = proposed(tmp_path)
    row = json.loads(retry.Path(rowref['path']).read_text())
    goal = tmp_path/'groups'/row['group_id']/'goal_plan.json'
    row['goal_plan'] = put(goal, {'group_id': row['group_id']})
    rowref = put(retry.Path(rowref['path']), row)
    body = json.loads(retry.Path(proposal['path']).read_text())
    body['row'] = rowref
    proposal = put(retry.Path(proposal['path']), body)
    prepared = retry.prepare(tmp_path, proposal)
    approval = approve(tmp_path, prepared)
    private = goal.parent/'unrelated.json'
    private.write_bytes(b'{}\n'); private.chmod(0o600)
    goal.chmod(0o600); before = goal.read_bytes()
    result = retry.register(tmp_path, prepared, approval)
    assert goal.read_bytes() == before
    assert stat.S_IMODE(goal.stat().st_mode) == 0o640
    assert stat.S_IMODE(private.stat().st_mode) == 0o600
    assert row['goal_plan'] in [r['reference'] for r in result['control_access']]


def test_external_goal_private_parent_is_not_widened(tmp_path):
    from scripts.campaign_20261110.tests.test_runtime_suite import put
    base, _, _, rowref, proposal = proposed(tmp_path)
    row = json.loads(retry.Path(rowref['path']).read_text())
    goal = tmp_path/'groups'/row['group_id']/'goal_plan.json'
    row['goal_plan'] = put(goal, {'group_id': row['group_id']})
    rowref = put(retry.Path(rowref['path']), row)
    body = json.loads(retry.Path(proposal['path']).read_text()); body['row'] = rowref
    prepared = retry.prepare(tmp_path, put(retry.Path(proposal['path']), body))
    approval = approve(tmp_path, prepared)
    goal.parent.chmod(0o700)
    pointer = tmp_path/'runtime_validation/current.json'; before = pointer.read_bytes()
    with pytest.raises(ContractError, match='group traversable'):
        retry.register(tmp_path, prepared, approval)
    assert pointer.read_bytes() == before
    assert stat.S_IMODE(goal.parent.stat().st_mode) == 0o700
