import json
import os
import stat

import pytest

from scripts.campaign_20261110 import runtime_suite_retry as retry
from scripts.campaign_20261110.contracts import ContractError, canonical_json
from scripts.campaign_20261110.tests.test_runtime_suite_retry import proposed, approve


@pytest.mark.parametrize('mode', [0o600, 0o640, 0o644])
def test_registration_preserves_pointer_mode_under_restrictive_umask(tmp_path, mode):
    _, _, _, _, proposal = proposed(tmp_path)
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
