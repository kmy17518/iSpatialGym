import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from scripts.campaign_20261110 import infra_submit_validation as submit


def fixture(tmp_path, monkeypatch, *, explicit=True, native_alias='mental_rotation-soup_ladle'):
    root=tmp_path/'campaign';snapshot=root/'code/frozen';snapshot.mkdir(parents=True)
    commit=snapshot/'.snapshot_commit';commit.write_text('frozen source\n')
    customs=root/'private/customs.json';customs.parent.mkdir();customs.write_text(json.dumps({'mental_rotation-soup_ladle':{'generation':{}}}))
    source=root/'templates/native';source.mkdir(parents=True)
    (source/'template_meta.json').write_text(json.dumps({'variant':native_alias,'customs':str(customs)}))
    group=root/'groups/validation';group.mkdir(parents=True)
    config={'purpose':'validation','template':{'task':'mental_rotation','scene_setup':{'customs_alias':'mental_rotation-soup_ladle'}},
            'validation_key':{'code_sha256':hashlib.sha256(commit.read_bytes()).hexdigest()}}
    (group/'campaign_config.json').write_text(json.dumps(config));(group/'source.json').write_text(json.dumps({'template_dir':str(source)}))
    argv=['submit','--root',str(root),'--snapshot',str(snapshot),'--group',str(group),'--key','mental_rotation-soup_ladle',
          '--tag','validation','--v2-root',str(root/'v2'),'--index','9000000','--template-customs','--dry-run']
    if explicit:argv+=['--customs-file',str(customs)]
    monkeypatch.setattr(sys,'argv',argv)
    calls=[]
    monkeypatch.setattr(submit.subprocess,'run',lambda cmd,**kwargs: calls.append(cmd) or SimpleNamespace(returncode=0,stdout='dry run',stderr=''))
    return customs,calls


def test_private_customs_is_selected_and_visible_to_portable_input_staging(tmp_path,monkeypatch):
    customs,calls=fixture(tmp_path,monkeypatch)
    submit.main()
    assert len(calls)==1
    cmd=calls[0];env=[cmd[i+1] for i,arg in enumerate(cmd) if arg=='--extra-env']
    assert 'ISG_CUSTOMS='+str(customs) in env
    assert 'ISPATIALGYM_CAMPAIGN_CUSTOMS_SHA256='+hashlib.sha256(customs.read_bytes()).hexdigest() in env
    assert cmd[cmd.index('--customs')+1]=='template'
    assert '--dry-run' in cmd


def test_mismatched_native_alias_is_rejected_before_submission(tmp_path,monkeypatch):
    _,calls=fixture(tmp_path,monkeypatch,native_alias='mental_rotation-other')
    with pytest.raises(AssertionError,match='native customs alias differs'):
        submit.main()
    assert calls==[]


def test_ordinary_template_submission_does_not_invent_a_customs_override(tmp_path,monkeypatch):
    _,calls=fixture(tmp_path,monkeypatch,explicit=False)
    submit.main()
    assert all(not arg.startswith('ISG_CUSTOMS=') for arg in calls[0])
