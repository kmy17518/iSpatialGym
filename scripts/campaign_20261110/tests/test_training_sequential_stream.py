from pathlib import Path
import json
import pytest
from scripts.campaign_20261110 import training_sequential_stream as stream
from scripts.campaign_20261110.contracts import ContractError,write_json_atomic


def test_native_failure_does_not_authorize_collect_or_matrix(tmp_path,monkeypatch):
    batch=tmp_path/'batch';batch.mkdir();author=tmp_path/'author.py';author.write_text('raise RuntimeError("must not run")')
    request=batch/'requests.json';write_json_atomic(request,{'campaign_id':'campaign_20261110','records':[{'candidate_id':'a'},{'candidate_id':'b'}]})
    config={'batch':str(batch),'requests':stream.ref(request),'prepared':str(tmp_path/'prepared'),'authorizer':stream.ref(author),'authorizations':str(tmp_path/'auth'),'runtime':str(tmp_path/'runtime'),'campaign_started_at':0}
    def failed(*args):raise ContractError('native canary did not write exactly one successful sample')
    monkeypatch.setattr(stream,'native_evidence',failed)
    monkeypatch.setattr(stream,'prepare_matrix',lambda *a:pytest.fail('failed source cannot author matrix'))
    state=stream.run_cycle(tmp_path,config,{})
    assert {r['status'] for r in state['records'].values()}=={'engineering_blocked'}
    assert not(tmp_path/'auth').exists()


def test_matrix_does_not_replace_pinned_sources(tmp_path):
    batch=tmp_path/'batch';directory=batch/'sequential_verification_r1';directory.mkdir(parents=True)
    write_json_atomic(directory/'requests.json',{'runtime':'native','canaries':[{'path':'old','sha256':'a'}],'records':[]})
    with pytest.raises(ContractError,match='source identity changed'):
        stream.prepare_matrix(tmp_path,batch,[{'path':'new','sha256':'b'}],Path('native'),0)


def test_snapshot_preserves_previous_exact_bytes(tmp_path):
    p=tmp_path/'state.json';stream.publish_snapshot(p,{'phase':'native'})
    first=p.read_bytes();key=stream.file_digest(p)
    stream.publish_snapshot(p,{'phase':'replay'})
    assert (tmp_path/'state_history'/(key+'.json')).read_bytes()==first
    assert json.loads(p.read_text())['phase']=='replay'
