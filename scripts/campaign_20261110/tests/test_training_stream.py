import json
from pathlib import Path
import pytest
from scripts.campaign_20261110 import contracts as c,training_stream as stream


def test_native_gate_requires_exact_fresh_single_output(tmp_path):
    out=tmp_path/'canary/native';tro=out/'alias/template/instances/test__50000-tro_state.json'
    c.write_json_atomic(tro,{'ispatialgym':{'instance_id':'test__50000'}})
    c.write_json_atomic(tro.parent.parent/'template_meta.json',{'n_written':1})
    row={'output':str(out),'candidate_id':'c','config_sha256':'a'*64,'native_runtime':'runtime','start':50000}
    assert stream.native_evidence(row,tmp_path) is None
    gate={'campaign_id':'campaign_20261110','candidate_id':'c','config_sha256':'a'*64,'native_runtime':'runtime',
          'returncode':0,'tro_count':1,'tros':[{'path':str(tro),'sha256':c.file_digest(tro)}],
          'gate':{'allowed':True,'category':'validation_canary'}}
    c.write_json_atomic(out/'sampling_gate.json',gate)
    proof=stream.native_evidence(row,tmp_path)
    assert proof['metadata']==str(tro.parent.parent/'template_meta.json')
    gate['returncode']=124;c.write_json_atomic(out/'sampling_gate.json',gate)
    with pytest.raises(c.ContractError,match='successful sample'):stream.native_evidence(row,tmp_path)
    gate['returncode']=0;c.write_json_atomic(out/'sampling_gate.json',gate);tro.write_text('{}')
    with pytest.raises(c.ContractError,match='TRO changed'):stream.native_evidence(row,tmp_path)


def test_one_ready_configuration_advances_while_another_is_pending_or_failed(tmp_path,monkeypatch):
    requests=tmp_path/'requests.json';rows=[{'candidate_id':i,'config_path':str(tmp_path/i/'campaign_config.json')} for i in ('ready','pending','failed')]
    c.write_json_atomic(requests,{'campaign_id':'campaign_20261110','records':rows})
    calls=[]
    def prepare(row,root):
        calls.append(row['candidate_id'])
        if row['candidate_id']=='pending':return None
        if row['candidate_id']=='failed':raise ValueError('native validation failed')
        return {'status':'ready_for_guarded_gpu_submission','start':50000,'config_sha256':'a'*64}
    monkeypatch.setattr(stream,'prepare_annotated',prepare)
    first=stream.observe(tmp_path,requests)
    assert first['records']['ready']['status']=='ready_for_guarded_gpu_submission'
    assert first['records']['pending']['status']=='native_pending'
    assert first['records']['failed']['status']=='blocked'
    stream.observe(tmp_path,requests)
    assert calls.count('ready')==1
    assert not (tmp_path/'dataset').exists()
