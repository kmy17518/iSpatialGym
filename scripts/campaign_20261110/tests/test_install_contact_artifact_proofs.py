import json
import pytest
from scripts.campaign_20261110 import install_contact_artifact_proofs as m


def fixture(root):
    def ref(path, value):
        path.write_text(json.dumps(value));return {'path':str(path),'sha256':m.sha(path)}
    key='exact';prepared=ref(root/'prepared.json',{'physical_dedup_key':key});terminal=ref(root/'terminal.json',{'returncode':-11})
    payload=ref(root/'map.json',{'exact':'map'});runtime=ref(root/'runtime.json',{'frozen':True})
    manifest=ref(root/'tickets.json',{'jobs':[{'physical_dedup_key':key,'prepared':prepared,'input_refs':[payload],'runtime_ref':runtime}]})
    row={'physical_dedup_key':key,'prepared':prepared,'terminal':terminal,'status':'verified_positive','native_reader_passed':True,
         'process_exit_success':False,'original_returncode':-11,'verified_files':[payload],'completion_evidence':[terminal], 'verified_at':200}
    audit=ref(root/'audit.json',{'schema':'isg-completed-contact-artifact-audit-v1','manifest':manifest,'rows':[row]})
    out=root/'reader';out.mkdir();(out/'state.json').write_text(json.dumps({'rows':{key:{'terminal':terminal,'status':'invalid_or_failed'}}}))
    return audit,out,root/'map.json'


def test_adopts_immutable_proof_without_changing_live_state(tmp_path):
    audit,out,_=fixture(tmp_path);before=(out/'state.json').read_bytes()
    assert m.install(audit,out)['proof_count']==1 and not(out/'verifications').exists()
    assert m.install(audit,out,True)['proof_count']==1
    assert m.install(audit,out,True)['proof_count']==1
    assert (out/'state.json').read_bytes()==before
    proof=json.loads(next((out/'verifications').rglob('*.json')).read_text())
    assert proof['verification_at']==200 and proof['process_exit_success'] is False


def test_changed_payload_refuses_adoption(tmp_path):
    audit,out,payload=fixture(tmp_path);payload.write_text('changed')
    with pytest.raises(ValueError,match='Input changed'):m.install(audit,out,True)
    assert not(out/'verifications').exists()
