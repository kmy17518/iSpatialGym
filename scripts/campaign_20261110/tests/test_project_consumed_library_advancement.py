import json
import pytest
from scripts.campaign_20261110.project_consumed_library_advancement import project
from scripts.campaign_20261110.vast_prebuild import sha, write


def test_submission_projection_preserves_actual_native_work_times(tmp_path):
    proof=tmp_path/'proof.json'; write(proof, {'physical_dedup_key':'lib','status':'verified_positive','verified_at':20})
    ref=lambda p:{'path':str(p),'sha256':sha(p)}
    side=tmp_path/'infra/coordinator_dispatch/preparation_verification/lib.json'
    write(side, {'physical_dedup_key':'lib','status':'verified_positive','verified_at':20,'terminal_at':10,
                 'verification_started_at':15,'evidence':[ref(proof)]})
    receipt=tmp_path/'infra/coordination/preparation_dispatch/receipts/map.json'
    bindings=tmp_path/'bindings.json'; write(bindings, {'schema':'isg-library-downstream-bindings-v1','bindings':[
        {'library_physical_key':'lib','verification':ref(proof),'downstream':[{'physical_dedup_key':'map','submission_path':str(receipt)}]}]})
    project(tmp_path,ref(bindings)); pending=json.loads(side.read_text()); assert pending['downstream_submitted_at'] is None
    write(receipt, {'physical_dedup_key':'map','job_id':'42','submitted_at':30})
    project(tmp_path,ref(bindings)); done=json.loads(side.read_text())
    assert (done['terminal_at'],done['verification_started_at'],done['verified_at']) == (10,15,20)
    assert done['downstream_submitted_at']==30 and done['blocker'] is None and done['next_check_at'] is None
    write(receipt, {'physical_dedup_key':'unrelated','job_id':'42','submitted_at':30})
    project(tmp_path,ref(bindings)); assert json.loads(side.read_text())['downstream_submitted_at'] is None


def test_projection_rejects_changed_native_proof(tmp_path):
    proof=tmp_path/'proof.json'; write(proof, {'physical_dedup_key':'lib','status':'verified_positive','verified_at':20})
    ref=lambda p:{'path':str(p),'sha256':sha(p)}
    bindings=tmp_path/'bindings.json'; write(bindings, {'schema':'isg-library-downstream-bindings-v1','bindings':[
        {'library_physical_key':'lib','verification':ref(proof),'downstream':[]}]})
    proof.write_text('{}')
    with pytest.raises(AssertionError): project(tmp_path,ref(bindings))
