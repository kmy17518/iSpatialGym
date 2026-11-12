import copy
import json
import pytest
from scripts.campaign_20261110 import training_release as release
from scripts.campaign_20261110.contracts import ContractError, file_digest, write_json_atomic


def fixture(root, kind='physics_action_open_loop'):
    def save(name, value):
        path=root/name;write_json_atomic(path,value)
        return {'path':str(path),'sha256':file_digest(path)}
    request=save('request.json',{'schema':'isg-training-batch-probe-request-v1'})
    observation={'scheduler_job_id':'42'}
    restoration={'native_final_pre_action_observation':{'observation':observation},'native_observation':observation}
    report={'loader':{'campaign_restoration':restoration},'campaign_restoration':restoration}
    report_ref=save('report.json',report)
    probe=save('output/probe.json',{'request':request,'kind':kind,'entries':[{'report':report_ref}]})
    submission={'request':request,'jobs':[{'job_id':'42','kind':kind,'returncode':0,'output':str(root/'output')}]}
    submitted=save('submission.json',submission)
    raw=root/'scheduler.txt';raw.write_text('42|COMPLETED|0:0|2026-11-12T01:00:00|2026-11-12T01:01:00|60|cpu=8,gres/gpu=1\n')
    execution={'schema':'isg-batch-execution-observation-v1','request':request,'submission':submitted,
        'jobs':[{'job_id':'42','kind':kind,'state':'COMPLETED','exit_code':'0:0','start':'2026-11-12T01:00:00',
                 'end':'2026-11-12T01:01:00','elapsed_seconds':60,'allocated_resources':'cpu=8,gres/gpu=1'}],
        'scheduler_evidence':{'path':str(raw),'sha256':file_digest(raw),'command':[
            'sacct','-X','-P','-n','-j','42','--format=JobIDRaw,State,ExitCode,Start,End,ElapsedRaw,AllocTRES']}}
    return save,probe,execution,submission,report


@pytest.mark.parametrize('kind',['physics_action_open_loop','kinematic_observation_rendering'])
def test_complete_native_probe_requires_exact_submitted_clean_process(tmp_path,kind):
    save,probe,execution,_,_=fixture(tmp_path,kind)
    assert release.verify_probe_execution(tmp_path,probe,save('execution.json',execution))['job_id']=='42'


@pytest.mark.parametrize('state,code',[('FAILED','139:0'),('COMPLETED','139:0'),('RUNNING','0:0'),('CANCELLED','0:0')])
def test_probe_before_teardown_does_not_clear_process_failure(tmp_path,state,code):
    save,probe,execution,_,_=fixture(tmp_path)
    execution['jobs'][0].update(state=state,exit_code=code)
    with pytest.raises(ContractError,match='did not complete cleanly'):
        release.verify_probe_execution(tmp_path,probe,save('execution.json',execution))


@pytest.mark.parametrize('change',['raw_failure','raw_duplicate','missing_raw','wrong_request','wrong_output','other_job','summary_mismatch'])
def test_process_proof_cannot_borrow_a_different_or_relabelled_execution(tmp_path,change):
    save,probe,execution,submission,report=fixture(tmp_path)
    raw=tmp_path/'scheduler.txt'
    if change=='raw_failure':raw.write_text(raw.read_text().replace('COMPLETED|0:0','FAILED|139:0'))
    if change=='raw_duplicate':raw.write_text(raw.read_text()*2)
    if change=='missing_raw':raw.write_text(raw.read_text().replace('42|','43|'))
    if change=='wrong_request':execution['request']=save('other_request.json',{})
    if change=='wrong_output':
        submission['jobs'][0]['output']=str(tmp_path/'different')
        execution['submission']=save('submission.json',submission)
    if change=='other_job':execution['jobs'][0]['job_id']='43'
    if change=='summary_mismatch':execution['jobs'][0]['elapsed_seconds']=61
    execution['scheduler_evidence']['sha256']=file_digest(raw)
    with pytest.raises(ContractError):release.verify_probe_execution(tmp_path,probe,save('execution.json',execution))


def test_original_failed_artifact_stays_immutable_during_later_clean_audit(tmp_path):
    save,probe,execution,_,_=fixture(tmp_path)
    failed=copy.deepcopy(execution);failed['jobs'][0].update(state='FAILED',exit_code='139:0')
    old=save('old_failure.json',failed);before=(tmp_path/'old_failure.json').read_bytes()
    release.verify_probe_execution(tmp_path,probe,save('execution.json',execution))
    assert (tmp_path/'old_failure.json').read_bytes()==before and file_digest(old['path'])==old['sha256']
