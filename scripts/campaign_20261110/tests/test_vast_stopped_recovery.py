import hashlib
import json
import pytest
from scripts.campaign_20261110.infra_vast.worker.recovery import policy, verified_job, SCHEMA


def test_policy_requires_exact_campaign_and_separates_recovery_from_new_worker(tmp_path):
    p=tmp_path/'policy.json';p.write_text(json.dumps({'schema':SCHEMA,'campaign_id':'campaign_20261110','recover_instances':{'1':{'purpose':'archive'}}}))
    assert policy(p,'1')=={'purpose':'archive'}
    assert policy(p,'2') is None
    p.write_text(json.dumps({'schema':SCHEMA,'campaign_id':'old','recover_instances':{}}))
    with pytest.raises(ValueError):policy(p,'1')


def test_recovery_retains_original_bytes_and_rejects_eval_or_changed_config(tmp_path):
    job=tmp_path/'out/job';job.mkdir(parents=True);config=tmp_path/'config.json'
    config.write_text(json.dumps({'template':{'split':'train'}}));raw=config.read_bytes()
    (job/'job.env').write_text('ISPATIALGYM_CAMPAIGN_CONFIG='+str(config)+'\n')
    row={'job_id':'900001','job_dir':str(job),'config_sha256':hashlib.sha256(raw).hexdigest()}
    assert verified_job(row,tmp_path)==job
    assert config.read_bytes()==raw
    config.write_text(json.dumps({'template':{'split':'eval_standard'}}))
    with pytest.raises(ValueError,match='differs'):verified_job(row,tmp_path)
    row['config_sha256']=hashlib.sha256(config.read_bytes()).hexdigest()
    with pytest.raises(ValueError,match='standalone'):verified_job(row,tmp_path)


def test_recovery_cannot_archive_an_external_directory(tmp_path):
    with pytest.raises(ValueError,match='outside'):
        verified_job({'job_id':'900001','job_dir':'/etc'},tmp_path)


def test_archive_preserves_native_and_logs_without_decrypted_assets(tmp_path,monkeypatch):
    import subprocess
    from scripts.campaign_20261110.infra_vast.worker import recovery as r
    job=tmp_path/'out/job';(job/'inst').mkdir(parents=True)
    config=tmp_path/'config.json';config.write_text(json.dumps({'template':{'split':'train'}}))
    (job/'job.env').write_text('ISPATIALGYM_CAMPAIGN_CONFIG='+str(config)+'\n')
    tro=job/'inst/example-tro_state.json';tro.write_text('{"original":true}')
    (job/'sample.log').write_text('preserved failure evidence')
    (job/'private.usd').write_text('must not be transported')
    before={str(p):p.read_bytes() for p in job.rglob('*') if p.is_file()}
    monkeypatch.setenv('ISG_S3','s3://private/jobs/og-isg261110')
    uploads=[];monkeypatch.setattr(r,'s5',lambda *args:uploads.append(args))
    row={'job_id':'900001','job_dir':str(job),'config_sha256':r.digest(config)}
    r.recover({'purpose':'preserve_stopped_training_artifacts','jobs':[row]},'1',root=tmp_path,work_root=tmp_path/'recovery')
    archive=tmp_path/'recovery/1/900001.tar.zst'
    output=subprocess.run(['tar','--zstd','-tf',str(archive)],capture_output=True,text=True,check=True).stdout
    assert 'example-tro_state.json' in output and 'sample.log' in output
    assert 'private.usd' not in output
    assert before=={str(p):p.read_bytes() for p in job.rglob('*') if p.is_file()}
    report=json.loads((tmp_path/'recovery/1/report.json').read_text())
    assert report['collection_started'] is False and report['jobs'][0]['archive_sha256']==r.digest(archive)
    assert len(uploads)==2


def test_shell_boot_routing_needs_no_python_and_fails_closed(tmp_path):
    import subprocess
    from pathlib import Path
    route=Path(__file__).resolve().parents[1]/'infra_vast/worker/boot_mode.sh'
    p=tmp_path/'control.txt';p.write_text(SCHEMA+' campaign_20261110\n123 recover\n')
    def run(iid):return subprocess.run(['/bin/bash',str(route),str(p),iid],capture_output=True,text=True)
    assert run('123').stdout.strip()=='recover'
    assert run('124').stdout.strip()=='normal'
    p.write_text('wrong campaign_20261110\n123 recover\n');assert run('123').returncode!=0
    p.write_text(SCHEMA+' campaign_20261110\n123 unknown\n');assert run('123').returncode!=0
