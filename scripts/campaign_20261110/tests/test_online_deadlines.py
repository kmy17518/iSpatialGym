import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
import pytest
from scripts.campaign_20261110.online_deadline_watch import process,expired,watch
from scripts.campaign_20261110.class_wallclock import resolve,SCHEMA
from scripts.campaign_20261110.contracts import file_digest


def test_stuck_owned_sampler_is_stopped_but_unrelated_process_is_not(tmp_path):
    child=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
    unrelated=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)'])
    try:
        settings=tmp_path/'settings.json';settings.write_text(json.dumps({'state_dir':str(tmp_path)}))
        (tmp_path/'state.json').write_text(json.dumps({'current':{'id':'g','deadline':time.time()-5},
            'sampling_operation':{**process(child.pid),'active':True,'member_deadline':time.time()-5}}))
        watch(settings,child.pid,poll_s=.01,term_grace_s=.05)
        assert child.wait(timeout=2)<0
        assert unrelated.poll()is None
        assert (tmp_path/'hard_deadline.json').exists()
    finally:
        for p in(child,unrelated):
            if p.poll()is None:p.terminate()
            p.wait(timeout=2)


def test_policy_is_pinned_and_missing_class_fails_closed(tmp_path):
    path=tmp_path/'policy.json';path.write_text(json.dumps({'schema':SCHEMA,'classes':{'a':{'eval_standard':{'group_limit_s':1500,'member_limit_s':700}}}}))
    ref={'path':str(path),'sha256':file_digest(path)}
    assert resolve(ref,'a','eval_standard')['group_limit_s']==1500
    with pytest.raises(ValueError,match='missing class'):resolve(ref,'b','eval_standard')
    path.write_text('{}')
    with pytest.raises(ValueError,match='changed'):resolve(ref,'a','eval_standard')
