import json

import pytest

from scripts.campaign_20261110 import overflow_readiness as policy
from scripts.campaign_20261110.contracts import ContractError


def inputs(root):
    rows = {'infra/capacity_recovery/ledger.json': {'observed_at': 1000},
            'vast/metrics.json': {'observed_at_unix': 1000, 'spend_estimate_usd': 600,
                'future_compute_commitment_usd': 10, 'transfer_reserve_usd': 300},
            'vast/config.json': {'budget_usd': 3000, 'max_dph_gpu': 1, 'dispatch_job_ids': ['1']},
            'vast/queue/1.json': {'time_s': 7200}}
    for name, value in rows.items():
        p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value))
    return {'Reason': 'Resources', 'TimeLimit': '01:00:00'}


def test_zero_delay_requires_fresh_capacity_budget_and_counts_unmaterialized_queue(tmp_path):
    scheduler=inputs(tmp_path)
    proof=policy.check(tmp_path,scheduler,now=1000)
    assert proof['maximum_budget_including_queue_and_job']==913
    assert proof['queued_reservation_usd']==2 and proof['fleet_ceiling'] is None
    assert all(v['sha256'] for v in proof['sources'].values())
    with pytest.raises(ContractError,match='fresh cluster'):
        policy.check(tmp_path,scheduler,now=1181)


@pytest.mark.parametrize('reason',['Dependency','JobHeldUser','InvalidAccount',''])
def test_independent_ready_work_only(tmp_path,reason):
    scheduler=inputs(tmp_path);scheduler['Reason']=reason
    with pytest.raises(ContractError,match='resource-ready'):
        policy.check(tmp_path,scheduler,now=1000)


def test_no_unfunded_or_unbounded_overflow(tmp_path):
    scheduler=inputs(tmp_path)
    path=tmp_path/'vast/metrics.json';m=json.loads(path.read_text());m['spend_estimate_usd']=2999;path.write_text(json.dumps(m))
    with pytest.raises(ContractError,match='insufficient funded'):
        policy.check(tmp_path,scheduler,now=1000)
    scheduler['TimeLimit']='UNLIMITED'
    with pytest.raises(ContractError,match='bounded scheduler'):
        policy.check(tmp_path,scheduler,now=1000)
