import json
from datetime import datetime,timezone
from pathlib import Path
import pytest
from scripts.campaign_20261110.dashboard.sync import deadline_checkpoint


def ledger():
    return {'campaign_id':'campaign_20261110','observed_at':'2026-11-12T01:00:00Z','accepted':{
        'counts':{'train':1142},'source':{'observed_at':'2026-11-12T00:59:00Z'}},
        'canonical':{'validated':100,'total':100},'sources':{'preparation_status':{'state':'stale','observed_at':'2026-11-12T00:00:00Z'}},
        'release_decisions':{'decision':{'standalone_training_allowed':False}},
        'preparation':{'source_summary':{'dependency_statuses':{'verified':4205,'not_applicable':2},
            'required_unique_logical_requests':48793,'selected_configuration_slots':14620,
            'configuration_slots_with_declared_dependencies_complete':20,'fully_served_configurations':0}}}


def test_canonical_completion_does_not_release_production_or_refresh_old_evidence(tmp_path):
    p=tmp_path/'execution_ledger.json';p.write_text(json.dumps(ledger()));before=p.read_bytes()
    x=deadline_checkpoint(tmp_path,datetime(2026,11,12,2,tzinfo=timezone.utc))
    assert p.read_bytes()==before
    assert x['observed_at']=='2026-11-12T01:00:00Z' and x['computed_at'].startswith('2026-11-12T02:')
    assert x['sources']['preparation_status']['state']=='stale'
    assert x['canonical']['validated']==100 and x['forecasts']['accepted_production']['status']=='gate_closed_eta_unavailable'
    assert x['remaining_unique_artifacts'] is None and x['unresolved_logical_requirements']==44586
    assert all(v['eta_range'] is None for v in x['forecasts'].values())
    assert x['forecasts']['accepted_production']['required_training_accepted_per_hour']==98858/17


def test_deadline_passed_is_not_shutdown_or_fabricated_throughput(tmp_path):
    (tmp_path/'execution_ledger.json').write_text(json.dumps(ledger()))
    x=deadline_checkpoint(tmp_path,datetime(2026,11,13,tzinfo=timezone.utc))
    assert x['hours_remaining']==0 and x['deadline_is_automatic_shutdown'] is False
    assert x['forecasts']['accepted_production']['required_training_accepted_per_hour'] is None


def test_wrong_campaign_preserves_prior_checkpoint(tmp_path):
    x=ledger();x['campaign_id']='older';(tmp_path/'execution_ledger.json').write_text(json.dumps(x))
    p=tmp_path/'streaming_deadline_checkpoint.json';p.write_text('previous')
    with pytest.raises(ValueError,match='campaign differs'):deadline_checkpoint(tmp_path)
    assert p.read_text()=='previous'
