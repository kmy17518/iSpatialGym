from copy import deepcopy
import pytest
from scripts.campaign_20261110.prepare_floating_vast import normalized_entry


def query():
    return dict(production_authorized=False,benchmark_credit=0,native_time_seconds=1800,
                hard_time_seconds=2400,scene='house',room='living_room_0',id='query_x',
                class_id='object_scaling-test',physical_dedup_key='abc',request={'sha256':'123'},
                sample_argv=['python','-u','-m','ispatialgym.sampling.sample_isg','--scene','house',
                             '--room','living_room_0','--max-attempts','3','--templates','1',
                             '--instances-per-template','1','--no-write-back','--out','${NATIVE_OUTPUT}'])


def test_adapter_changes_only_buffering_and_owned_output():
    original=query();before=deepcopy(original);entry=normalized_entry(original)
    assert original==before
    expected=before['sample_argv'];expected.pop(1);expected[-1]='${CAMPAIGN_OUT}/floating_feasibility/query_x'
    assert entry['sample_argv']==expected and entry['production_released'] is False
    assert entry['physical_dedup_key']==before['physical_dedup_key']


@pytest.mark.parametrize('change',[{'production_authorized':True},{'benchmark_credit':1},
                                 {'native_time_seconds':3600},{'room':'other_room'}])
def test_rejects_changed_scope_or_expanded_bounds(change):
    value=query();value.update(change)
    with pytest.raises(ValueError):normalized_entry(value)


def test_rejects_preexisting_index_or_different_draw_count():
    value=query();value['sample_argv']+=['--start-index','7']
    with pytest.raises(ValueError):normalized_entry(value)
    value=query();value['sample_argv'][value['sample_argv'].index('--max-attempts')+1]='100'
    with pytest.raises(ValueError):normalized_entry(value)
