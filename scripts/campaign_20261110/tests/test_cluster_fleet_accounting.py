from scripts.campaign_20261110.infra_cluster_fleet_accounting import parse


def row(jid,comment='',submission='',gpu='gres/gpu=1',elapsed='100',name='isg261110_prereq'):
    return '|'.join([jid,jid,'shiyuc',name,comment,'FAILED','2026-11-10T01:00:00','2026-11-10T01:01:00',
                     '2026-11-10T01:02:40',elapsed,gpu,'1:0',submission])+'|\n'


def test_allocation_failures_and_replay_count_without_credentials_or_double_counted_steps(tmp_path):
    text=row('123','campaign_20261110_prereq',submission='opaque private arguments')
    text+=row('123.batch','campaign_20261110_prereq')
    text+=row('124',submission=str(tmp_path/'tools/infra_open_loop.sh'),elapsed='200')
    text+=row('125','campaign_20261110_cpu',gpu='cpu=2,mem=3G',elapsed='1000')
    records,ambiguous,errors=parse(text,tmp_path)
    assert len(records)==3 and not ambiguous and not errors
    assert sum(row['allocated_gpu_seconds'] for row in records)==300
    assert all('submission' not in row for row in records)


def test_names_alone_do_not_establish_campaign_ownership_and_explicit_arrays_are_supported(tmp_path):
    text=row('123_0')+row('123_1')+row('999')+row('888','another_campaign',name='other_job')
    records,ambiguous,errors=parse(text,tmp_path,explicit=['123'])
    assert {row['job_id'] for row in records}=={'123_0','123_1'}
    assert [row['job_id'] for row in ambiguous]==['999'] and not errors


def test_conflicting_rows_or_bad_gpu_denominators_are_not_silently_combined(tmp_path):
    text=row('123','campaign_20261110_x')+row('123','campaign_20261110_x',elapsed='200')
    text+=row('124','campaign_20261110_x',elapsed='invalid')
    records,ambiguous,errors=parse(text,tmp_path)
    assert len(records)==1 and len(errors)==2


def test_array_display_identity_is_preserved_alongside_physical_allocation_id(tmp_path):
    text=row('900').replace('900|900|','900|123_0|')+row('901').replace('901|901|','901|123_1|')
    records,ambiguous,errors=parse(text,tmp_path,explicit=['123'])
    assert {(r['job_id'],r['array_task_id']) for r in records}=={('900','123_0'),('901','123_1')}
    assert not ambiguous and not errors
