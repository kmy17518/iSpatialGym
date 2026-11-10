import pytest

from scripts.campaign_20261110.replay_scheduling import dual_partition_resources


def node(name, gpu, partition='svl', state='MIXED'):
    return f'NodeName={name} Partitions={partition} CfgTRES=cpu=72,gres/gpu=8,gres/gpu:{gpu}=8 State={state}'


def test_dual_partition_keeps_supported_gpus_and_excludes_every_unsupported_node():
    raw = '\n'.join([node('svl10', 'titanrtx'), node('viscam6', 'a5000', 'viscam'),
        node('svl2', '2080ti'), node('visionlab-dgx1', 'v100'), node('svl9', 'titanrtx'),
        node('viscam-hgx-1', 'h200', 'viscam'), node('viscam4', 'a40', 'viscam', 'MIXED+DRAIN')])
    args, proof = dual_partition_resources(raw)
    assert proof['eligible_nodes'] == ['svl10', 'viscam6']
    assert set(proof['excluded_nodes']) == {'svl2', 'visionlab-dgx1', 'svl9', 'viscam-hgx-1', 'viscam4'}
    assert args[:6] == ['-A', 'viscam', '-p', 'svl,viscam', '--gres=gpu:1', '--exclude=svl2,svl9,viscam-hgx-1,viscam4,visionlab-dgx1']
    assert not any('nodelist' in argument for argument in args)
    assert proof['capacity_reserved'] is False


def test_missing_unknown_or_unsupported_inventory_cannot_schedule_any_gpu():
    for raw in ('', node('svl3', '2080ti'), node('svl10', 'titanrtx', state='UNKNOWN'),
                'NodeName=svl10 Partitions=svl CfgTRES=cpu=72 State=IDLE'):
        with pytest.raises(ValueError, match='no verified'):
            dual_partition_resources(raw)
