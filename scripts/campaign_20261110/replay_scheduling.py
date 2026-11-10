"""Schedule primary-account replays on verified campaign-compatible GPU nodes."""
import re

ALLOWED_GPUS = {'titanrtx', '3090', 'a40', 'a5000', 'a6000', 'l40s'}
EXCLUDED_NODES = {'svl9', 'viscam5', 'viscam7', 'viscam11', 'viscam12',
                  'viscam14', 'viscam15', 'viscam16', 'viscam-hgx-1', 'viscam-hgx-2'}


def dual_partition_resources(node_output):
    """Allow either partition, excluding every unsupported or unhealthy node.

    This is an eligibility inventory, not a reservation or free-capacity claim.
    Slurm applies both partition quotas and chooses one node for each replay.
    """
    excluded, eligible = set(), set()
    for line in node_output.splitlines():
        row = dict(re.findall(r'(?:^|\s)(\w+)=(\S*)', line))
        if not set(row.get('Partitions', '').split(',')) & {'svl', 'viscam'}:
            continue
        node = row['NodeName']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', node):
            raise ValueError('unsafe scheduler node name')
        tres = dict(item.split('=', 1) for item in row.get('CfgTRES', '').split(',') if '=' in item)
        models = {key.split(':', 1)[1] for key in tres if key.startswith('gres/gpu:') and int(tres[key]) > 0}
        supported = (models and models <= ALLOWED_GPUS and node not in EXCLUDED_NODES
            and not any(flag in row.get('State', '') for flag in ('DRAIN', 'DOWN', 'FAIL', 'UNKNOWN')))
        (eligible if supported else excluded).add(node)
    if not eligible:
        raise ValueError('no verified compatible replay nodes')
    arguments = ['-A', 'viscam', '-p', 'svl,viscam', '--gres=gpu:1']
    if excluded:
        arguments += ['--exclude=' + ','.join(sorted(excluded))]
    return arguments, {'eligible_nodes': sorted(eligible), 'excluded_nodes': sorted(excluded),
                       'account': 'viscam', 'partitions': ['svl', 'viscam'], 'capacity_reserved': False}
