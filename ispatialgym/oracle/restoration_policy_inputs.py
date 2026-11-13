"""Explicit CLI/queue selection of a reviewed per-source restoration policy.

The manifest selects existing bounded policies. It grants no production
authority and does not bypass the loader's independent source execution gate.
"""
import json
import os
import re
from pathlib import Path
from scripts.campaign_20261110.contracts import file_digest, require
from .restoration_policy import bind


def from_arguments(args, tro_path, runtime, *, provided=None, environment=None):
    path = getattr(args, 'restoration_policy_manifest', None)
    sha = getattr(args, 'restoration_policy_manifest_sha256', None)
    if path is None and sha is None:
        return provided
    require(path is not None and isinstance(sha, str) and re.fullmatch('[0-9a-f]{64}', sha),
            'restoration policy manifest needs its exact SHA256')
    path = Path(path).resolve(); runtime = Path(runtime).resolve(); tro = Path(tro_path).resolve()
    require(file_digest(path) == sha, 'restoration policy manifest changed')
    value = json.loads(path.read_text()); root = Path(value['campaign_root']).resolve()
    require(value['schema'] == 'isg-bounded-loader-policy-manifest-v1'
            and value['campaign_id'] == root.name == 'campaign_20261110'
            and value['production_released'] is False and type(value['benchmark_credit']) is int
            and value['benchmark_credit'] == 0 and path.is_relative_to(root)
            and tro.is_relative_to(root) and runtime.is_relative_to(root/'code')
            and value['native_runtime'] == {'path': str(runtime/'.snapshot_commit'),
                'sha256': file_digest(runtime/'.snapshot_commit')},
            'restoration policy manifest scope or runtime differs')
    rows = value['entries']
    require(isinstance(rows, list) and 0 < len(rows) <= 24
            and len({r['source_tro']['path'] for r in rows}) == len(rows),
            'restoration policy source entries are missing or ambiguous')
    for row in rows:
        for reference in (row['source_tro'], row['request']):
            target = Path(reference['path']).resolve()
            require(target.is_relative_to(root) and str(target) == reference['path']
                    and file_digest(target) == reference['sha256'],
                    'restoration policy entry changed or escaped campaign')
    selected = [r for r in rows if r['source_tro']['path'] == str(tro)]
    require(len(selected) == 1, 'exact incoming source has no reviewed restoration policy')
    row = selected[0]
    policy = bind(root, row['request'], row['source_tro'], runtime)
    require(provided is None or provided == policy, 'CLI and caller restoration policies conflict')
    request = json.loads(Path(row['request']['path']).read_text())
    environment = os.environ if environment is None else environment
    require(all(environment.get(k) == v for k, v in request.get('source_environment', {}).items()),
            'CLI restoration source environment differs from reviewed request')
    return policy
