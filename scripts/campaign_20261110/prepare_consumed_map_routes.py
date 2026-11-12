"""Materialize typed Vast transport for reviewed map chunks; never enqueue."""
import argparse
import json
from pathlib import Path
import time

from scripts.campaign_20261110 import vast_map_prebuild as maps
from scripts.campaign_20261110.vast_prebuild import sha, write, require


def prepare(root, native, helper, index, admission, base_policy, output, reader_registry):
    root, native, helper, index, admission, base_policy, output, reader_registry = map(Path, (root, native, helper, index, admission, base_policy, output, reader_registry))
    require(not output.exists(), 'Transport author output already exists')
    output.mkdir(parents=True); ref = lambda p: {'path': str(p), 'sha256': sha(p)}
    selected = json.loads(index.read_text()); approved = json.loads(admission.read_text()); sources = []; errors = []
    allowed = {r['manifest']['path']: r['manifest']['sha256'] for r in approved['approved_manifests']}
    registry = json.loads(reader_registry.read_text()); registry_items = list(registry['manifests'])
    base = json.loads(base_policy.read_text()); base['previous'] = ref(base_policy); base['admission_policy'] = ref(admission)
    base.update(max_new_jobs_per_cycle=128, max_examined_per_cycle=4096, max_cycle_seconds=15)
    for number, chunk in enumerate(reversed(selected['chunks'])):
        manifest = Path(chunk['manifest']['path']); require(allowed.get(str(manifest)) == sha(manifest) == chunk['manifest']['sha256'], 'Unreviewed source chunk')
        value = json.loads(manifest.read_text()); rows = []
        for i, ticket in enumerate(value['jobs']):
            key = ticket['physical_dedup_key']; token = 'consumed_standard_map_' + key[:24]
            try:
                job = maps.prepare_consumed(root, manifest, i, native, helper, token)
                proof = maps.verify(job, root)
                require(proof['physical_dedup_key'] == key, 'Transport key differs')
                rows.append({'source_index': i, 'physical_dedup_key': key, 'claim_path': ticket['claim_path'],
                    'submission_path': ticket['submission_path'], 'job_dir': str(job), 'native_inputs': ref(job / 'native_inputs.json')})
            except (OSError, ValueError, KeyError, TypeError) as exc:
                errors.append({'source_manifest': ref(manifest), 'source_index': i, 'physical_dedup_key': key,
                               'error': type(exc).__name__ + ': ' + str(exc), 'dispatch_authorized': False})
        route = output / ('route_' + manifest.stem + '.json')
        write(route, {'schema': 'isg-consumed-map-vast-route-v1', 'observed_at': time.time(), 'source_manifest': ref(manifest),
            'jobs': rows, 'helper': str(helper), 'helper_refs': [ref(helper / 'scripts/campaign_20261110' / name) for name in ('vast_prebuild.py', 'vast_map_prebuild.py', 'job_map_prebuild.sh')],
            'physical_claim_owner': '/root', 'enqueued': False, 'benchmark_credit': 0, 'production_authorized': False})
        sources.append({'id': 'consumed_standard_' + manifest.stem, 'enabled': True, 'route': ref(route)})
        registry_items.append({'manifest': ref(manifest), 'route': ref(route), 'output': str(output.parent / 'native_readers' / manifest.stem)})
        policy = output / ('intake_policy_%03d.json' % (number + 1))
        write(policy, {**base, 'sources': base['sources'] + sources, 'prepared_at': time.time(), 'transport_author': ref(__file__)})
        reg = output / ('verifier_registry_%03d.json' % (number + 1))
        write(reg, {'schema': 'isg-consumed-map-verifier-inputs-v1', 'owner': '/root', 'manifests': registry_items})
        write(output / 'status.json', {'schema': 'isg-consumed-map-route-author-v1', 'observed_at': time.time(),
            'ready_routes': len(sources), 'source_chunks': len(selected['chunks']), 'ready_jobs': sum(len(json.loads(Path(s['route']['path']).read_text())['jobs']) for s in sources),
            'latest_intake_policy': ref(policy), 'latest_verifier_registry': ref(reg), 'errors': errors,
            'complete': number + 1 == len(selected['chunks']), 'benchmark_credit': 0, 'dispatch_authority': False})
        print(json.dumps({'ready_route': ref(route), 'jobs': len(rows), 'intake_policy': ref(policy), 'verifier_registry': ref(reg), 'errors': len(errors)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'native', 'helper', 'index', 'admission', 'base-policy', 'output', 'reader-registry'): parser.add_argument('--' + name, required=True)
    a = parser.parse_args(); prepare(a.root, a.native, a.helper, a.index, a.admission, a.base_policy, a.output, a.reader_registry)
