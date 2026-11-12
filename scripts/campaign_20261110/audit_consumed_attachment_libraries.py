"""Read exact attachment builder outputs; distinguish positive, zero and failure."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time

from scripts.campaign_20261110.contracts import file_digest, require, write_json_atomic
from scripts.campaign_20261110.reservation_policy import active_policy


def audit(root, manifest):
    from ispatialgym.grasping import schema as S, candidates as C
    from ispatialgym.sampling.knowledge import is_ready
    root = Path(root); manifest = Path(manifest)
    ref = lambda p: {'path': str(p), 'sha256': file_digest(p)}
    pointer, reservations, assets = active_policy(root)
    require(pointer is not None, 'Reservation policy unavailable')
    expected_switches = S.candidate_switches(asdict(C.CandidateConfig()))
    rows = []; digest_cache = {}
    def checked(reference):
        path = reference['path']
        if path not in digest_cache: digest_cache[path] = file_digest(path)
        require(digest_cache[path] == reference['sha256'], 'Input bytes changed: ' + path)
    for ticket in json.loads(manifest.read_text())['jobs']:
        directory = Path(ticket['prepared']['path']).parent
        row = {'id': ticket['id'], 'physical_dedup_key': ticket['physical_dedup_key'],
               'configuration_ids': ticket['configuration_ids'], 'selected_dependency_ids': ticket['selected_dependency_ids'],
               'prepared': ticket['prepared'], 'native_reader_passed': False}
        try:
            terminal_path = directory / 'terminal.json'
            if not terminal_path.exists():
                row['status'] = 'terminal_not_observed'; rows.append(row); continue
            terminal = json.loads(terminal_path.read_text()); row['terminal'] = ref(terminal_path)
            row['job_id'] = terminal['job_id']; row['duration_seconds'] = terminal['finished_at'] - terminal['started_at']
            checked(ticket['prepared']); prepared = json.loads(Path(ticket['prepared']['path']).read_text())
            for reference in prepared['input_refs']: checked(reference)
            require(terminal['prepared_sha256'] == ticket['prepared']['sha256'] and
                    terminal['physical_dedup_key'] == ticket['physical_dedup_key'], 'Terminal identity differs')
            require(terminal['returncode'] == 0 and terminal['native_validator_done'], 'Native builder did not finish validation')
            entry = prepared['physical_identity']['job']; row['entry'] = entry
            model_key = entry['category'] + '/' + entry['model']
            require(model_key not in reservations['train_exclude_models'] and
                    entry['category'] not in reservations['train_exclude_categories'] and
                    entry['asset_sha256'] not in assets['reserved_asset_hashes'] and is_ready(entry['model']), 'Reserved or unready target')
            checked(terminal['library']); path = Path(terminal['library']['path']); data = json.loads(path.read_text())
            candidates = json.loads((path.parent / 'candidates.json').read_text())
            validation = json.loads((path.parent / 'validation.json').read_text())
            require(data['schema'] == S.LIBRARY and data['constraint'] == 'attachment_free', 'Not an attachment library')
            require(all(data[k] == entry[k] == candidates[k] for k in ('category', 'model', 'dataset', 'scale', 'constraint')), 'Native model/scale/constraint differs')
            require(data['arm'] == candidates['arm'] == validation['config']['arm'] == data['policy']['arm'] == 'right', 'Native arm differs')
            require(data['provenance']['validator'] == validation['validator'] == 'isg-tabletop-validator/1', 'Native validator differs')
            require(data['policy_key'] == validation['policy_key'] == candidates['policy_key'] and data['physics_profile'], 'Native physics/policy differs')
            require(data['candidate_switches'] == S.candidate_switches(candidates['config']) == expected_switches and
                    data['attach_axis'] == candidates['config']['attach_axis'] == C.CandidateConfig().attach_axis,
                    'Native default candidate/attachment policy differs')
            require(not any(t.get('error') or t.get('traceback') for t in validation['trials']), 'Native validator recorded an exception')
            lib = S.load_model(entry['category'], entry['model'], entry['scale'], str(directory / 'glib'),
                               constraint='attachment_free', rest_hint_wxyz=None,
                               require={'candidate_switches': expected_switches, 'attach_axis': C.CandidateConfig().attach_axis})
            require(lib is not None and Path(lib.path) == path and len(lib.grasps()) == len(data['grasps']), 'Native consumer resolver differs')
            valid = {(t.get('candidate'), t.get('rest_id')) for t in validation['trials'] if t.get('verdict') == 'valid'}
            require(all(g['physics'] == 'tested' and (g['name'], g['rest_id']) in valid for g in data['grasps']), 'Positive grasp lacks native trial')
            for rest in {g['rest_id'] for g in data['grasps']}:
                S.check_compatible(lib, scale=entry['scale'], rest_id=rest, rest_kind='stable')
            row.update(status='verified_positive' if data['grasps'] else 'valid_zero', native_reader_passed=True,
                       positive_grasps=len(data['grasps']), native_identity=lib.identity(), native_identity_key=lib.identity_key(),
                       library=terminal['library'], files=[ref(p) for p in sorted(path.parent.iterdir()) if p.is_file()],
                       trial_reasons=dict(Counter(reason for t in validation['trials'] for reason in t.get('reasons', []))))
        except (OSError, ValueError, KeyError, TypeError) as exc:
            row.update(status='observation_unavailable' if isinstance(exc, OSError) else 'invalid_or_failed', reason=str(exc))
        rows.append(row)
    return {'schema': 'isg-consumed-attachment-native-reader-v1', 'observed_at': time.time(), 'owner': '/root/templates_assets',
            'source': ref(manifest), 'reservation_policy': pointer, 'native_reader': ref(S.__file__),
            'native_candidate_generator': ref(C.__file__), 'rows': rows, 'summary': dict(Counter(r['status'] for r in rows)),
            'positive_grasps': sum(r.get('positive_grasps', 0) for r in rows if r['status'] == 'verified_positive'),
            'consumer_aliases_and_maps_separate': True, 'benchmark_credit': 0, 'production_authorized': False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ('root', 'manifest', 'output'): parser.add_argument('--' + key, required=True)
    args = parser.parse_args(); output = Path(args.output)
    require(not output.exists() and output.resolve().is_relative_to(Path(args.root).resolve()), 'Fresh owned output required')
    result = audit(args.root, args.manifest); write_json_atomic(output, result)
    print(json.dumps({'output': str(output), 'sha256': file_digest(output), 'summary': result['summary'], 'positive_grasps': result['positive_grasps']}))


if __name__ == '__main__': main()
