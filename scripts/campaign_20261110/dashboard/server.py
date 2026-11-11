"""Loopback-only, read-only dashboard. Requests read local monitoring mirrors only."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import re
from urllib.parse import urlparse

STATIC = Path(__file__).parent / 'static'
TARGETS = {'eval_standard': 1000, 'eval_model': 891, 'eval_category': 109, 'train': 100000}
# These two campaign geometry probes predate the campaign job-name prefix.
# Exact IDs prevent unrelated similarly named probes from entering this view.
GEOMETRY_PROBE_IDS = {'18423600', '18423963'}
SECRET = re.compile(r'(password|secret|credential|access_token|api_key|authorization|private_key)', re.I)


def sanitize(value):
    if isinstance(value, dict):
        return {k: sanitize(v) for k, v in value.items() if not SECRET.search(str(k)) and k not in ('job_env', 'environment', 'command', 'env', '_key_name')}
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        value = re.sub(r'(?i)(bearer\s+)[A-Za-z0-9._-]+', r'\1[redacted]', value)
        return re.sub(r'(?i)((?:token|password|secret|api_key)=)[^\s&]+', r'\1[redacted]', value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_source(path, *, mapping=True):
    try:
        data = json.loads(path.read_text())
        if mapping and not isinstance(data, dict):
            raise ValueError('Expected a JSON observation object')
        return data, {'available': True, 'file_mtime': path.stat().st_mtime}
    except (OSError, ValueError) as exc:
        return {}, {'available': False, 'error': type(exc).__name__ + ': ' + str(exc)}


def snapshot(root, inventory):
    live, ls = read_source(root / 'live.json')
    cluster, cs = read_source(root / 'infra/cluster_snapshot.json')
    evidence, es = read_source(root / 'dashboard/dashboard_evidence.json')
    group_yield, ys = read_source(root / 'dashboard/goal_group_yield.json')
    heldout, hrs = read_source(root / 'dashboard/heldout_recovery.json')
    capacity_recovery, crs = read_source(root / 'dashboard/capacity_recovery.json')
    recovery, rs = read_source(root / 'dashboard/standard_recovery.json')
    canonical, cvs = read_source(root / 'dashboard/canonical_validation.json')
    training, tes = read_source(root / 'dashboard/training_eligibility.json')
    production_gate, pgs = read_source(root / 'dashboard/training_production_gate.json')
    containment, cts = read_source(root / 'dashboard/training_containment.json')
    requirements, reqs = read_source(root / 'dashboard/training_requirements.json')
    training_stream, tss = read_source(root / 'dashboard/training_stream.json')
    existing_reviews, ers = read_source(root / 'dashboard/existing_native_reviews.json')
    change_impact, cis = read_source(root / 'dashboard/change_impact.json')
    current_eligibility, ces = read_source(root / 'dashboard/current_eligibility.json')
    fleet, fs = read_source(root / 'dashboard/vast_fleet_gpu_time.json')
    cluster_fleet, cfs = read_source(root / 'dashboard/cluster_fleet_gpu_time.json')
    yield_error, yes = read_source(root / 'dashboard/goal_group_yield_error.json')
    audit_error, aes = read_source(root / 'dashboard/dashboard_evidence_error.json')
    sync, ss = read_source(root / 'dashboard/sync_status.json')
    candidates, ins = read_source(inventory / 'train_candidate_summary.json')
    runtime, _ = read_source(root / 'infra/runtime.json')
    if live and live.get('campaign_id') != 'campaign_20261110':
        live = {}; ls = {'available': False, 'error': 'Campaign identity mismatch'}
    if evidence and evidence.get('campaign_id') != 'campaign_20261110':
        evidence = {}; es = {'available': False, 'error': 'Campaign identity mismatch'}
    if group_yield and group_yield.get('campaign_id') != 'campaign_20261110':
        group_yield = {}; ys = {'available': False, 'error': 'Campaign identity mismatch'}
    if capacity_recovery and capacity_recovery.get('campaign_id') != 'campaign_20261110':
        capacity_recovery = {}; crs = {'available': False, 'error': 'Campaign identity mismatch'}
    if heldout and heldout.get('campaign_id') != 'campaign_20261110':
        heldout = {}; hrs = {'available': False, 'error': 'Campaign identity mismatch'}
    if recovery and recovery.get('campaign_id') != 'campaign_20261110':
        recovery = {}; rs = {'available': False, 'error': 'Campaign identity mismatch'}
    if canonical and canonical.get('campaign_id') != 'campaign_20261110':
        canonical = {}; cvs = {'available': False, 'error': 'Campaign identity mismatch'}
    if training and training.get('campaign_id') != 'campaign_20261110':
        training = {}; tes = {'available': False, 'error': 'Campaign identity mismatch'}
    if production_gate and production_gate.get('campaign_id') != 'campaign_20261110':
        production_gate = {}; pgs = {'available': False, 'error': 'Campaign identity mismatch'}
    if containment and containment.get('campaign_id') != 'campaign_20261110':
        containment = {}; cts = {'available': False, 'error': 'Campaign identity mismatch'}
    if requirements and requirements.get('campaign_id') != 'campaign_20261110':
        requirements = {}; reqs = {'available': False, 'error': 'Campaign identity mismatch'}
    if current_eligibility and current_eligibility.get('campaign_id') != 'campaign_20261110':
        current_eligibility = {}; ces = {'available': False, 'error': 'Campaign identity mismatch'}
    if change_impact and change_impact.get('campaign_id') != 'campaign_20261110':
        change_impact = {}; cis = {'available': False, 'error': 'Campaign identity mismatch'}
    if training_stream and training_stream.get('campaign_id') != 'campaign_20261110':
        training_stream = {}; tss = {'available': False, 'error': 'Campaign identity mismatch'}
    if existing_reviews and existing_reviews.get('campaign_id') != 'campaign_20261110':
        existing_reviews = {}; ers = {'available': False, 'error': 'Campaign identity mismatch'}
    if fleet and fleet.get('campaign_id') != 'campaign_20261110':
        fleet = {}; fs = {'available': False, 'error': 'Campaign identity mismatch'}
    if cluster_fleet and cluster_fleet.get('campaign_id') != 'campaign_20261110':
        cluster_fleet = {}; cfs = {'available': False, 'error': 'Campaign identity mismatch'}
    live_keys = ('updated_at', 'cluster_observed_at', 'cluster_probe_error', 'vast_observed_at',
        'campaign_vast_metrics', 'acceptance_evidence', 'annotated_coverage', 'campaign_results',
        'cluster_by_user_partition', 'cluster_capacity', 'family_controller', 'pending')
    live = {k: v for k, v in live.items() if k in live_keys}
    gates = []
    for jid, row in cluster.get('validation', {}).get('gates', {}).items():
        gate = row.get('gate') or {}
        result = gate.get('result') or {}
        gates.append({'id': jid, 'class_id': row.get('key'), 'status': row.get('status'),
            'demo_successes': row.get('demo_successes', 0), 'demo_failures': row.get('demo_failures', 0),
            'replay_job_id': row.get('replay_job_id'), 'source_job_dir': row.get('source_job_dir'),
            'output': row.get('output'), 'at': gate.get('validated_at', row.get('updated_at')),
            'reason': row.get('error') or result.get('reason'), 'success': gate.get('success'),
            'replay_mode': gate.get('replay_mode'),
            'recovery': [{k: attempt.get(k) for k in ('status', 'replay_job_id', 'output', 'error')}
                for attempt in row.get('attempt_history', [])],
            'retry_request': row.get('retry_request'),
            'report_paths': gate.get('replay_meta_paths', [])})
    campaign_ids = {str(j.get('job_id')) for j in runtime.get('jobs', [])} | GEOMETRY_PROBE_IDS
    campaign_ids.update(str(g[k]) for g in gates for k in ('id', 'replay_job_id') if g.get(k))
    jobs = [dict(j, campaign_owned=True) for j in cluster.get('jobs', [])
        if str(j.get('id')) in campaign_ids or j.get('name', '').startswith('isg261110_')]
    cluster_view = {k: cluster.get(k) for k in ('checked_at', 'summary', 'by_user_partition', 'capacity', 'nodes', 'qos', 'probe_error')}
    cluster_view['jobs'] = jobs
    return sanitize({'campaign_id': 'campaign_20261110', 'served_at': datetime.now(timezone.utc).isoformat(),
        'targets': TARGETS, 'deadline': '2026-11-12T19:00:00Z', 'refresh_seconds': 30,
        'sources': {'capacity_recovery': crs, 'heldout_recovery': hrs, 'live': ls, 'cluster': cs, 'evidence': es, 'inventory': ins, 'sync': ss, 'group_yield': ys, 'fleet_gpu_time': fs, 'cluster_fleet_gpu_time': cfs, 'standard_recovery': rs, 'training_eligibility': tes, 'training_production_gate': pgs, 'training_requirements': reqs, 'canonical_validation': cvs, 'training_containment': cts, 'training_stream': tss, 'change_impact': cis, 'current_eligibility': ces},
        'training_eligibility': {**{k:v for k,v in training.items() if k not in ('records','inputs','cohorts')},
            'cohorts': [{k:v for k,v in row.items() if k!='instance_ids'} for row in training.get('cohorts',[])]} if training else {},
        'training_production_gate': production_gate,
        'training_containment': containment,
        'training_requirements': requirements,
        'training_stream': training_stream,
        'existing_native_reviews': existing_reviews,
        'existing_native_reviews_source': ers,
        'current_eligibility': {k:v for k,v in current_eligibility.items() if k not in ('records','groups','decisions')},
        'change_impact': {k:v for k,v in change_impact.items() if k not in ('decisions','inventory')},
        'heldout_recovery': heldout,
        'capacity_recovery': capacity_recovery,
        'standard_recovery': recovery,
        'canonical_validation': canonical,
        'cluster_fleet_gpu_time': {**{k:v for k,v in cluster_fleet.items() if k != 'records'}, 'allocation_count': len(cluster_fleet.get('records', []))},
        'fleet_gpu_time': {k:v for k,v in fleet.items() if k not in ('rentals','sources','attempts_missing_complete_intervals')},
        'group_yield': {k:v for k,v in group_yield.items() if k not in ('sources','groups','jobs')},
        'yield_error': yield_error if yes['available'] else {},
        'audit_error': audit_error if aes['available'] else {}, 'sync': sync,
        'live': live, 'evidence': evidence, 'gates': gates,
        'cluster': cluster_view,
        'candidates': {k: candidates.get(k) for k in ('candidate_templates', 'represented_standard_classes', 'per_class', 'per_task', 'reservation_sha256', 'schema_revision', 'readiness')}})


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        host = self.headers.get('Host', '').split(':')[0]
        if host not in ('127.0.0.1', 'localhost'):
            return self.send_body(403, b'Loopback access only', 'text/plain')
        path = urlparse(self.path).path
        if path == '/api/status':
            data = snapshot(self.server.data_root, self.server.inventory)
            return self.send_body(200, json.dumps(data, allow_nan=False).encode(), 'application/json')
        if path in ('/reports/live', '/reports/return', '/reports/goal-yield-text'):
            try:
                name = {'/reports/live':'LIVE.md','/reports/return':'RETURN_REPORT.txt','/reports/goal-yield-text':'GOAL_GROUP_YIELD.md'}[path]
                return self.send_body(200, (self.server.data_root / name).read_bytes(), 'text/plain; charset=utf-8')
            except OSError:
                return self.send_body(404, b'Report unavailable', 'text/plain')
        audits = {'/reports/capacity-recovery': 'capacity_recovery.json', '/reports/heldout-recovery': 'heldout_recovery.json', '/reports/reconciliation': 'reconciliation.json',
            '/reports/canonical-validation': 'canonical_validation.json',
            '/reports/training-eligibility': 'training_eligibility.json',
            '/reports/training-production-gate': 'training_production_gate.json',
            '/reports/training-containment': 'training_containment.json',
            '/reports/training-requirements': 'training_requirements.json',
            '/reports/training-stream': 'training_stream.json',
            '/reports/existing-native-reviews': 'existing_native_reviews.json',
            '/reports/change-impact': 'change_impact.json',
            '/reports/current-eligibility': 'current_eligibility.json',
            '/reports/standard-recovery': 'standard_recovery.json',
            '/reports/replay-quality': 'gate_quality_audit.json',
            '/reports/gpu-denominator-coverage': 'cohort_rental_denominator_coverage.json',
            '/reports/cluster-fleet-gpu-time': 'cluster_fleet_gpu_time.json',
            '/reports/fleet-gpu-time': 'vast_fleet_gpu_time.json',
            '/reports/accepted-release': 'accepted_release.json',
            '/reports/goal-yield': 'goal_group_yield.json',
            '/reports/vast-runtime': 'vast_runtime_resources.json', '/reports/browser-qa': 'browser_check.json'}
        if path in audits:
            data, status = read_source(self.server.data_root / 'dashboard' / audits[path], mapping=False)
            if path in ('/reports/existing-native-reviews', '/reports/capacity-recovery', '/reports/heldout-recovery', '/reports/current-eligibility', '/reports/change-impact', '/reports/training-stream', '/reports/training-containment', '/reports/canonical-validation', '/reports/training-eligibility', '/reports/training-production-gate', '/reports/training-requirements', '/reports/standard-recovery', '/reports/goal-yield', '/reports/accepted-release', '/reports/fleet-gpu-time', '/reports/cluster-fleet-gpu-time', '/reports/replay-quality', '/reports/gpu-denominator-coverage') and status['available'] and (
                    not isinstance(data, dict) or data.get('campaign_id') != 'campaign_20261110'):
                data = {}; status = {'available': False, 'error': 'Campaign identity mismatch'}
            return self.send_body(200 if status['available'] else 404,
                json.dumps(sanitize(data if status['available'] else status), indent=2).encode(), 'application/json')
        if path in ('/reports/evidence', '/reports/validation', '/reports/resources'):
            data = snapshot(self.server.data_root, self.server.inventory)
            item = data[{'/reports/evidence': 'evidence', '/reports/validation': 'gates', '/reports/resources': 'cluster'}[path]]
            return self.send_body(200, json.dumps(item, indent=2).encode(), 'application/json')
        if path.startswith('/reports/job/'):
            jid = path.rsplit('/', 1)[-1]
            rows = [row for row in snapshot(self.server.data_root, self.server.inventory)['gates'] if row['id'] == jid]
            return self.send_body(200 if rows else 404, json.dumps(rows[0] if rows else {'error':'Unknown campaign job'}, indent=2).encode(), 'application/json')
        files = {'/': ('index.html', 'text/html; charset=utf-8'), '/app.js': ('app.js', 'text/javascript'), '/style.css': ('style.css', 'text/css')}
        if path in files:
            name, mime = files[path]
            return self.send_body(200, (STATIC / name).read_bytes(), mime)
        self.send_body(404, b'Not found', 'text/plain')

    def do_HEAD(self):
        self.send_body(405, b'Read-only GET endpoints', 'text/plain')

    do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = do_HEAD

    def send_body(self, code, body, mime):
        self.send_response(code)
        self.send_header('Content-Type', mime)
        self.send_header('Content-Length', str(len(body)))
        self.send_header('Cache-Control', 'no-store')
        self.send_header('X-Content-Type-Options', 'nosniff')
        self.send_header('Referrer-Policy', 'no-referrer')
        self.send_header('Content-Security-Policy', "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'")
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', required=True, type=Path)
    parser.add_argument('--inventory', required=True, type=Path)
    parser.add_argument('--port', type=int, default=8765)
    args = parser.parse_args()
    server = ThreadingHTTPServer(('127.0.0.1', args.port), Handler)
    server.data_root, server.inventory = args.data_root.resolve(), args.inventory.resolve()
    print(f'Campaign dashboard: http://127.0.0.1:{args.port}/', flush=True)
    server.serve_forever()


if __name__ == '__main__':
    main()
