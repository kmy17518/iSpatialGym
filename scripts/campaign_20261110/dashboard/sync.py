"""Independent read-only SSH mirror; never invoked by HTTP requests."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
from zoneinfo import ZoneInfo

REMOTE = '/vision/u/minyeong/projects/isg_home/campaign_20261110/infra'
FILES = ('dashboard_evidence.json', 'dashboard_evidence_error.json', 'goal_group_yield.json', 'goal_group_yield_error.json')
FLEET_FILE = 'vast_fleet_gpu_time.json'
CLUSTER_FLEET_FILE = 'cluster_fleet_gpu_time.json'
AUDIT_FILES = {'canonical_validation.json': 'infra/standard_recovery/canonical_validation.json','gate_quality_audit.json': 'validation/gate_quality_audit.json',
               'training_eligibility.json': 'accounting/training_eligibility.json',
               'training_production_gate.json': 'controllers/training_production_gate.json',
               'training_containment.json': 'infra/training_production_gate/containment_status.json',
               'training_requirements.json': 'manifests/training_requirements.json',
               'standard_recovery.json': 'infra/standard_recovery/ledger.json',
               'cohort_rental_denominator_coverage.json': 'accounting/cohort_rental_denominator_coverage.json'}


def atomic(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def sync(output, ssh_options=()):
    remote = "import json,pathlib; p=pathlib.Path(%r); names=%r; data={n:json.loads((p/n).read_text()) for n in names if (p/n).exists()}; report=p/'GOAL_GROUP_YIELD.md'; data['goal_yield_report']=report.read_text() if report.exists() else None; data.update({n:json.loads((p.parent/'accounting'/n).read_text()) for n in %r if (p.parent/'accounting'/n).exists()}); print(json.dumps(data))" % (REMOTE, FILES, (FLEET_FILE, CLUSTER_FLEET_FILE))
    remote = remote.replace('print(json.dumps(data))',
        "data.update({n:json.loads((p.parent/relative).read_text()) for n,relative in %r.items() if (p.parent/relative).exists()}); print(json.dumps(data))" % AUDIT_FILES)
    # The fixed remote program reads only the allowlisted campaign audit files.
    command = "python3 - <<'PY'\n" + remote + "\nPY"
    result = subprocess.run(['ssh', '-oBatchMode=yes', '-oConnectTimeout=15',
        *ssh_options, 'minyeong@svl11', command], text=True, capture_output=True, timeout=45, check=True)
    bundle = json.loads(result.stdout)
    evidence = bundle.get('dashboard_evidence.json')
    if not evidence or evidence.get('campaign_id') != 'campaign_20261110':
        raise ValueError('Current campaign allocation evidence is unavailable or mismatched')
    for name in FILES:
        if name in bundle:
            atomic(output / name, bundle[name])
    for name in (FLEET_FILE, CLUSTER_FLEET_FILE):
        fleet = bundle.get(name) or {}
        if fleet.get('campaign_id') == 'campaign_20261110':
            atomic(output / name, fleet)
    for name in AUDIT_FILES:
        audit = bundle.get(name) or {}
        if audit.get('campaign_id') == 'campaign_20261110':
            atomic(output / name, audit)
    if bundle.get('goal_yield_report'):
        target=output.parent/'GOAL_GROUP_YIELD.md';temporary=target.with_suffix('.tmp')
        temporary.write_text(bundle['goal_yield_report']);temporary.replace(target)
    return evidence['observed_at']


def return_report(output):
    """Refresh the owner's dated report independently of browser requests."""
    evidence = json.loads((output / 'dashboard_evidence.json').read_text())
    live = json.loads((output.parent / 'live.json').read_text())
    notes_path = output / 'operator_notes.json'
    notes = json.loads(notes_path.read_text()) if notes_path.exists() else {}
    now = datetime.now(ZoneInfo('America/Chicago')).isoformat()
    def chicago(value):
        if not value:
            return 'unavailable'
        observed = datetime.fromisoformat(value.replace('Z', '+00:00'))
        age = (datetime.now(timezone.utc) - observed).total_seconds()
        return observed.astimezone(ZoneInfo('America/Chicago')).strftime('%b %d, %Y %I:%M:%S %p %Z') + (' [STALE]' if age > 300 else '')
    def dollars(value):
        return f'${value:,.2f}' if isinstance(value, (int, float)) else 'unavailable'
    lines = ['CAMPAIGN 20261110 — PROGRESS FOR YOUR RETURN', '', 'Report refreshed: ' + now,
        'Browser: http://127.0.0.1:8765/ (local, read-only, 30-second refresh)', '',
        'Acceptance observation: ' + chicago(evidence['observed_at']),
        'Accepted counts come from locked SQL and three camera ledgers.']
    names = {'eval_standard': 'Standard eval', 'eval_model': 'Model holdout',
        'eval_category': 'Category/room holdout', 'train': 'Training'}
    for key, title in names.items():
        value, target = evidence['counts'][key], evidence['targets'][key]
        if key == 'train':
            eligibility_path=output/'training_eligibility.json'
            eligibility=json.loads(eligibility_path.read_text()) if eligibility_path.exists() else {}
            eligible=eligibility.get('eligible_accepted') if eligibility.get('campaign_id')=='campaign_20261110' else None
            lines.append(f'Training historically accepted: {value:,}; eligible goal count: '+(f'{eligible:,} / {target:,}; {max(0,target-eligible):,} remaining' if isinstance(eligible,int) else 'UNAVAILABLE — eligibility audit pending; historical totals do not count toward completion.'))
            lines.append('Training excluded/flagged: '+str(eligibility.get('excluded','unavailable'))+'; unverified: '+str(eligibility.get('unverified','unavailable'))+'; replacements: '+str(eligibility.get('replacement_count','unavailable')))
            continue
        lines.append(f'{title}: {value:,} / {target:,}; {max(0, target-value):,} remaining')
    gate_path=output/'training_production_gate.json'
    gate=json.loads(gate_path.read_text()) if gate_path.exists() else {}
    lines += ['','STANDALONE TRAINING PRODUCTION GATE',json.dumps(gate) if gate else 'Gate observation unavailable; training remains ineligible without proof.',
        'Requirements and evidence: http://127.0.0.1:8765/#training-production',
        'Historical acceptance and artifacts are preserved. Goal progress and replacement capacity use eligibility evidence.']
    canonical_path=output/'canonical_validation.json'
    if canonical_path.exists():
        canonical=json.loads(canonical_path.read_text())
        if canonical.get('campaign_id')=='campaign_20261110':
            lines += ['','CANONICAL TEMPLATE VALIDATION', 'Observed: '+chicago(canonical.get('observed_at')),
                str(canonical.get('validated'))+' / 100 exact canonical classes verified; states: '+json.dumps(canonical.get('state_counts')),
                'This artifact audit is distinct from historical accepted class coverage. Five owner-artifact holds are excluded.',
                'Proof: http://127.0.0.1:8765/reports/canonical-validation']
    recovery_path=output/'standard_recovery.json'
    if recovery_path.exists():
        recovery=json.loads(recovery_path.read_text())
        if recovery.get('campaign_id')=='campaign_20261110':
            lines += ['', 'STANDARD EVAL RECOVERY — COVERAGE PRIORITY',
                'Ledger observation: '+chicago(recovery.get('observed_at')),
                f"Covered classes: {recovery['covered_classes']}/{recovery['standard_classes']}; zero classes: {recovery['zero_classes']}.",
                'New first accepted eval classes: '+(', '.join(recovery['newly_covered']) or 'none yet'),
                'Zero tasks: '+', '.join(recovery['zero_tasks']), recovery['activity_semantics'],
                'Exact per-class ledger: http://127.0.0.1:8765/#standard-recovery']
            report_lines=['STANDARD EVAL RECOVERY — campaign_20261110', 'Observed: '+chicago(recovery.get('observed_at')),
                f"Covered classes: {recovery['covered_classes']}/{recovery['standard_classes']}; remaining zero: {recovery['zero_classes']}",
                'First accepted eval since baseline: '+(', '.join(recovery['newly_covered']) or 'none yet'), recovery['activity_semantics'], '']
            for row in recovery['classes']:
                report_lines += [f"{row['class_id']}: {row['accepted_standard']}/10 accepted — {row['state']}",
                    f"Owner: {row['owner']}; engineering: {row['engineering_status']}; worker observation: {row['worker_observed_at']}",
                    'Blocker: '+str(row['blocker']), 'Next action: '+str(row['next_action']),
                    'Active jobs/controller: '+json.dumps([row['active_jobs'],row['controller']]),
                    'Exact configuration/runtime/native/demo/replay: '+json.dumps({k:row.get(k) for k in ('configuration','runtime','native','demo','replay')}),
                    'Successful reference / causal diagnosis: '+json.dumps(row.get('diagnosis',{})), '']
            target=output.parent/'STANDARD_RECOVERY.txt';temp=target.with_suffix('.tmp');temp.write_text('\n'.join(report_lines)+'\n');temp.replace(target)
    lines += ['', 'Resource observation: ' + chicago(live.get('cluster_observed_at')),
        f"Campaign cluster: {live.get('campaign_cluster_running_gpus', 'unavailable')} GPUs running; "
        f"{live.get('campaign_cluster_pending_gpus', 'unavailable')} GPUs pending."]
    yield_path = output / 'goal_group_yield.json'
    if yield_path.exists():
        y = json.loads(yield_path.read_text())
        lines += ['', 'GOAL-GROUP YIELD (camera tasks excluded)', 'Observed: ' + chicago(y['observed_at'])]
        for mode, row in y['summary'].items():
            h=row['success_histogram']; a=row['accepted']
            percentage=row['at_least_two_percent']
            lines += [f"{mode}: {row['completed_groups']} completed; 0/1/2/3 distinct successful members: "
                + '/'.join(str(h[str(i)]) for i in range(4)) + f"; >=2: {percentage:.1f}%" if percentage is not None else f'{mode}: no completed groups']
            lines += [f"  Accepted standard/model/category/train: {a['eval_standard']}/{a['eval_model']}/{a['eval_category']}/{a['train']}",
                f"  Successful members {row['distinct_successful_members']}; observed attempted slots >= {row['observed_attempted_slots']}; exact denominator "
                + ('available' if row['attempt_denominator_complete'] else 'unavailable for historical runs'),
                '  Group states: '+json.dumps(row['pending_running_unstarted'])]
        lines += ['All-backend GPU-hour yield is unavailable where historical startup/retry intervals are missing.',
            'Exact collection-only Slurm rates, class/runtime breakdowns and evidence: http://127.0.0.1:8765/#goal-yield']
    vast = live.get('campaign_vast_metrics') or {}
    fleet_path = output / FLEET_FILE
    if fleet_path.exists():
        f = json.loads(fleet_path.read_text())
        if f.get('campaign_id') == 'campaign_20261110':
            lines += ['', 'Vast fleet GPU-time observation: ' + chicago(datetime.fromtimestamp(f['observed_at_unix'], timezone.utc).isoformat()),
                f"Post-instrumentation rental window: {f['fleet_window_estimated_gpu_seconds']/3600:.3f} GPU-hours; "
                f"{f['fleet_window_measured_attributed_worker_gpu_seconds']/3600:.3f} attributed to retained worker intervals; "
                f"{f['fleet_window_unattributed_gpu_seconds']/3600:.3f} not yet attributed.",
                f.get('boundary_precision', ''),
                'This includes fleet startup, idle and failed attempts. It is not a completed-group yield denominator; active work and historical gaps remain separate.']
            if f.get('recorded_fleet_estimated_gpu_seconds') is not None:
                lines += [f"All recorded campaign rentals: {f['recorded_fleet_estimated_gpu_seconds']/3600:.3f} estimated GPU-hours; "
                    f"{f['recorded_fleet_measured_attributed_worker_gpu_seconds']/3600:.3f} measured worker GPU-hours; "
                    f"{f['recorded_fleet_unattributed_gpu_seconds']/3600:.3f} unattributed GPU-hours.",
                    f"Historical GPU counts recovered for {f.get('historical_count_recovery',{}).get('recovered_rentals',0)} rentals from exact offer/machine evidence. "
                    'Approximate rental boundaries and unavailable per-group idle attribution prevent a full accepted/GPU-hour ratio.']
    cluster_fleet_path = output / CLUSTER_FLEET_FILE
    if cluster_fleet_path.exists():
        f = json.loads(cluster_fleet_path.read_text())
        if f.get('campaign_id') == 'campaign_20261110':
            lines += ['', 'Campaign-wide Slurm allocation accounting: ' + chicago(datetime.fromtimestamp(f['observed_at_unix'], timezone.utc).isoformat()),
                f"Verified allocation GPU-hours: {f['verified_allocated_gpu_seconds']/3600:.3f}; "
                f"{len(f.get('records', []))} allocations; {len(f.get('ambiguous_campaign_named_allocations', []))} unresolved allocation identities.",
                f.get('scope', ''), f.get('warning', '')]
    lines += ['Vast observation: ' + chicago(live.get('vast_observed_at')),
        f"Vast: {vast.get('running_gpus', 'unavailable')} provider GPUs running; "
        f"{vast.get('worker_ready_gpus', 'unavailable')} ready/running workers; "
        f"{vast.get('worker_booting_gpus', 'unavailable')} booting.",
        f"Vast estimated spend: {dollars(vast.get('spend_estimate_usd'))}; "
        f"reserved compute: {dollars(vast.get('future_compute_commitment_usd'))}; "
        f"transfer/storage reserve: {dollars(vast.get('transfer_reserve_usd'))}; cap $3,000.",
        '', 'Deadline: November 12 at 13:00 Chicago (19:00 UTC); no automatic shutdown.',
        'See the dashboard for measured per-task accepted throughput and current deficits.']
    mirror = json.loads((output / 'sync_status.json').read_text())
    if mirror.get('ok') is False:
        lines += ['Latest mirror FAILED: ' + mirror.get('error', 'unknown error')]
    if live.get('cluster_probe_error'):
        lines += ['Latest cluster probe FAILED: ' + str(live['cluster_probe_error'])]
    for heading, key in [('Completed / verified', 'completed'), ('In progress', 'in_progress'),
                         ('Parked / questions for your return', 'parked')]:
        lines += ['', heading + ':'] + ['- ' + item for item in notes.get(key, [])]
    lines += ['', 'Operator notes last updated: ' + notes.get('updated_at', 'unavailable'),
        'The evidence timestamps above distinguish new observations from last-known state.',
        'Existing LIVE.md and live.json remain owned by the campaign monitor.']
    path = output.parent / 'RETURN_REPORT.txt'
    temporary = path.with_suffix('.tmp')
    temporary.write_text('\n'.join(lines) + '\n')
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--ssh-hostname')
    parser.add_argument('--ssh-hostkey-alias')
    args = parser.parse_args()
    if bool(args.ssh_hostname) != bool(args.ssh_hostkey_alias):
        parser.error('An explicit SSH hostname requires its existing trusted host-key alias')
    ssh_options=['-o','HostName='+args.ssh_hostname,'-o','HostKeyAlias='+args.ssh_hostkey_alias] if args.ssh_hostname else []
    args.output.mkdir(parents=True, exist_ok=True)
    while True:
        status = {'checked_at': datetime.now(timezone.utc).isoformat(), 'campaign_id': 'campaign_20261110'}
        try:
            status.update(ok=True, source_observed_at=sync(args.output,ssh_options))
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            status.update(ok=False, error=type(exc).__name__ + ': ' + str(exc), previous_evidence_preserved=True)
        atomic(args.output / 'sync_status.json', status)
        try:
            return_report(args.output)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(json.dumps({'report_error': type(exc).__name__ + ': ' + str(exc)}), flush=True)
        print(json.dumps(status), flush=True)
        if args.once:
            break
        time.sleep(30)


if __name__ == '__main__':
    main()
