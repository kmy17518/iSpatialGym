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


def atomic(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2) + '\n')
    temp.replace(path)


def sync(output):
    remote = "import json,pathlib; p=pathlib.Path(%r); names=%r; data={n:json.loads((p/n).read_text()) for n in names if (p/n).exists()}; report=p/'GOAL_GROUP_YIELD.md'; data['goal_yield_report']=report.read_text() if report.exists() else None; fleet=p.parent/'accounting'/%r; data[%r]=json.loads(fleet.read_text()) if fleet.exists() else {}; print(json.dumps(data))" % (REMOTE, FILES, FLEET_FILE, FLEET_FILE)
    # The fixed remote program reads only the allowlisted campaign audit files.
    command = "python3 - <<'PY'\n" + remote + "\nPY"
    result = subprocess.run(['ssh', '-oBatchMode=yes', '-oConnectTimeout=15',
        'minyeong@svl11', command], text=True, capture_output=True, timeout=45, check=True)
    bundle = json.loads(result.stdout)
    evidence = bundle.get('dashboard_evidence.json')
    if not evidence or evidence.get('campaign_id') != 'campaign_20261110':
        raise ValueError('Current campaign allocation evidence is unavailable or mismatched')
    for name in FILES:
        if name in bundle:
            atomic(output / name, bundle[name])
    fleet = bundle.get(FLEET_FILE) or {}
    if fleet.get('campaign_id') == 'campaign_20261110':
        atomic(output / FLEET_FILE, fleet)
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
        lines.append(f'{title}: {value:,} / {target:,}; {max(0, target-value):,} remaining')
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
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    while True:
        status = {'checked_at': datetime.now(timezone.utc).isoformat(), 'campaign_id': 'campaign_20261110'}
        try:
            status.update(ok=True, source_observed_at=sync(args.output))
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
