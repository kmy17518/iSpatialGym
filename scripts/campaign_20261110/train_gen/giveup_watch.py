#!/usr/bin/env python3
"""Sub-template give-up for one running train worker (handoff section 5).

  python3 giveup_watch.py --source <job_dir> --pgid <process group of run.sh> [--poll 20]

Rule: if no member has been PUBLISHED within subtemplate_giveup_s of the first group start, stop the worker. If members
were published but are in flight at that time, wait for their demos (bounded by demo_safety_cap_s); give up only if none
of them succeeds and nothing new was published. A worker with any accepted member is never given up. The give-up is
recorded through the preparation gate (`train_validity_gate.py giveup`), then the process group gets SIGINT, then SIGKILL.
Exit codes: 0 worker ended normally, 10 gave up, 11 hard time limit (TIME_LIMIT_S + 300 s) reached and killed.
"""
import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path

GATE = '/vision/u/minyeong/projects/isg_home/campaign_20261110/training_preparation/preparation_vast_20261114T034355Z/validity/code/train_validity_gate.py'
GIVEUPS = '/vision/u/minyeong/projects/isg_home/campaign_20261110/training_generation/train_online_r1/subtemplate_giveups.jsonl'


def alive(pgid):
    try:
        os.killpg(pgid, 0); return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def members(state):
    groups = list(state.get('groups', [])) + list(state.get('open', [])) + ([state['current']] if state.get('current') else [])
    ms = [m for g in groups for m in g.get('members', [])]
    starts = [g['started_at'] for g in groups if g.get('started_at') is not None]
    return ms, (min(starts) if starts else None)


def stop(pgid, grace=60):
    for sig, wait in ((signal.SIGINT, grace), (signal.SIGTERM, 30), (signal.SIGKILL, 0)):
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        t = time.time()
        while time.time() - t < wait and alive(pgid):
            time.sleep(1)
        if not alive(pgid):
            return


def decide(state, now, giveup_s, cap_s):
    """-> ('keep'|'giveup', reason). Pure function (unit-tested)."""
    ms, first = members(state)
    if first is None or now - first < giveup_s:
        return 'keep', 'before give-up time'
    if any(m['status'] == 'accepted' for m in ms):
        return 'keep', 'has accepted member'
    published = [m for m in ms if m.get('published_at')]
    if not published:
        return 'giveup', f'no member published within {giveup_s:.0f} s'
    pending = [m for m in published if m['status'] == 'pending']
    if pending:
        if all(now - m['published_at'] > cap_s + 120 for m in pending):
            return 'giveup', 'in-flight demos exceeded the safety cap without a result'
        return 'keep', f'{len(pending)} published member(s) in flight'
    late = [m for m in published if m['published_at'] - first > giveup_s]
    if late:
        # something new was published after the give-up time and has resolved without success: re-arm from it
        if now - max(m['published_at'] for m in late) < giveup_s:
            return 'keep', 're-armed by a late publication'
    return 'giveup', f'{len(published)} published member(s) resolved, none accepted'


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--source', required=True); ap.add_argument('--pgid', type=int, required=True)
    ap.add_argument('--poll', type=float, default=20.0); a = ap.parse_args()
    job = Path(a.source); src = json.loads((job / 'source.json').read_text())
    giveup_s = float(src['limits']['subtemplate_giveup_s']); cap_s = float(src['limits']['demo_safety_cap_s'])
    hard = time.time() + float(src['seconds']) + 300
    log = (job / 'giveup_watch.jsonl').open('a')
    while alive(a.pgid):
        now = time.time()
        if now > hard:
            log.write(json.dumps({'at': now, 'action': 'hard_limit_kill'}) + '\n'); log.flush()
            stop(a.pgid); return 11
        try:
            state = json.loads((job / 'online/state.json').read_text())
        except (OSError, ValueError):
            time.sleep(a.poll); continue
        verdict, reason = decide(state, now, giveup_s, cap_s)
        if verdict == 'giveup':
            row = {'at': now, 'action': 'giveup', 'reason': reason, 'job': os.environ.get('SLURM_JOB_ID'), 'key': src['key']}
            log.write(json.dumps(row) + '\n'); log.flush()
            ms, first = members(state)
            p = subprocess.run([sys.executable, GATE, 'giveup', str(job / 'campaign_config.json'), '--giveups', GIVEUPS,
                                '--job', f"{os.environ.get('SLURM_JOB_ID', 'local')}:{src['key']}", '--seconds', str(int(now - first)), '--reason', reason],
                               capture_output=True, text=True)
            (job / 'GIVEUP.json').write_text(json.dumps({**row, 'gate_rc': p.returncode, 'gate_out': p.stdout[-2000:],
                                                         'gate_err': p.stderr[-2000:]}, indent=1))
            stop(a.pgid); return 10
        time.sleep(a.poll)
    return 0


if __name__ == '__main__':
    sys.exit(main())
