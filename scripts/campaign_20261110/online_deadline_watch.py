"""Hard backstop for only the sampler process owned by this job.

Normal deadlines close a group in-process. If a native call fails to return,
terminate that worker, preserving its durable groups and caches. A process with
unproven ancestry or a reused PID is never signalled.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import signal
import time


def process(pid):
    text = Path('/proc', str(pid), 'stat').read_text()
    fields = text[text.rfind(')') + 2:].split()
    return {'pid': int(pid), 'parent': int(fields[1]), 'start_ticks': int(fields[19])}


def descendant(pid, ancestor):
    seen = set()
    while pid > 1 and pid not in seen:
        if pid == ancestor: return True
        seen.add(pid)
        try: pid = process(pid)['parent']
        except (OSError, ValueError): return False
    return False


def expired(state, now):
    group = state.get('current')
    operation = state.get('sampling_operation')
    if not group or not operation or not operation.get('active'): return None
    deadline = min(float(group['deadline']), float(operation['member_deadline']))
    return {**operation, 'group_id': group['id'], 'deadline': deadline} if now > deadline + 2 else None


def watch(settings_path, ancestor, poll_s=1., term_grace_s=5.):
    settings = json.loads(Path(settings_path).read_text());root = Path(settings['state_dir'])
    known = process(ancestor)
    while True:
        try:
            if process(ancestor)['start_ticks'] != known['start_ticks']: return
        except OSError: return
        try: state = json.loads((root / 'state.json').read_text())
        except (OSError, ValueError): time.sleep(poll_s); continue
        late = expired(state, time.time())
        if late:
            pid = int(late['pid'])
            try: actual = process(pid)
            except OSError: return
            if not descendant(pid, ancestor) or actual['start_ticks'] != late['start_ticks']:
                raise RuntimeError('deadline target is not this job owned sampler')
            row = {**late, 'observed_at': time.time(), 'action': 'SIGTERM', 'owner_ancestor': ancestor}
            (root / 'hard_deadline.json').write_text(json.dumps(row) + '\n')
            os.kill(pid, signal.SIGTERM)
            until = time.monotonic() + term_grace_s
            while time.monotonic() < until:
                try:
                    if process(pid)['start_ticks'] != actual['start_ticks']: return
                except OSError: return
                time.sleep(.1)
            if process(pid)['start_ticks'] == actual['start_ticks']:
                os.kill(pid, signal.SIGKILL)
                row['action'] = 'SIGTERM_then_SIGKILL';row['kill_at'] = time.time()
                (root / 'hard_deadline.json').write_text(json.dumps(row) + '\n')
            return
        time.sleep(poll_s)


if __name__ == '__main__':
    p=argparse.ArgumentParser();p.add_argument('--settings',required=True);p.add_argument('--ancestor',type=int,required=True)
    args=p.parse_args();watch(args.settings,args.ancestor)
