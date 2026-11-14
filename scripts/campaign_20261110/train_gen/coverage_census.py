#!/usr/bin/env python3
"""Coverage census of the non-canonical variant author: does `author_variant.py author --dry-run` succeed on every gate-valid
row of the RUN classes, and if not, why (reason class of the first refusal).

    PYTHONPATH=$SNAP $PY coverage_census.py rows --out ROWS.jsonl            # RUN classes x gate `valid` rows
    PYTHONPATH=$SNAP $PY coverage_census.py run ROWS.jsonl --out PART.jsonl --workers 16 [--label r1]
    PYTHONPATH=$SNAP $PY coverage_census.py merge PART.jsonl --out coverage_r1.json

`run` imports the author once, caches its static inputs (validity manifest, registry, selection, scene jsons, probes), runs the
gate `check` in process (same `decide` as the subprocess) and forks workers. Each dry-run source goes under
VA/dryrun_cov/<label>/ and is removed after the row is recorded. Canonical rows (`train:canonical_*`) are authored by
author_canonical.py and are recorded as covered without a dry run.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import multiprocessing as mp
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import author_variant as AV  # noqa: E402

REASONS = [  # (reason class, message fragment) - first match wins
    ('library_identity', 'different library identities'),
    ('same_model_pair', 'share one model'),
    ('category_change', 'needs IDDL render'),
    ('category_change', 'category change'),
    ('family_unsupported', 'unsupported family'),
    ('family_unsupported', 'profile_for_config rejects'),
    ('multi_null_room', 'rooms must be one named room'),
    ('dict_joint', 'dict joint'),
    ('nonstandard_role', 'unsupported role'),
    ('nonstandard_role', 'more than one support role'),
    ('fixture_replacement', 'replacement fixture'),
    ('missing_height_probe', 'no height probe scene'),
    ('xy_support_scale', 'height scale changes x/y'),
    ('missing_prepared_artifact', 'registry configuration missing'),
    ('missing_prepared_artifact', 'no prepared grasp library'),
    ('missing_prepared_artifact', 'prepared map heights'),
    ('missing_prepared_artifact', 'has no lookup key'),
    ('support_unresolved', 'support scene instance unresolved'),
    ('support_unresolved', 'top height unknown'),
    ('support_unresolved', 'support instance model differs'),
    ('native_choice', 'native choice'),
    ('native_choice', 'native '),
    ('gate_refused', 'gate check rc='),
    ('asset_mismatch', 'asset bytes of'),
    ('not_ready', 'not ready'),
    ('no_default_scale', 'no ready default scale'),
    ('model_keyed_transfer', 'cannot be transferred'),
    ('render', 'IDDL render failed'),
    ('floor_not_unique', 'not unique'),
    ('wall_support', 'wall support'),
]


def classify(msg):
    for name, frag in REASONS:
        if frag in msg:
            return name
    return 'other'


def run_classes():
    return [r['class_id'] for r in csv.DictReader(AV.PROPOSAL.open()) if r['status'] == 'RUN']


def cmd_rows(out):
    gate = AV.gate_module()
    m = gate.load_manifest()
    classes = run_classes()
    n = 0
    with open(out, 'w') as f:
        for cid in classes:
            for r in m['rows']:
                if r['class_id'] == cid and r['status'] == 'valid':
                    f.write(json.dumps({k: r[k] for k in ('configuration_id', 'class_id', 'template_key', 'scene', 'rooms')}) + '\n')
                    n += 1
    print(json.dumps({'classes': len(classes), 'rows': n, 'out': str(out)}))


# ------------------------------------------------------------------------------------------------ cached author inputs

def install_caches():
    """Static inputs are read once per process tree (fork shares them); authored files (dry dirs) are never cached."""
    cache = {}
    orig_read = AV.read
    dry = str(AV.VA)

    def read(p):
        sp = str(p)
        if sp.startswith(dry) or sp.startswith(str(AV.T)):
            return orig_read(p)
        if sp not in cache:
            cache[sp] = orig_read(p)
        return cache[sp]
    AV.read = read

    gate = AV.gate_module()
    manifest = gate.load_manifest()
    gate.load_manifest = lambda *a, **k: manifest
    AV.gate_module = lambda: gate

    reg_rows = {r['configuration_id']: r for r in orig_read(AV.REGISTRY / 'configurations.json')['rows']}
    arts_all = {r['artifact_id']: r for r in orig_read(AV.REGISTRY / 'artifacts.json')['rows']}

    def registry_configuration(cid):
        cfg = reg_rows.get(cid)
        AV.require(cfg is not None and cfg.get('all_mandatory_consumers_resolve'), 'registry configuration missing or unresolved')
        ids = {a for us in cfg['units'].values() for u in us for a in u['artifacts']}
        return cfg, {a: arts_all[a] for a in ids if a in arts_all}
    AV.registry_configuration = registry_configuration

    class _P:   # gate `check` in process: the same decide() the subprocess runs
        def __init__(self, rc, out):
            self.returncode, self.stdout, self.stderr = rc, out, ''

    real_run = AV.subprocess.run

    def run(cmd, *a, **k):
        if len(cmd) >= 3 and cmd[1] == str(AV.GATE) and cmd[2] == 'check':
            cfgp = cmd[3]
            binding = cmd[cmd.index('--binding') + 1] if '--binding' in cmd else None
            giveups = cmd[cmd.index('--giveups') + 1] if '--giveups' in cmd else None
            r = gate.decide(cfgp, manifest, binding, gate.read_giveups(giveups))
            return _P(0 if r['decision'] == 'run' else 3, json.dumps(r))
        return real_run(cmd, *a, **k)
    AV.subprocess.run = run
    # warm heavy imports before the fork
    import numpy  # noqa: F401
    from ispatialgym.sampling.v2.problem import GroundedProblem, iter_template_choices  # noqa: F401
    from ispatialgym.sampling import campaign_runtime, campaign_augmentation, campaign_class_routes  # noqa: F401
    from scripts.campaign_20261110.readiness_catalog import ReadinessCatalog  # noqa: F401


def one(row):
    cid = row['configuration_id']
    base = {k: row[k] for k in ('class_id', 'configuration_id', 'template_key')}
    if cid.startswith('train:canonical_'):
        return dict(base, ok=True, reason='canonical', message='authored by author_canonical.py', seconds=0.0)
    t0 = time.time()
    try:
        s = AV.author(cid, 0, 2, 2, dry=True)
        shutil.rmtree(s['job_dir'], ignore_errors=True)
        extra = {'render': s.get('render')} if s.get('render') else {}
        return dict(base, ok=True, reason=None, message='', seconds=round(time.time() - t0, 2), **extra)
    except BaseException as exc:  # noqa: BLE001  (SystemExit / ContractError / anything)
        msg = f'{type(exc).__name__}: {exc}'
        tb = traceback.format_exc(limit=-3) if not isinstance(exc, (SystemExit,)) and 'ContractError' not in type(exc).__name__ else ''
        for d in AV.DRY_ROOT['path'].glob(f'va_0_*_{os.getpid()}_*'):
            shutil.rmtree(d, ignore_errors=True)
        return dict(base, ok=False, reason=classify(msg), message=msg[:600], trace=tb[-1500:], seconds=round(time.time() - t0, 2))


def cmd_run(rows_path, out, workers, label, limit=None, classes=None):
    rows = [json.loads(l) for l in Path(rows_path).read_text().splitlines() if l.strip()]
    if classes:
        rows = [r for r in rows if r['class_id'] in classes]
    if limit:
        rows = rows[:limit]
    AV.DRY_ROOT['path'] = AV.VA / 'dryrun_cov' / label
    AV.DRY_ROOT['path'].mkdir(parents=True, exist_ok=True)
    install_caches()
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    t0 = time.time()
    done = 0
    ctx = mp.get_context('fork')
    with open(out, 'w') as f, ctx.Pool(workers, maxtasksperchild=200) as pool:
        for res in pool.imap_unordered(one, rows, chunksize=4):
            f.write(json.dumps(res) + '\n')
            f.flush()
            done += 1
            if done % 200 == 0:
                print(f'{done}/{len(rows)} {time.time() - t0:.0f}s', flush=True)
    print(json.dumps({'rows': len(rows), 'seconds': round(time.time() - t0)}))


def cmd_merge(parts, out, label):
    res = {}
    for p in parts:
        for l in Path(p).read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                res[r['configuration_id']] = r
    rows = sorted(res.values(), key=lambda r: (r['class_id'], r['configuration_id']))
    per = collections.defaultdict(lambda: {'rows_ok': 0, 'rows_total': 0, 'canonical_rows': 0, 'reasons': collections.Counter(),
                                           'templates': collections.defaultdict(list)})
    for r in rows:
        c = per[r['class_id']]
        c['rows_total'] += 1
        c['rows_ok'] += bool(r['ok'])
        c['canonical_rows'] += r['reason'] == 'canonical'
        if not r['ok']:
            c['reasons'][r['reason']] += 1
        c['templates'][r['template_key']].append(bool(r['ok']))
    classes = {}
    for cid, c in sorted(per.items()):
        t = c['templates']
        classes[cid] = {'rows_ok': c['rows_ok'], 'rows_total': c['rows_total'], 'canonical_rows': c['canonical_rows'],
                        'templates_ok': sum(any(v) for v in t.values()), 'templates_all_rows_ok': sum(all(v) for v in t.values()),
                        'templates_total': len(t), 'refusals': dict(c['reasons'].most_common())}
    reasons = collections.Counter(r['reason'] for r in rows if not r['ok'])
    examples = {}
    for r in rows:
        if not r['ok'] and r['reason'] not in examples:
            examples[r['reason']] = {'configuration_id': r['configuration_id'], 'message': r['message']}
    msgs = collections.Counter((r['reason'], r['message'].split(':', 2)[-1][:120].strip()) for r in rows if not r['ok'])
    totals = {'rows_ok': sum(c['rows_ok'] for c in classes.values()), 'rows_total': len(rows),
              'canonical_rows': sum(c['canonical_rows'] for c in classes.values()),
              'templates_ok': sum(c['templates_ok'] for c in classes.values()),
              'templates_total': sum(c['templates_total'] for c in classes.values()),
              'classes_with_any_ok': sum(c['rows_ok'] > 0 for c in classes.values()), 'classes': len(classes)}
    doc = {'schema': 'isg-train-variant-coverage-v1', 'label': label, 'built_at': time.time(),
           'author': AV.ref(Path(AV.__file__).resolve()), 'snapshot': str(AV.SNAP),
           'validity_manifest': AV.read(AV.RP / 'validity/LATEST.json'),
           'definitions': {'ok': 'author_variant.author(--dry-run) wrote a source whose gate check exits 0, or a canonical row',
                           'templates_ok': 'template_key with >= 1 ok row', 'templates_all_rows_ok': 'template_key whose rows are all ok',
                           'reason': 'class of the first refusal (author checks run in order)'},
           'totals': totals, 'refusal_reasons': dict(reasons.most_common()), 'reason_examples': examples,
           'top_messages': [{'reason': k[0], 'message': k[1], 'rows': v} for k, v in msgs.most_common(40)],
           'classes': classes, 'rows': [{k: r.get(k) for k in ('class_id', 'configuration_id', 'template_key', 'ok', 'reason', 'message')}
                                         for r in rows]}
    AV.write_json_atomic(out, doc)
    print(json.dumps({'totals': totals, 'refusal_reasons': doc['refusal_reasons']}, indent=1))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    a = sub.add_parser('rows'); a.add_argument('--out', required=True)
    b = sub.add_parser('run'); b.add_argument('rows'); b.add_argument('--out', required=True)
    b.add_argument('--workers', type=int, default=16); b.add_argument('--label', default='r1'); b.add_argument('--limit', type=int)
    b.add_argument('--classes', default='')
    c = sub.add_parser('merge'); c.add_argument('parts', nargs='+'); c.add_argument('--out', required=True); c.add_argument('--label', default='r1')
    args = ap.parse_args()
    if args.cmd == 'rows':
        cmd_rows(args.out)
    elif args.cmd == 'run':
        cmd_run(args.rows, args.out, args.workers, args.label, args.limit, [x for x in args.classes.split(',') if x])
    else:
        cmd_merge(args.parts, args.out, args.label)


if __name__ == '__main__':
    main()
