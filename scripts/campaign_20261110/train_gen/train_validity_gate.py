#!/usr/bin/env python3
"""Train-generation gate: run a training sub-template only if preparation made it valid.

A train worker runs one campaign config = one *sub-template* (class + scene + the exact model, asset and applied scale of
every role). Preparation publishes, per sub-template, whether every artifact the training consumer needs (grasp
libraries, reach / interior / attachment / contact maps, articulation motions and prefixes, geometry) resolves to a
positive result through the consumer's own lookups. This gate is the only place the train driver asks that question.

    python3 train_validity_gate.py check <campaign_config.json> [--binding preparation_binding.json] [--giveups FILE]
        -> prints {"decision": "run"|"skip", ...}; exit 0 = run, 3 = skip, 2 = error
    python3 train_validity_gate.py valid --class CLASS_ID [--include-consumer-fix]   # authoring list (JSON lines)
    python3 train_validity_gate.py explain <campaign_config.json>                       # nearest selection rows, field by field
    python3 train_validity_gate.py giveup <campaign_config.json> --giveups FILE --job JOB --seconds S [--reason R]

Rules (fail closed):
  * the manifest is the one LATEST.json names, sha256-checked; it records the consumer snapshot it was validated against,
    and `check` refuses a config whose worker release differs from it unless --release-ok is given;
  * `run` only for status `valid` (`valid_if_consumer_fixed` stays `skip` until preparation re-validates against the fixed
    consumer: then those rows simply become `valid`);
  * a sub-template not in the training selection is `skip` (unknown), never `run`;
  * a sub-template with a recorded give-up (no accepted member within its give-up time) is `skip` until a give-up is
    cleared by the owner; give-ups are per sub-template, so sibling scales of the same template keep running.
Plain python3, no third-party imports; the builder (build_train_validity.py) imports `identity_*` from here.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

RP = '/vision/u/minyeong/projects/isg_home/campaign_20261110/training_preparation/preparation_vast_20261114T034355Z'
LATEST = os.environ.get('ISG_TRAIN_VALIDITY_LATEST', RP + '/validity/LATEST.json')
SCHEMA = 'isg-train-subtemplate-validity-v1'
NON_ROLE_CATEGORIES = {'floors', 'walls', 'ceilings'}
ROLE_ALIASES = {'distractors': 'distractor'}
RUNNABLE = {'valid'}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _scale(values) -> tuple:
    vals = [float(v) for v in (values or [])]
    if len(vals) == 1:
        vals = vals * 3
    return tuple(round(v, 4) for v in vals)


def identity_items(roles) -> list:
    """roles: iterable of dicts with role, category, model, asset_sha256, scale."""
    out = []
    for r in roles:
        if not r.get('role') or r.get('category') in NON_ROLE_CATEGORIES or r.get('semantic_kind') == 'floor':
            continue
        out.append((ROLE_ALIASES.get(r['role'], r['role']), r['category'], r['model'], r.get('asset_sha256') or '',
                    _scale(r['scale']), r.get('joint') or ''))
    return sorted(out)


def identity_sha(class_id: str, scene: str, roles, room=None) -> str:
    return _sha(json.dumps([class_id, scene, room or '', identity_items(roles)], separators=(',', ':')).encode())


def room_of(rooms):
    """The single task room (a campaign template has exactly one); multi-room selections keep their sorted list."""
    if isinstance(rooms, str):
        return rooms
    rooms = sorted(rooms or [])
    return rooms[0] if len(rooms) == 1 else ','.join(rooms)


def roles_from_selection(row) -> list:
    out = []
    for o in row.get('roles') or []:
        mk = o.get('model_key') or f"{o.get('category')}/{o.get('model')}"
        cat, model = mk.split('/', 1)
        out.append({'role': o.get('role'), 'category': o.get('category') or cat, 'model': o.get('model') or model,
                    'asset_sha256': o.get('asset_sha256'), 'scale': o.get('applied_scale'),
                    'semantic_kind': o.get('semantic_kind'), 'joint': o.get('joint')})
    return out


def roles_from_campaign_config(cfg) -> tuple:
    t = cfg['template']
    scales = (cfg.get('perturbations') or {}).get('scales') or {}
    roles = []
    for o in t['objects']:
        st = scales.get(o.get('instance')) or {}
        roles.append({'role': o.get('role'), 'category': o['category'], 'model': o['model'],
                      'asset_sha256': o.get('asset_sha256'), 'scale': st.get('applied') or o.get('default_scale'),
                      'joint': o.get('joint')})
    rooms = t.get('rooms') or ([t['room']] if t.get('room') else [])
    return t['class_id'], t['scene'], roles, room_of(rooms)


def load_manifest(latest=LATEST):
    ptr = json.loads(Path(latest).read_text())
    path = Path(ptr['path'])
    raw = path.read_bytes()
    if _sha(raw) != ptr['sha256']:
        raise SystemExit(f'validity manifest {path} changed since LATEST was written')
    m = json.loads(raw)
    if m.get('schema') != SCHEMA:
        raise SystemExit(f'unknown validity schema {m.get("schema")}')
    m['_by_id'] = {r['configuration_id']: r for r in m['rows']}
    m['_by_identity'] = {}
    for r in m['rows']:
        m['_by_identity'].setdefault(r['identity_sha256'], []).append(r)
    m['_pointer'] = ptr
    return m


def read_giveups(path):
    out = {}
    if path and Path(path).exists():
        for line in Path(path).read_text().splitlines():
            try:
                g = json.loads(line)
            except ValueError:
                continue
            if g.get('cleared'):
                out.pop(g.get('identity_sha256'), None)
            else:
                out[g.get('identity_sha256')] = g
    return out


def decide(config_path, manifest, binding_path=None, giveups=None, release_ok=False):
    cfg = json.loads(Path(config_path).read_text())
    class_id, scene, roles, room = roles_from_campaign_config(cfg)
    ident = identity_sha(class_id, scene, roles, room)
    res = {'config': str(config_path), 'class_id': class_id, 'identity_sha256': ident,
           'manifest': manifest['_pointer']['path'], 'manifest_sha256': manifest['_pointer']['sha256'],
           'consumer': manifest.get('consumer')}
    if cfg.get('split') != 'train':
        return dict(res, decision='skip', reason=f"campaign split {cfg.get('split')!r} is not 'train'")
    rows = manifest['_by_identity'].get(ident, [])
    if binding_path and Path(binding_path).exists():
        cid = json.loads(Path(binding_path).read_text()).get('configuration_id')
        row = manifest['_by_id'].get(cid)
        if row is None:
            return dict(res, decision='skip', reason=f'bound configuration {cid} is not in the training selection')
        if row['identity_sha256'] != ident:
            return dict(res, decision='skip', configuration_id=cid,
                        reason='campaign config does not match its bound preparation configuration (models/assets/scales differ)')
        rows = [row]
    if not rows:
        return dict(res, decision='skip', reason='not in the training selection (unknown sub-template); see `explain`')
    want = (manifest.get('consumer') or {}).get('release')
    have = (cfg.get('validation_key') or {}).get('code_sha256')
    if want and not release_ok and have and want.get('code_sha256') and have != want['code_sha256']:
        return dict(res, decision='skip', reason='worker release differs from the consumer this manifest was validated against')
    if len({r['status'] for r in rows}) > 1:
        return dict(res, decision='skip', candidates=[r['configuration_id'] for r in rows],
                    reason='ambiguous: several selection configurations share this identity with different validity; '
                           'write preparation_binding.json with the configuration_id this source was authored from')
    best = rows[0]
    res.update(configuration_id=best['configuration_id'], status=best['status'], template_key=best['template_key'],
               variant_id=best.get('variant_id'), blockers=best.get('blockers', []))
    g = (giveups or {}).get(ident)
    if g:
        return dict(res, decision='skip', reason=f"sub-template gave up: {g.get('reason') or 'no accepted member'} (job {g.get('job')})")
    if best['status'] in RUNNABLE:
        return dict(res, decision='run', reason='every required preparation lookup resolves')
    return dict(res, decision='skip', reason=best.get('reason') or best['status'])


def explain(config_path, manifest):
    cfg = json.loads(Path(config_path).read_text())
    class_id, scene, roles, room = roles_from_campaign_config(cfg)
    mine = set(identity_items(roles))
    cands = [r for r in manifest['rows'] if r['class_id'] == class_id]
    scored = []
    for r in cands:
        theirs = set(identity_items(r['identity_roles']))
        scored.append((len(mine ^ theirs) + (0 if r['scene'] == scene else 5) + (0 if room_of(r['rooms']) == room else 2), r, theirs))
    scored.sort(key=lambda x: x[0])
    out = {'class_id': class_id, 'scene': scene, 'room': room, 'config_items': sorted(mine), 'nearest': []}
    for d, r, theirs in scored[:3]:
        out['nearest'].append({'configuration_id': r['configuration_id'], 'scene': r['scene'], 'room': room_of(r['rooms']),
                               'status': r['status'],
                               'only_in_config': sorted(mine - theirs), 'only_in_selection': sorted(theirs - mine)})
    return out


def record_giveup(config_path, path, job, seconds, reason):
    cfg = json.loads(Path(config_path).read_text())
    class_id, scene, roles, room = roles_from_campaign_config(cfg)
    rec = {'identity_sha256': identity_sha(class_id, scene, roles, room), 'class_id': class_id, 'config': str(config_path),
           'job': job, 'seconds': seconds, 'reason': reason or 'no accepted member within sub-template give-up time',
           'at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--latest', default=LATEST)
    sub = ap.add_subparsers(dest='cmd', required=True)
    c = sub.add_parser('check'); c.add_argument('config'); c.add_argument('--binding'); c.add_argument('--giveups')
    c.add_argument('--release-ok', action='store_true')
    v = sub.add_parser('valid'); v.add_argument('--class', dest='class_id', required=True)
    v.add_argument('--include-consumer-fix', action='store_true')
    e = sub.add_parser('explain'); e.add_argument('config')
    g = sub.add_parser('giveup'); g.add_argument('config'); g.add_argument('--giveups', required=True)
    g.add_argument('--job', required=True); g.add_argument('--seconds', type=float, required=True); g.add_argument('--reason')
    a = ap.parse_args()
    if a.cmd == 'giveup':
        print(json.dumps(record_giveup(a.config, a.giveups, a.job, a.seconds, a.reason)))
        return 0
    m = load_manifest(a.latest)
    if a.cmd == 'check':
        r = decide(a.config, m, a.binding, read_giveups(a.giveups), a.release_ok)
        print(json.dumps(r))
        return 0 if r['decision'] == 'run' else 3
    if a.cmd == 'valid':
        ok = RUNNABLE | ({'valid_if_consumer_fixed'} if a.include_consumer_fix else set())
        for r in m['rows']:
            if r['class_id'] == a.class_id and r['status'] in ok:
                print(json.dumps({k: r[k] for k in ('configuration_id', 'status', 'variant_id', 'template_key', 'scene',
                                                    'rooms', 'identity_sha256', 'roles')}))
        return 0
    if a.cmd == 'explain':
        print(json.dumps(explain(a.config, m), indent=1))
        return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({'decision': 'error', 'reason': repr(exc)}))
        sys.exit(2)
