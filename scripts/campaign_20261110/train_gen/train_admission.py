#!/usr/bin/env python3
"""TRAIN admission of finalized online goal groups (training generation, handoff 2026-11-14 section 7).

  python train_admission.py admit [--only t2611_10001 ...] [--apply] [--report out.json]

Adapted from the eval admission tool (vast/online_admission_r1/tool/admission_r7.py, read-only reference).
One phase, run as minyeong (train sources are minyeong-owned). Dry run by default; --apply writes:

  <OUT>/receipts/<class>/<group_id>.json   immutable, write-once, content_sha256; an identical rerun is a no-op,
                                           a different content for an existing receipt is a CONFLICT (never overwritten)
  <OUT>/ledger/ledger_<sha256>.json        content-addressed ledger (receipts on disk + this scan's rejections)
  <OUT>/ledger/LATEST.json                 pointer + per_class counts (rewritten only when its bytes change)

Kept from eval (frozen native code imported from the TRAIN release snapshot, never reimplemented):
  release/runtime provenance (source/runtime native commit == release candidate_commit; runtime.json snapshot marker ==
  release snapshot; SDK bindings == the parent eval release's grouping proof; no physics mutations; returncode 0),
  pinned source.json file hashes, state/events/claim consistency, frozen online_groups.verify_result (oracle success,
  non-provisional HDF5, TRO/sidecar hashes, meta, hard QA, saved-start difference from earlier kept members),
  acceptance.report_passed (no goal override, restoration verified), collector provenance, restoration evidence,
  non-empty demo.hdf5 (h5py), contracts.validate_campaign_block, goal_identity.equivalent_goal, saved_member, and for
  Vast output durable_preservation.json with checksums_verified.
Dropped for train: baseline / cross-group goal uniqueness, the eval-member allocation, the per-class cap, the 195-unit
  baseline, workstream == eval_standard, and "result finished before the member deadline".
Train timing (ungated demos, section 3a): a member must be PUBLISHED before its member and group sampling deadline,
  carry demo_deadline == published_at + demo_safety_cap_s and demo_gate 'ungated_train_safety_cap' in its TRO, and its
  demo must finish by demo_deadline. A demo finishing after the group's sampling deadline is accepted and flagged.
Groups with >= 1 eligible member are admitted; every eligible member is allocated to 'train'. A member whose saved start
  equals an earlier kept member of the group (frozen verify_result) is excluded individually.
Never writes outside <OUT>; refuses any path under the eval tree R/vast.
"""
from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

R = Path(os.environ.get('ISG_TRAIN_ADMISSION_R', '/vision/u/minyeong/projects/isg_home/campaign_20261110'))
TG = R / 'training_generation'
SOURCES = TG / 'train_online_r1' / 'sources'
RELEASE = TG / 'releases' / 'release_train_r1.json'
OUT = Path(os.environ.get('ISG_TRAIN_ADMISSION_OUT', str(TG / 'admission_r1')))
EVAL_TREE = R / 'vast'                                   # read-only: parent eval release + proof; never written
SCHEMA = 'isg-train-admission-v1'
DEMO_GATE = 'ungated_train_safety_cap'
VAST_JOB = re.compile(r'^9000\d+$')
TOL = 1e-3

N = SimpleNamespace()   # frozen native modules (load_native)


def load_native(snapshot):
    snapshot = str(Path(snapshot).resolve())
    if getattr(N, 'snapshot', None) == snapshot:
        return N
    sys.path.insert(0, snapshot)
    from ispatialgym.sampling import online_groups as OG          # noqa: E402
    from ispatialgym.sampling import goal_identity as GI          # noqa: E402
    from scripts.campaign_20261110 import acceptance as ACC       # noqa: E402
    from scripts.campaign_20261110 import contracts as C          # noqa: E402
    for m in (OG, GI, ACC, C):
        need(Path(m.__file__).resolve().is_relative_to(Path(snapshot)), 'native module not from release snapshot: ' + m.__file__)
    N.OG, N.GI, N.ACC, N.C, N.snapshot = OG, GI, ACC, C, snapshot
    return N


# ------------------------------------------------------------------ helpers
def need(cond, msg):
    if not cond:
        raise ValueError(msg)
    return True


def read(p):
    return json.loads(Path(p).read_text())


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(obj):
    return hashlib.sha256(canonical(obj).encode()).hexdigest()


def canonical_bytes(obj):
    return (canonical(obj) + '\n').encode()


def ref(p):
    p = Path(p)
    return {'path': str(p), 'sha256': sha256_file(p), 'bytes': p.stat().st_size}


def last_json_line(p):
    for line in reversed(Path(p).read_text().splitlines()):
        line = line.strip()
        if line.startswith('{'):
            return json.loads(line)
    raise ValueError('no JSON line in ' + str(p))


class Checks:
    """Ordered check log (same contract as eval). origin: 'frozen:<module.func>' or 'own'."""
    def __init__(self):
        self.rows = []

    def run(self, name, origin, fn, *, blocker=False):
        try:
            detail = fn()
            ok = detail is not False
            self.rows.append({'check': name, 'origin': origin, 'ok': ok,
                              **({'detail': detail} if detail not in (None, True, False) else {})})
            return ok
        except PermissionError as e:
            self.rows.append({'check': name, 'origin': origin, 'ok': False, 'unreadable': True,
                              'error': 'PermissionError: ' + str(e.filename or e)})
        except FileNotFoundError as e:
            self.rows.append({'check': name, 'origin': origin, 'ok': False, 'missing': True, 'blocker': blocker,
                              'error': 'FileNotFoundError: ' + str(e.filename or e)})
        except Exception as e:
            self.rows.append({'check': name, 'origin': origin, 'ok': False, 'blocker': blocker,
                              'error': type(e).__name__ + ': ' + str(e)[:400]})
        return False

    def failed(self):
        return [r for r in self.rows if not r['ok']]


def guard_write(path):
    path = Path(path).resolve()
    need(path.is_relative_to(OUT.resolve()), 'write outside admission root: %s' % path)
    need(not path.is_relative_to(EVAL_TREE.resolve()), 'refusing to write under the eval tree: %s' % path)
    return path


def write_once_bytes(path, data):
    """Atomic create-only (link a fsynced temp into place); never overwrite. -> created | identical | conflict."""
    path = guard_write(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return 'identical' if path.read_bytes() == data else 'conflict'
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name + '.')
    try:
        with os.fdopen(fd, 'wb') as f:
            f.write(data); f.flush(); os.fsync(f.fileno())
        os.chmod(tmp, 0o444)
        try:
            os.link(tmp, path)
        except FileExistsError:
            return 'identical' if path.read_bytes() == data else 'conflict'
        dfd = os.open(path.parent, os.O_RDONLY); os.fsync(dfd); os.close(dfd)
        return 'created'
    finally:
        os.unlink(tmp)


def write_if_changed(path, data):
    """Mutable pointer file: atomic replace, skipped when the bytes are already identical. -> written | unchanged."""
    path = guard_write(path); path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.read_bytes() == data:
        return 'unchanged'
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.' + path.name + '.')
    with os.fdopen(fd, 'wb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.chmod(tmp, 0o644); os.replace(tmp, path)
    return 'written'


# ----------------------------------------------------------------- release
def eval_parent_physx(rel):
    """SDK bindings sha of the parent eval release's grouping proof (the train release record carries no proof)."""
    parent = rel['parents']['eval_release']
    hits = []
    for p in sorted((EVAL_TREE / 'online_grouped_r1').glob('release_*.json')):
        x = read(p)
        if x.get('candidate_commit') == parent and x.get('status') == 'frozen':
            hits.append((p, x))
    need(hits, 'parent eval release %s not found' % parent)
    p, x = hits[-1]
    gp = x['grouping_proof']
    need(sha256_file(gp['path']) == gp['sha256'], 'parent eval grouping proof changed')
    return {'release': ref(p), 'expected_physx_bindings_sha256': read(gp['path'])['runtime']['expected_physx_bindings_sha256']}


def release_identity(path=RELEASE):
    path = Path(path); rel = read(path); snap = Path(rel['snapshot']).resolve()
    need(rel.get('schema') == 'isg-train-release-v1' and rel.get('status') == 'frozen', 'train release not frozen')
    need(snap.parent == (R / 'code').resolve(), 'release snapshot not under R/code')
    out = {'release': ref(path), 'name': rel['release'], 'candidate_commit': rel['candidate_commit'], 'snapshot': str(snap),
           'frozen_at': rel['frozen_at'], 'code_sha256': rel['validation_key']['code_sha256'],
           'manifest_sha256': sha256_file(snap / '.snapshot_files_sha256.json'),
           'snapshot_commit_marker': ref(snap / '.snapshot_commit'),
           'snapshot_commit_text': (snap / '.snapshot_commit').read_text().strip(),
           'snapshot_b1k_text': (snap / '.snapshot_b1k').read_text().strip(),
           'parent_eval': eval_parent_physx(rel)}
    out['expected_physx_bindings_sha256'] = out['parent_eval']['expected_physx_bindings_sha256']
    need(out['manifest_sha256'] == rel['snapshot_manifest_sha256'], 'snapshot manifest changed')
    need(out['snapshot_commit_text'] == rel['candidate_commit'], 'snapshot commit marker differs')
    return out


def release_brief(rel):
    return {k: rel[k] for k in ('release', 'name', 'candidate_commit', 'snapshot', 'manifest_sha256', 'code_sha256', 'frozen_at')}


# ------------------------------------------------------------ worker level
def native_train_key(cfg):
    """The validation key the native worker stamps on train TROs.

    The job config may carry the eval template (split eval_standard; campaign_runtime._load allows it for split train)
    with the eval template fingerprint; OnlineGroups replaces the template split with the config split, so every train
    TRO/group config carries the fingerprint of that train template (contracts.Template.fingerprint, frozen).
    """
    from dataclasses import replace
    data = dict(cfg['template']); data['objects'] = tuple(N.C.ObjectRef(**o) for o in data['objects'])
    t = N.C.Template(**data)
    need(cfg['validation_key']['template_sha256'] == t.fingerprint, 'config validation key template mismatch')
    need(cfg['split'] == t.split or (t.split == 'eval_standard' and cfg['split'] == 'train'), 'config split mismatch')
    if t.split != cfg['split']:
        t = replace(t, split=cfg['split'])
    return {**cfg['validation_key'], 'template_sha256': t.fingerprint}


def worker_running(job):
    """True while the worker (or its runtime observer) has not finished."""
    if (job / 'online' / '.worker_active').exists():
        return 'online/.worker_active present'
    rt = job / 'runtime.json'
    if not rt.exists():
        return 'no runtime.json'
    st = read(rt)
    if st.get('status') == 'observing' or st.get('finished_at') is None:
        return 'runtime.json status %r (job %s)' % (st.get('status'), (job / 'jobid').read_text().strip() if (job / 'jobid').exists() else '?')
    return None


def check_worker(job, rel):
    ck = Checks(); info = {}; src = {}; settings = {}

    def source():
        s = read(job / 'source.json'); src.update(s); info['source'] = ref(job / 'source.json')
        need(s.get('key') == job.name and Path(s['job_dir']).resolve() == job.resolve(), 'source key/job_dir differs')
        need(s.get('workstream') == 'train', 'workstream not train: %r' % s.get('workstream'))
        need(Path(s['release']['path']).resolve() == Path(rel['release']['path']).resolve()
             and s['release']['sha256'] == rel['release']['sha256'], 'source release ref is not release_train_r1')
        need(s.get('native_commit') == rel['candidate_commit'], 'source native commit differs from release')
        need(Path(s.get('snapshot', '')).resolve() == Path(rel['snapshot']), 'source snapshot differs from release')
        bad = []
        for k in ('settings', 'campaign_config', 'job_env', 'wrapper'):
            p = Path(s[k]['path'])
            if p.resolve().parent != job.resolve() or sha256_file(p) != s[k]['sha256']:
                bad.append(k)
        need(not bad, 'source sha256 refs changed: ' + ','.join(bad))
        return {k: s[k]['sha256'] for k in ('settings', 'campaign_config', 'job_env', 'wrapper')}
    ck.run('release_binding_and_source_refs_sha256', 'own', source)

    def binding():
        pb = src['preparation_binding']
        for k in ('configuration_id', 'template_key', 'variant_id', 'manifest_revision', 'manifest_sha256'):
            need(pb.get(k), 'preparation_binding lacks ' + k)
        need(pb.get('status') == 'valid', 'preparation status %r' % pb.get('status'))
        b = Path(pb['binding']['path'])
        need(b.resolve().parent == job.resolve() and sha256_file(b) == pb['binding']['sha256'], 'preparation_binding.json changed')
        need(read(b).get('configuration_id') == pb['configuration_id'], 'preparation_binding.json configuration_id differs')
        gates = {}
        for name in ('gate_author.txt', 'gate_dispatch.txt'):
            if (job / name).exists():
                g = last_json_line(job / name)
                need(g.get('decision') == 'run' and g.get('configuration_id') == pb['configuration_id'],
                     '%s decision %r / configuration differs' % (name, g.get('decision')))
                gates[name.split('.')[0]] = {'ref': ref(job / name), 'manifest': g.get('manifest'),
                                             'manifest_sha256': g.get('manifest_sha256'), 'status': g.get('status')}
        need('gate_dispatch' in gates, 'no dispatch-time gate record')
        need(gates.get('gate_author', {}).get('manifest_sha256', pb['manifest_sha256']) == pb['manifest_sha256'],
             'author gate manifest differs from source binding')
        info['preparation_binding'] = {**{k: pb.get(k) for k in ('configuration_id', 'template_key', 'variant_id', 'status',
                                                                  'scene', 'rooms', 'identity_sha256', 'roles',
                                                                  'manifest_revision', 'manifest_sha256')},
                                       'binding': pb['binding'], 'gates': gates}
    ck.run('preparation_binding', 'own', binding)

    def config():
        c = read(job / 'campaign_config.json'); info['campaign_config'] = ref(job / 'campaign_config.json')
        need(c.get('split') == 'train', 'campaign_config split is %r' % c.get('split'))
        need(c['template']['class_id'] == src.get('class_id'), 'config template class differs from source')
        need(c['validation_key']['code_sha256'] == rel['code_sha256'], 'config validation_key.code_sha256 is not the release')
        p = c['perturbations']
        pert = N.C.Perturbations({n: N.C.ScaleState(**v) for n, v in p['scales'].items()}, tuple(p['removed_objects']), p['lighting'])
        need(c['validation_key']['scale_sha256'] == pert.scale_fingerprint, 'config validation key scale mismatch')
        info['validation_key'] = c['validation_key']; info['perturbations'] = p
        info['tro_validation_key'] = native_train_key(c)
        return {'tro_template_sha256': info['tro_validation_key']['template_sha256'],
                'config_template_sha256': c['validation_key']['template_sha256']}
    ck.run('campaign_config_train', 'own + frozen:contracts.Template/Perturbations', config)

    def sett():
        s = read(job / 'settings.json'); settings.update(s); info['settings'] = ref(job / 'settings.json')
        need(Path(s['state_dir']).resolve() == (job / 'online').resolve(), 'settings state_dir differs')
        need(Path(s['results']).resolve() == (job / 'results.jsonl').resolve(), 'settings results differs')
        need(int(s.get('minimum_members', 0)) >= 1, 'minimum_members < 1')
        need(s.get('registry') and Path(s['registry']).resolve() == (job / 'registry_private').resolve(),
             'registry is not the private per-worker registry')
        cap = float(s.get('demo_safety_cap_s', math.inf))
        need(math.isfinite(cap) and cap > 0, 'no finite demo_safety_cap_s')
        need(abs(cap - float(src['limits']['demo_safety_cap_s'])) < TOL, 'settings demo cap differs from source limits')
        info['demo_safety_cap_s'] = cap
        return {'worker_id': s['worker_id'], 'minimum_members': s['minimum_members'], 'target_members': s.get('target_members'),
                'group_limit_s': s.get('group_limit_s'), 'member_limit_s': s.get('member_limit_s'), 'demo_safety_cap_s': cap}
    ck.run('settings', 'own', sett)

    def runtime():
        snap = Path(rel['snapshot']); rt = read(job / 'runtime.json')
        gpus = sorted({r.get('name') for r in (rt.get('gpu') or {}).get('rows', [])} - {None})
        info['runtime'] = {'path': str(job / 'runtime.json'), 'sha256': sha256_file(job / 'runtime.json'),
                           'host': rt.get('host'), 'started_at': rt.get('started_at'), 'finished_at': rt.get('finished_at'),
                           'status': rt.get('status'), 'gpu_models': gpus}
        nr = rt.get('native_runtime') or {}
        need(rt.get('schema') == 'isg-external-native-runtime-observation-v1', 'runtime schema')
        need(nr.get('commit') == rel['candidate_commit'], 'runtime native commit differs')
        need(Path(nr.get('path', '')).resolve() == (snap / '.snapshot_commit').resolve()
             and nr.get('sha256') == rel['snapshot_commit_marker']['sha256'], 'runtime snapshot marker differs from release snapshot')
        need(rt.get('status') == 'observed_expected_sdk', 'runtime status %r' % rt.get('status'))
        need(rt.get('expected_bindings_observed') is True and rt.get('unexpected_bindings') == [], 'unexpected SDK bindings')
        need(rt.get('expected_physx_bindings_sha256') == rel['expected_physx_bindings_sha256'], 'SDK bindings sha differs from eval proof')
        need(rt.get('physics_mutations') is False, 'physics mutations observed')
        need(rt.get('native_returncode') == 0, 'native returncode %r' % rt.get('native_returncode'))
        need(rt.get('request_sha256') == src['settings']['sha256'], 'runtime request differs from settings sha')
        need(rt.get('class_id') == src.get('class_id'), 'runtime class differs')
        cmd = rt.get('command') or []
        need(len(cmd) == 3 and Path(cmd[1]).resolve().is_relative_to(snap) and Path(cmd[2]).resolve() == job.resolve(),
             'runtime command not the frozen job entry')
        return {'gpu_models': gpus, 'host': rt.get('host')}
    ck.run('runtime_provenance', 'own', runtime)

    def state():
        st = read(job / 'online/state.json'); info['state'] = ref(job / 'online/state.json')
        need(st.get('schema') == 'isg-online-groups-v1', 'state schema')
        need(st['owner'] == settings.get('worker_id'), 'state owner differs from settings worker_id')
        need(st['settings_sha256'] == sha256_file(job / 'settings.json'), 'state settings sha differs')
        need(st['config_sha256'] == sha256_file(job / 'campaign_config.json'), 'state config sha differs')
        need(not st.get('open') and not st.get('current'), 'worker left open/current groups')
        info['_state'] = st
    ck.run('state_json', 'own', state)

    def events():
        rows = []
        with open(job / 'online/events.jsonl') as f:
            for line in f:
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    pass
        info['events'] = ref(job / 'online/events.jsonl'); info['_events'] = rows
        need(all(r.get('worker') == settings.get('worker_id') for r in rows), 'events from another worker')
    ck.run('events_jsonl', 'own', events)

    def results():
        rows = {}
        with open(job / 'results.jsonl') as f:
            for line in f:
                try:
                    row = json.loads(line); rows[row['instance']] = row   # native collect(): last row wins
                except (ValueError, KeyError):
                    continue
        info['results'] = ref(job / 'results.jsonl'); info['_results'] = rows
    ck.run('results_jsonl', 'own', results)

    vast = {}
    def vast_files():
        b, pres, dur = job / 'backend.json', job / 'preservation.json', job / 'durable_preservation.json'
        vast.update(backend_json=b.exists(), preservation=pres.exists(), durable=dur.exists())
        if b.exists():
            info['backend'] = {k: v for k, v in read(b).items() if k in ('backend', 'instance', 'gpu', 'attempt_id', 'start', 'end')}
        if pres.exists() and dur.exists():
            p, d = read(pres), read(dur)
            vast['durable_ok'] = (d.get('checksums_verified') is True and d['outputs']['sha256'] == sha256_file(pres)
                                  and str(d.get('job_id')) == str(p.get('job_id')) and d['outputs'].get('file_count') == len(p['files']))
            vast['job_id'] = str(p.get('job_id')); vast['files'] = p['files']
            info['preservation'] = ref(pres); info['durable_preservation'] = ref(dur)
        return {k: v for k, v in vast.items() if k != 'files'}
    ck.run('vast_backend_preservation', 'own', vast_files)
    info['_vast'] = vast; info['_src'] = src; info['_settings'] = settings
    info['slurm_job_id'] = (job / 'jobid').read_text().strip() if (job / 'jobid').exists() else None
    return ck, info


# ------------------------------------------------------------ member level
def hdf5_nonempty(path):
    import h5py
    n = [0, 0]
    with h5py.File(path, 'r', locking=False) as f:
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                n[0] += 1; n[1] += int(obj.size)
        f.visititems(visit)
        keys = list(f.keys())
    need(keys and n[0] > 0 and n[1] > 0, 'empty HDF5')
    return {'top_keys': len(keys), 'datasets': n[0], 'elements': n[1]}


def ungated_timing(member, group, row, online_group, cap):
    """Train replacement for eval's 'result finished before the member deadline' (pure; unit-tested).

    Requires publication before the member AND group sampling deadline, demo_deadline == published_at + cap (state and
    TRO agree, gate ungated_train_safety_cap), and the demo finished by demo_deadline. Returns timing detail.
    """
    need('demo_deadline' in member and 'published_at' in member, 'member has no demo_deadline/published_at (not ungated)')
    pub, dl, gdl = float(member['published_at']), float(member['deadline']), float(group['deadline'])
    need(pub <= dl, 'member published after its member sampling deadline')
    need(pub <= gdl, 'member published after its group sampling deadline')
    need(abs(float(member['demo_deadline']) - (pub + cap)) < TOL, 'demo_deadline != published_at + demo_safety_cap_s')
    og = online_group or {}
    need(og.get('demo_gate') == DEMO_GATE, 'TRO demo_gate %r' % og.get('demo_gate'))
    need(abs(float(og.get('demo_deadline', -1)) - float(member['demo_deadline'])) < 1e-6, 'TRO demo_deadline differs from state')
    need(abs(float(og.get('group_deadline', -1)) - gdl) < 1e-6, 'TRO group_deadline differs from state')
    need(abs(float(og.get('member_deadline', -1)) - dl) < 1e-6, 'TRO member_deadline differs from state')
    fin = row.get('finished_at_unix')
    need(fin is not None and float(fin) <= float(member['demo_deadline']), 'demo finished after demo_deadline (safety cap)')
    need(float(fin) >= pub, 'demo finished before publication')
    start = float(group['started_at'])
    return {'published_at': pub, 'finished_at_unix': float(fin), 'member_deadline': dl, 'group_sampling_deadline': gdl,
            'demo_deadline': float(member['demo_deadline']), 'seconds_to_publish': pub - start,
            'seconds_to_finish': float(fin) - start, 'demo_seconds': float(fin) - pub,
            'demo_finished_after_group_sampling_deadline': float(fin) > gdl}


def verify_member(job, member, row, kept, group, class_id, rel, cfg_key, cfg_pert, owner_worker, cap, vast):
    ck = Checks(); iid = member['instance_id']; art = {}; state = {}
    ck.run('result_row_present', 'own', lambda: need(row is not None, 'no results.jsonl row'))
    if row is None:
        return ck, art, None

    def frozen_verify():
        N.OG.verify_result(member, row, kept)
    ck.run('verify_result', 'frozen:online_groups.verify_result', frozen_verify)

    def stored_evidence():
        e = member.get('evidence') or {}
        need(e.get('result') == row, 'stored evidence result differs from results.jsonl row')
        need(sha256_file(e['demo']['path']) == e['demo']['sha256'], 'demo changed since online acceptance')
        need(sha256_file(e['report']['path']) == e['report']['sha256'], 'report changed since online acceptance')
    ck.run('online_evidence_unchanged', 'own', stored_evidence)

    tro_p = Path(member['tro']['path']); side_p = Path(member['sidecar']['path'])
    out = Path(row['out']); meta_p = out / 'meta.json'; demo_p = out / 'demo.hdf5'
    snap = Path(rel['snapshot'])

    def paths():
        for p in (tro_p, side_p, meta_p, demo_p):
            need(p.resolve().is_relative_to(job.resolve()), 'artifact outside job dir: ' + str(p))
        art.update(tro=ref(tro_p), sidecar=ref(side_p), report=ref(meta_p), demo=ref(demo_p), result_row_sha256=digest(row))
    ck.run('artifacts_in_job_and_hashed', 'own', paths)

    ck.run('report_passed(no goal override, restoration verified, QA)', 'frozen:acceptance.report_passed',
           lambda: N.ACC.report_passed(read(meta_p), 'oracle'))

    def provenance():
        meta = read(meta_p)
        need(meta.get('instance_id') == iid and Path(meta['tro_path']).resolve() == tro_p.resolve(), 'meta identity differs')
        need(meta.get('tro_sha256_verified') is True, 'collector did not verify TRO sha')
        for name, prov in (('meta', meta.get('provenance') or {}), ('result', row.get('provenance') or {})):
            need(Path(prov.get('isg_root', '')).resolve() == snap, name + ' isg_root is not the release snapshot')
            need(Path(prov.get('b1k_root', '')).resolve() == (snap / 'Behavior-1K').resolve(), name + ' b1k_root not in snapshot')
            need(str(prov.get('b1k_commit', '')).startswith(rel['snapshot_b1k_text'].split('+')[0]), name + ' B1K commit differs')
            need(prov.get('isg_dirty') is False, name + ' isg dirty')
        need(row.get('reused_world') == meta.get('reused_world'), 'result/meta reused_world differ')
        need(meta.get('wall_clock_exceeded') is not True, 'oracle wallclock exceeded')
    ck.run('collector_provenance', 'own', provenance)

    def restoration():
        meta = read(meta_p); L = meta.get('loader') or {}
        need((L.get('campaign_restoration') or {}).get('scale_verified') is True, 'campaign restoration not verified')
        tpc = L.get('tro_pose_check')
        need(isinstance(tpc, dict) and tpc.get('n_objects', 0) > 0, 'no TRO pose check after restore')
        need(tpc.get('mismatch') == [] or (tpc.get('reapplied') is True and tpc.get('after_reapply') == []),
             'TRO objects off their pose after restore')
        if meta.get('reused_world'):
            for name, prov in (('meta', meta.get('provenance') or {}), ('result', row.get('provenance') or {})):
                need(prov.get('queue_scene_restore') is True and prov.get('track_state_updates') is True,
                     name + ': reused world without queue scene restore/state tracking')
        return {'reused_world': bool(meta.get('reused_world')), 'tro_pose_reapplied': bool(tpc.get('reapplied'))}
    ck.run('restoration_checks', 'own (reads native run_isg loader evidence)', restoration)

    ck.run('demo_hdf5_nonempty', 'own(h5py)', lambda: hdf5_nonempty(demo_p))

    def tro_block():
        t = read(tro_p); m = t['ispatialgym']; b = m['campaign']
        state.update(online_group=b['runtime']['online_group'],
                     record={'class_id': class_id, 'requested_goal': b['runtime']['goal_group'].get('requested_goal'),
                             'goal_group': b['runtime']['goal_group']})
        N.C.validate_campaign_block(b, object_bindings=m['object_bindings'])
        need(m['instance_id'] == iid, 'TRO instance id differs')
        need(b['split'] == 'train' and b['class_id'] == class_id, 'TRO split/class differs')
        need(b['goal_group_id'] == group['id'], 'TRO goal_group_id differs')
        og = b['runtime']['online_group']
        need(og['group_id'] == group['id'] and og['worker'] == owner_worker, 'TRO online_group binding differs')
        need(b['validation_key'] == cfg_key, 'TRO validation key differs from the native train key of the job campaign_config')
        need(read(group['config']['path'])['validation_key'] == cfg_key, 'group config validation key differs')
        need(b['validation_key']['code_sha256'] == rel['code_sha256'], 'TRO code sha is not the release')
        need(b['created_at'] >= rel['frozen_at'], 'member created before release freeze')
        need(digest(b['perturbations']) == digest(cfg_pert), 'TRO perturbations differ from campaign_config')
        state.update(perturbations=b['perturbations'], created_at=b['created_at'])
    ck.run('tro_campaign_block', 'frozen:contracts.validate_campaign_block + own', tro_block)

    def timing():
        t = ungated_timing(member, group, row, state.get('online_group'), cap); state['timing'] = t
        return t
    ck.run('ungated_train_timing', 'own (published before sampling deadline; demo within demo_safety_cap_s)', timing)

    ck.run('native_goal_equal_to_group', 'frozen:goal_identity.equivalent_goal',
           lambda: need(N.GI.equivalent_goal(group['goal'], state['record']), 'member native goal differs from group goal'))

    def saved():
        s = N.OG.saved_member(str(tro_p)); state['saved'] = s
        return {'state_digest': digest({k: v for k, v in s.items() if k != 'instance_id'})}
    ck.run('saved_member', 'frozen:online_groups.saved_member', saved)

    if vast.get('group_is_vast'):
        def preserved():
            files = vast.get('files') or {}
            need(vast.get('durable_ok') is True, 'durable preservation missing or checksums not verified')
            need(str(row.get('job_id')) == vast.get('job_id'), 'member job %s is not the preserved Vast job %s' % (row.get('job_id'), vast.get('job_id')))
            bad = []
            for p in (tro_p, side_p, meta_p, demo_p):
                rp = str(p.resolve().relative_to(job.resolve())); e = files.get(rp)
                if e is None or e['sha256'] != sha256_file(p):
                    bad.append(rp + (' (absent from preservation)' if e is None else ' (sha differs)'))
            need(not bad, 'unpreserved Vast artifacts: ' + '; '.join(bad))
        ck.run('vast_durable_preservation', 'own', preserved)
    return ck, art, state


def member_category(rows):
    """Exclusion category from a member's failed checks (duplicate start is excluded individually)."""
    errs = ' '.join(r.get('error', '') for r in rows if not r['ok'])
    if 'saved-state difference' in errs:
        return 'duplicate_start'
    if 'ungated' in errs or 'deadline' in errs or 'safety cap' in errs:
        return 'timing'
    return 'verification_failed'


def collect_members(group, results, verify):
    """Re-verify accepted members in state order; kept grows with each passing member (pure orchestration)."""
    kept, out, states = [], [], []
    for m in [x for x in group['members'] if x['status'] == 'accepted']:
        row = results.get(m['instance_id'])
        ck, art, state = verify(m, row, list(kept))
        ok = not ck.failed()
        if ok:
            kept.append(m); states.append((m['instance_id'], state))
        out.append({'instance_id': m['instance_id'], 'eligible': ok, 'checks': ck.rows, 'artifacts': art,
                    'timing': (state or {}).get('timing'), 'perturbations': (state or {}).get('perturbations'),
                    'result_job_id': (row or {}).get('job_id'), 'host': (row or {}).get('host'),
                    'excluded_category': None if ok else member_category(ck.rows),
                    'excluded_reason': None if ok else '; '.join(r.get('error', r['check']) for r in ck.failed())})
    return out, states


# ------------------------------------------------------------- group level
def collect_group(job, g, wck, info, rel):
    src = info.get('_src') or {}; st = info.get('_state') or {}
    evs = info.get('_events') or []; results = info.get('_results') or {}
    class_id = src.get('class_id'); gid = g['id']
    rec = {'class_id': class_id, 'group_id': gid, 'source_key': job.name, 'job_dir': str(job)}
    gck = Checks(); group = g

    def find():
        hits = [x for x in st.get('groups', []) if x['id'] == gid]
        need(len(hits) == 1, 'group finalized %d times in state.json' % len(hits))
        acc = [m for m in group['members'] if m['status'] == 'accepted']
        need(group.get('finished_at') is not None, 'group not finalized')
        need(len(acc) == group['accepted_members'], 'accepted-member counts disagree')
        need(group['claim'].get('claimed') is True, 'goal not claimed')
        need(sha256_file(group['plan']['path']) == group['plan']['sha256'], 'group plan changed')
        need(sha256_file(group['config']['path']) == group['config']['sha256'], 'group config changed')
        need(read(group['config']['path']).get('split') == 'train', 'group config split differs')
        need(group['goal']['class_id'] == class_id, 'group goal class differs')
        return {'accepted': len(acc), 'members_total': len(group['members']), 'reason': group.get('reason'),
                'qualifies': group.get('qualifies')}
    gck.run('state_group_finalized', 'own', find)

    def events():
        mine = [e for e in evs if e.get('group_id') == gid]
        fin = [e for e in mine if e['event'] == 'group_finalized']
        need(len(fin) == 1 and fin[0]['accepted'] == group['accepted_members'], 'group_finalized event mismatch')
        sel = [e for e in mine if e['event'] in ('goal_selected', 'goal_selected_proposal_only')]
        need(len(sel) == 1, 'goal selection event count %d' % len(sel))
        if 'claim' in sel[0]:
            need(sel[0]['claim'] == group['claim'], 'goal_selected claim differs from state')
        need(N.GI.equivalent_goal(sel[0]['goal'], group['goal']), 'goal_selected goal differs')
        acc_ev = [e['instance_id'] for e in mine if e['event'] == 'member_accepted']
        acc_st = [m['instance_id'] for m in group['members'] if m['status'] == 'accepted']
        need(sorted(acc_ev) == sorted(acc_st) and len(set(acc_ev)) == len(acc_ev), 'member_accepted events differ from state')
        need(any(e['event'] == 'group_started' for e in mine), 'no group_started event')
    gck.run('events_consistent', 'own', events)

    def receipt():
        r = group['claim']['receipt']; d = read(r['path'])
        need(Path(r['path']).resolve().is_relative_to((job / 'registry_private').resolve()), 'claim receipt not in the private registry')
        need(sha256_file(r['path']) == r['sha256'], 'claim receipt changed')
        need(d['owner'] == st['owner'] + '/' + gid, 'claim receipt owner differs')
        need(N.GI.equivalent_goal(d['record'], group['goal']), 'claim receipt goal differs')
    gck.run('goal_claim_receipt', 'frozen:goal_identity.equivalent_goal + own', receipt)

    vast = dict(info.get('_vast') or {})
    acc = [m for m in group['members'] if m['status'] == 'accepted']
    vast['group_is_vast'] = bool(vast.get('backend_json') or vast.get('durable') or vast.get('preservation') or any(
        VAST_JOB.match(str((results.get(m['instance_id']) or {}).get('job_id', ''))) for m in acc))
    rec['vast_run'] = vast['group_is_vast']
    if vast['group_is_vast']:
        gck.run('vast_preservation_present', 'own',
                lambda: need(vast.get('preservation') and vast.get('durable') and vast.get('durable_ok'),
                             'Vast output lacks preservation.json/durable_preservation.json with checksums_verified'),
                blocker=True)
    cap = info.get('demo_safety_cap_s') or math.inf
    verify = lambda m, row, kept: verify_member(job, m, row, kept, group, class_id, rel, info.get('tro_validation_key'),
                                                info.get('perturbations'), st.get('owner'), cap, vast)
    members, states = collect_members(group, results, verify)
    pairs = []
    for i in range(len(states)):
        for j in range(i + 1, len(states)):
            c = N.GI.start_comparison(states[i][1]['saved'], states[j][1]['saved'])
            pairs.append({'a': states[i][0], 'b': states[j][0], 'different': c['different'], 'within_noise': c['within_noise'],
                          'comparable': c['comparable'], 'reasons': c['reasons'], 'missing': c['missing']})
    rec['start_comparisons'] = pairs
    gck.run('physical_goal', 'frozen:goal_identity.physical_goal/goal_key',
            lambda: rec.update(physical_goal=N.GI.physical_goal(group['goal']), goal_fingerprint=N.GI.goal_key(group['goal'])))
    elig = [m for m in members if m['eligible']]
    rec.update(goal=group['goal'], claim=group.get('claim'), plan=group.get('plan'), group_config=group.get('config'),
               timing={'started_at': group.get('started_at'), 'sampling_deadline': group.get('deadline'),
                       'finished_at': group.get('finished_at'), 'parked_at': group.get('parked_at'),
                       'parked_after_sampling_deadline': bool(group.get('parked_after_sampling_deadline')),
                       'parked_resolution': group.get('parked_resolution'), 'native_reason': group.get('reason'),
                       'native_timing': group.get('timing'),
                       'seconds_to_accepted_members_published': sorted(m['timing']['seconds_to_publish'] for m in elig),
                       'seconds_to_accepted_members_finished': sorted(m['timing']['seconds_to_finish'] for m in elig),
                       'members_demo_finished_after_sampling_deadline': sorted(
                           m['instance_id'] for m in elig if m['timing']['demo_finished_after_group_sampling_deadline'])},
               nonaccepted_members=[{'instance_id': m['instance_id'], 'status': m['status'], 'reason': m.get('reason')}
                                    for m in group['members'] if m['status'] != 'accepted'])
    rec['worker_checks'] = wck.rows; rec['group_checks'] = gck.rows; rec['members'] = members
    rec['eligible_members'] = [m['instance_id'] for m in elig]
    return rec


def decide_group(rec):
    """-> (decision, category, reasons). Pure: admitted iff no worker/group failure and >= 1 eligible member."""
    reasons, blocked = [], []
    if not rec.get('members'):   # the native worker accepted nobody: nothing to verify or admit
        return 'rejected', 'no_accepted_member', ['native group finalized with 0 accepted members (%s)' % (
            (rec.get('timing') or {}).get('native_reason'))]
    for scope in ('worker_checks', 'group_checks'):
        for r in rec.get(scope, []):
            if not r['ok']:
                (blocked if (r.get('unreadable') or r.get('blocker') or r.get('missing')) else reasons).append(
                    '%s: %s' % (r['check'], r.get('error', 'failed')))
    if blocked:
        return 'rejected', 'blocked_unverifiable', blocked + reasons
    if reasons:
        return 'rejected', 'verification_failed', reasons
    if not rec.get('eligible_members'):
        return 'rejected', 'no_eligible_member', ['member %s: %s' % (m['instance_id'][-14:], m['excluded_reason'])
                                                  for m in rec['members']][:6]
    return 'admitted', None, []


RECEIPT_FIELDS = ('class_id', 'group_id', 'source_key', 'job_dir', 'vast_run', 'goal', 'goal_fingerprint', 'physical_goal',
                  'claim', 'plan', 'group_config', 'timing', 'start_comparisons', 'nonaccepted_members', 'eligible_members')


def build_receipt(rec, info, rel):
    src = info['_src']; pb = info['preparation_binding']
    members = []
    for m in rec['members']:
        members.append({'instance_id': m['instance_id'], 'eligible': m['eligible'], 'allocation': 'train' if m['eligible'] else None,
                        'artifacts': m['artifacts'], 'timing': m['timing'], 'perturbations': m['perturbations'],
                        'result_job_id': m['result_job_id'], 'host': m['host'],
                        'checks_passed': [r['check'] + ' [' + r['origin'] + ']' for r in m['checks'] if r['ok']],
                        'excluded_category': m['excluded_category'], 'excluded_reason': m['excluded_reason']})
    job_ids = sorted({str(m['result_job_id']) for m in rec['members'] if m['eligible'] and m['result_job_id'] is not None})
    body = {'schema': SCHEMA + '-receipt', 'decision': 'admitted', 'workstream': 'train',
            'release': rel['release'], 'admitting_release': release_brief(rel), 'source_release_ref': src['release'],
            'preparation_binding': {k: pb.get(k) for k in ('configuration_id', 'template_key', 'variant_id', 'status', 'scene',
                                                           'rooms', 'identity_sha256', 'manifest_revision', 'manifest_sha256',
                                                           'binding', 'gates')},
            'campaign_config': info['campaign_config'], 'validation_key': info['validation_key'],
            'tro_validation_key': info['tro_validation_key'],
            'perturbations': {k: info['perturbations'][k] for k in ('scales', 'removed_objects', 'lighting')},
            'compute': {'backend': 'vast' if rec['vast_run'] else 'slurm', 'host': info['runtime']['host'],
                        'gpu_models': info['runtime']['gpu_models'], 'slurm_job_id': None if rec['vast_run'] else info.get('slurm_job_id'),
                        'vast_job_id': (info['_vast'].get('job_id') if rec['vast_run'] else None), 'vast_backend': info.get('backend'),
                        'member_job_ids': job_ids},
            'limits': src.get('limits'), 'worker_id': info['_settings'].get('worker_id'),
            'worker_refs': {k: v for k, v in info.items() if k in ('source', 'settings', 'campaign_config', 'runtime', 'state',
                                                                     'events', 'results', 'preservation', 'durable_preservation')},
            **{k: rec[k] for k in RECEIPT_FIELDS},
            'members': members,
            'excluded_members': [{'instance_id': m['instance_id'], 'category': m['excluded_category'], 'reason': m['excluded_reason']}
                                 for m in rec['members'] if not m['eligible']],
            'allocation': {m['instance_id']: 'train' for m in rec['members'] if m['eligible']}}
    body['content_sha256'] = digest(body)
    return body


# ------------------------------------------------------------------- ledger
def build_ledger(receipts, scan, rel, tool_ref):
    per_class = {}
    for d in receipts:
        c = per_class.setdefault(d['class_id'], {'groups': 0, 'members': 0, '_sub': set(), '_tpl': set(),
                                                 'members_demo_after_sampling_deadline': 0, 'excluded_members': 0})
        c['groups'] += 1; c['members'] += len(d['allocation']); c['excluded_members'] += len(d.get('excluded_members', []))
        c['_sub'].add(d['preparation_binding']['configuration_id']); c['_tpl'].add(d['preparation_binding']['template_key'])
        c['members_demo_after_sampling_deadline'] += len(d['timing']['members_demo_finished_after_sampling_deadline'])
    for c in per_class.values():
        c['sub_templates'] = len(c.pop('_sub')); c['templates'] = len(c.pop('_tpl'))
    keys = ('groups', 'members', 'sub_templates', 'templates', 'members_demo_after_sampling_deadline', 'excluded_members')
    totals = {k: sum(c[k] for c in per_class.values()) for k in keys}
    totals['classes'] = len(per_class)
    return {'schema': SCHEMA + '-ledger', 'release': rel['release'], 'tool': tool_ref,
            'per_class': dict(sorted(per_class.items())), 'totals': totals,
            'receipts': sorted([{'class_id': d['class_id'], 'group_id': d['group_id'], 'source_key': d['source_key'],
                                 'configuration_id': d['preparation_binding']['configuration_id'],
                                 'members': sorted(d['allocation']), 'content_sha256': d['content_sha256']} for d in receipts],
                               key=lambda x: (x['class_id'], x['group_id'])),
            'scan': scan}


def receipts_on_disk():
    root = OUT / 'receipts'
    return [read(p) for p in sorted(root.glob('*/*.json'))] if root.exists() else []


def apply_receipts(bodies):
    out = []
    for b in bodies:
        need(re.fullmatch(r'[A-Za-z0-9_.-]+', b['class_id']) and re.fullmatch(r'online-[0-9a-f]{32}', b['group_id']), 'unsafe receipt path')
        status = write_once_bytes(OUT / 'receipts' / b['class_id'] / (b['group_id'] + '.json'), canonical_bytes(b))
        out.append({'class_id': b['class_id'], 'group_id': b['group_id'], 'status': status, 'content_sha256': b['content_sha256']})
    return out


def write_ledger(ledger):
    data = canonical_bytes(ledger); h = hashlib.sha256(data).hexdigest()
    path = OUT / 'ledger' / ('ledger_%s.json' % h)
    status = write_once_bytes(path, data)
    latest = {'schema': SCHEMA + '-latest', 'ledger': {'path': str(path), 'sha256': h},
              'per_class': ledger['per_class'], 'totals': ledger['totals']}
    lstatus = write_if_changed(OUT / 'ledger' / 'LATEST.json', (json.dumps(latest, indent=1, sort_keys=True) + '\n').encode())
    return {'ledger': str(path), 'sha256': h, 'ledger_status': status, 'latest_status': lstatus}


# ---------------------------------------------------------------------- run
def scan_sources(only):
    out = []
    for job in sorted(SOURCES.iterdir()):
        if not job.is_dir() or (only and job.name not in only):
            continue
        out.append(job)
    return out


def admit(a):
    t0 = time.time()
    rel = release_identity(); load_native(rel['snapshot'])
    tool_ref = ref(Path(__file__))
    scan = {'only': sorted(a.only) if a.only else None, 'sources': {}, 'rejected_groups': []}
    bodies, records = [], []
    for job in scan_sources(a.only):
        if (job / 'SUPERSEDED.json').exists():
            scan['sources'][job.name] = {'status': 'skipped_superseded'}; continue
        running = worker_running(job)
        if running:
            scan['sources'][job.name] = {'status': 'pending_worker_not_finished', 'why': running}; continue
        wck, info = check_worker(job, rel)
        st = info.get('_state') or {}
        groups = st.get('groups', [])
        srow = {'status': 'checked', 'class_id': (info.get('_src') or {}).get('class_id'), 'groups_finalized': len(groups),
                'worker_failures': [r['check'] + ': ' + r.get('error', '') for r in wck.failed()], 'admitted': 0, 'rejected': 0}
        for g in groups:
            rec = collect_group(job, g, wck, info, rel)
            decision, category, reasons = decide_group(rec)
            rec.update(decision=decision, category=category, reasons=reasons); records.append(rec)
            if decision == 'admitted':
                bodies.append(build_receipt(rec, info, rel)); srow['admitted'] += 1
            else:
                srow['rejected'] += 1
                scan['rejected_groups'].append({'class_id': rec['class_id'], 'group_id': rec['group_id'], 'source_key': job.name,
                                                'category': category, 'reasons': reasons[:6]})
        scan['sources'][job.name] = srow
    scan['rejected_groups'].sort(key=lambda x: (x['class_id'], x['group_id']))
    result = {'mode': 'apply' if a.apply else 'dry_run', 'release': rel['name'],
              'admitted_this_scan': len(bodies), 'rejected_this_scan': len(scan['rejected_groups']),
              'members_admitted_this_scan': sum(len(b['allocation']) for b in bodies),
              'members_excluded_this_scan': sum(len(b['excluded_members']) for b in bodies),
              'sources': scan['sources'], 'rejected_groups': scan['rejected_groups'],
              'admitted_groups': [{'class_id': b['class_id'], 'group_id': b['group_id'], 'members': sorted(b['allocation']),
                                   'excluded': b['excluded_members'],
                                   'demo_after_sampling_deadline': b['timing']['members_demo_finished_after_sampling_deadline'],
                                   'seconds_to_published': b['timing']['seconds_to_accepted_members_published'],
                                   'seconds_to_finished': b['timing']['seconds_to_accepted_members_finished'],
                                   'content_sha256': b['content_sha256']} for b in bodies]}
    if a.apply:
        need(getpass.getuser() == 'minyeong', 'apply runs as minyeong')
        res = apply_receipts(bodies)
        result['receipts'] = res
        result['conflicts'] = [x for x in res if x['status'] == 'conflict']
        result.update(write_ledger(build_ledger(receipts_on_disk(), scan, rel, tool_ref)))
    result['seconds'] = round(time.time() - t0, 1)
    if a.report:
        Path(a.report).write_text(json.dumps({'result': result, 'records': records}, indent=1, sort_keys=True, default=str) + '\n')
    print(json.dumps(result, indent=1, sort_keys=True, default=str))
    return 1 if result.get('conflicts') else 0


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='cmd', required=True)
    d = sub.add_parser('admit')
    d.add_argument('--only', nargs='*', help='restrict to these source keys')
    d.add_argument('--apply', action='store_true', help='write immutable receipts + content-addressed ledger')
    d.add_argument('--report', help='write the full per-group check records here (outside OUT is fine; not a record)')
    a = p.parse_args(argv)
    return admit(a)


if __name__ == '__main__':
    sys.exit(main())
