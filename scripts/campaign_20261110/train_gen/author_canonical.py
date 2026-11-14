#!/usr/bin/env python3
"""Author TRAIN sources for the canonical (eval-template) training sub-template of a class.

Runs on the cluster with plain python3 (stdlib only). Isolation: writes only under
T=R/training_generation/train_online_r1; eval sources (V/production_r1) are read only.

  python3 author_canonical.py --class <cid> --groups <n> [--kcollect 2] [--tag smoke] [--dry-run]

Steps: newest eval source of the class -> copy campaign_config with split=train and the train
release code_sha256 -> match the preparation validity row by the gate's own identity -> gate
`check` (must exit 0) -> settings (train limits from the proposal, private registry,
demo_safety_cap_s) -> job.env/run.sh rewritten to the train snapshot and train paths ->
source.json. Ordinals are allocated by atomic mkdir; index start = 2000600000000 + (ord-10000)*1e5.
"""
import argparse, csv, glob, hashlib, json, os, re, shlex, subprocess, sys, time
from pathlib import Path

R = Path('/vision/u/minyeong/projects/isg_home/campaign_20261110')
V = R / 'vast/online_grouped_r1'
T = R / 'training_generation/train_online_r1'
RP = R / 'training_preparation/preparation_vast_20261114T034355Z'
GATE = RP / 'validity/code/train_validity_gate.py'
SNAP = R / 'code/train_699ba262b46e'
RELEASE = R / 'training_generation/releases/release_train_r2.json'
PROPOSAL = R / 'training_generation/handoff_20261114/train_class_wallclock_proposal.csv'
OBSERVER = R / 'tools/eval_runtime_observer_5da2a8fc.py'
GIVEUPS = T / 'subtemplate_giveups.jsonl'
INDEX_BASE = 2000600000000
SEARCH_EXCLUDE = re.compile(r'(_r\d+|\.)')


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p): return {'path': str(p), 'sha256': sha(p)}
def wjson(p, d):
    p = Path(p); tmp = p.with_name('.' + p.name + '.tmp'); tmp.write_text(json.dumps(d, indent=2) + '\n'); os.replace(tmp, p)


def eval_index():
    """class -> newest eval source dir (by campaign_config mtime); cached per call in T/eval_source_index.json."""
    cache = T / 'eval_source_index.json'
    if cache.exists() and time.time() - cache.stat().st_mtime < 6 * 3600:
        return json.loads(cache.read_text())
    best, mags = {}, {}
    for src in glob.glob(str(V / 'production_r1/*/source.json')):
        try:
            s = json.loads(Path(src).read_text()); cfg = Path(src).with_name('campaign_config.json')
            m = cfg.stat().st_mtime
            mr = json.loads(Path(src).with_name('settings.json').read_text()).get('magnitude_range')
        except (OSError, ValueError):
            continue
        cid = s.get('class_id')
        if cid and (cid not in best or m > best[cid][1]):
            best[cid] = (str(Path(src).parent), m)
        if cid and mr:
            lo, hi = mags.get(cid, (mr[0], mr[1])); mags[cid] = (min(lo, mr[0]), max(hi, mr[1]))
    out = {'sources': {k: v[0] for k, v in best.items()}, 'magnitude_union': {k: list(v) for k, v in mags.items()}}
    wjson(cache, out)
    return out


def load_env(path):
    env = {}
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.startswith('#') and '=' in line:
            k, v = line.split('=', 1); vals = shlex.split(v); env[k] = vals[0] if vals else ''
    return env


def dump_env(env):
    return ''.join(f'{k}={shlex.quote(str(v))}\n' for k, v in env.items())


def proposal_row(cid):
    for row in csv.DictReader(PROPOSAL.open()):
        if row['class_id'] == cid:
            return row
    raise SystemExit('class not in proposal: ' + cid)


def gate(*args):
    p = subprocess.run([sys.executable, str(GATE), *map(str, args)], capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def gate_identity(cfg_path):
    """The gate's own identity of a config (import the gate module; no reimplementation)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location('tvg', GATE); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    cid, scene, roles, room = mod.roles_from_campaign_config(json.loads(Path(cfg_path).read_text()))
    return mod.identity_sha(cid, scene, roles, room)


def allocate():
    (T / 'ordinals').mkdir(parents=True, exist_ok=True)
    existing = [int(p.name) for p in (T / 'ordinals').iterdir() if p.name.isdigit()]
    o = max(existing + [9999]) + 1
    while True:
        if 10080 <= o < 10100:   # index ranges of the test slots k=80..99
            o += 1; continue
        try:
            (T / 'ordinals' / str(o)).mkdir(); return o
        except FileExistsError:
            o += 1


def seconds_for(groups, L, cap):
    return int(min(14400, 1500 + groups * L + cap + 420))


def author(cid, groups, kcollect=2, tag='canon', dry=False, lighting=None, gpu2080=False, limits=None):
    row = proposal_row(cid)
    if row['status'] != 'RUN':
        raise SystemExit(f'{cid} is {row["status"]}')
    idx = eval_index(); src = Path(idx['sources'][cid])
    cfg = json.loads((src / 'campaign_config.json').read_text())
    release = json.loads(RELEASE.read_text())
    L, Lm, cap = int(row['train_group_limit_s']), int(row['train_member_limit_s']), int(row['demo_safety_cap_s'])
    if limits:   # bounded demonstrations only (e.g. the ungated-demo test); recorded in source.json
        L, Lm = int(limits[0]), int(limits[1])
    if cid == 'configuration_matching-articulation_open_large_scale_01':
        cap = max(cap, 3600)
    cfg['split'] = 'train'
    cfg['validation_key']['code_sha256'] = release['validation_key']['code_sha256']
    if lighting:
        cfg['perturbations']['lighting'] = lighting
    o = allocate(); key = f't2611_{o}'; job = T / 'sources' / key
    job.mkdir(parents=True)
    wjson(job / 'campaign_config.json', cfg)
    # preparation binding: the valid row with the gate identity of this config
    rc, out, err = gate('valid', '--class', cid)
    rows = [json.loads(l) for l in out.splitlines() if l.strip()]
    ident = gate_identity(job / 'campaign_config.json')
    match = [r for r in rows if r['identity_sha256'] == ident]
    if len(match) != 1:
        (job / 'AUTHOR_FAILED.json').write_text(json.dumps({'reason': 'no unique valid row', 'identity': ident, 'n_valid': len(rows), 'matches': len(match)}))
        raise SystemExit(f'{cid}: {len(match)} valid rows match identity {ident}')
    wjson(job / 'preparation_binding.json', {'configuration_id': match[0]['configuration_id']})
    rc, out, err = gate('check', job / 'campaign_config.json', '--binding', job / 'preparation_binding.json', '--giveups', GIVEUPS)
    (job / 'gate_author.txt').write_text(f'rc={rc}\n{out}\n{err}')
    if rc != 0:
        raise SystemExit(f'{cid}: gate check rc={rc}: {out} {err}')
    latest = json.loads((RP / 'validity/LATEST.json').read_text())
    seconds = seconds_for(groups, L, cap)
    start = INDEX_BASE + (o - 10000) * 100000
    es = json.loads((src / 'settings.json').read_text())
    settings = {'worker_id': 'og-isgtrain-' + key, 'state_dir': str(job / 'online'), 'results': str(job / 'results.jsonl'),
                'groups': groups, 'target_members': 3, 'minimum_members': 1, 'group_limit_s': L, 'member_limit_s': Lm,
                'demo_safety_cap_s': cap,
                # train: every eval bucket of the class (union of eval sources' ranges); per-bucket balance dropped (owner)
                'magnitude_range': idx['magnitude_union'].get(cid) or es.get('magnitude_range', [0.0, 1.0]), 'index_count': 100000,
                'require_qualifying_groups': False, 'registry': str(job / 'registry_private')}
    wjson(job / 'settings.json', settings)
    env = load_env(src / 'job.env')
    old_job = str(src)
    for k, v in list(env.items()):
        if old_job in v:
            env[k] = v.replace(old_job, str(job))
    env.update(ISGE_CODE=str(SNAP), ISGF_SCRIPTS=str(SNAP / 'scripts/trainfast'), ISGF_ROOT=str(R),
               ISGF_JB_NFS=str(T / 'jobs'), ISGF_PREDEMO_ROOT=str(T), ISGF_SAMPLING_CACHE=str(job / 'sampling_cache'),
               ISPATIALGYM_CAMPAIGN_CONFIG=str(job / 'campaign_config.json'), ISPATIALGYM_ONLINE_GROUPS=str(job / 'settings.json'),
               START=str(start), JOB_TAG=key, KCOLLECT=str(1 if gpu2080 else kcollect), TIME_LIMIT_S=str(seconds),
               SAMPLE_LIMIT_S=str(seconds - 600), DEMO_LIMIT_S=str(cap))
    if 'ISPATIALGYM_MAP_OVERLAY_SHARDS' in env:
        env['ISPATIALGYM_MAP_OVERLAY_SHARDS'] = str(job / 'map_overlay')
    extra = re.sub(r'--wallclock \d+(\.\d+)?', f'--wallclock {cap}', env.get('ISGF_COLLECT_EXTRA', ''))
    if '--wallclock' not in extra:
        extra = (extra + f' --wallclock {cap}').strip()
    if cid == 'configuration_matching-block_design_01' and '--max-steps' not in extra:
        extra += ' --max-steps 6000'   # class_config_fix_r1
    env['ISGF_COLLECT_EXTRA'] = extra
    env.pop('ISGF_WALLCLOCK_POLICY', None)
    leftovers = [k for k, v in env.items() if 'online_grouped_r1/production_r1' in v or 'online_grouped_r1/code_' in v]
    if leftovers:
        raise SystemExit(f'{cid}: eval paths left in job.env: {leftovers}')
    (job / 'job.env').write_text(dump_env(env))
    (job / 'sampling_cache').mkdir(exist_ok=True)
    request = sha(job / 'settings.json')
    sdk = re.search(r'--expected-sdk-sha256 ([0-9a-f]{64})', (src / 'run.sh').read_text()).group(1)
    py = '/vision/u/minyeong/miniconda3/envs/ispatialgym-vectorized-envs/bin/python'
    run = f"""#!/bin/bash
set -euo pipefail
umask 0002
unset CONDA_PREFIX CONDA_DEFAULT_ENV CONDA_SHLVL _CE_CONDA _CE_M PYTHONHOME
export PATH=/vision/u/minyeong/miniconda3/envs/ispatialgym-vectorized-envs/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
source /vision/u/minyeong/miniconda3/etc/profile.d/conda.sh
conda activate ispatialgym-vectorized-envs
hash -r
export OPTIX_CACHE_PATH=/tmp/isgtrain_optix_${{SLURM_JOB_ID:-local}}
python -c 'import sys,numpy; assert sys.version_info[:2] == (3,11); print("NATIVE_ENV_READY",sys.executable,numpy.__version__,flush=True)'
# dispatch-time gate (the manifest can move): skip on any non-zero exit
python3 {GATE} check {job}/campaign_config.json --binding {job}/preparation_binding.json --giveups {GIVEUPS} > {job}/gate_dispatch.txt 2>&1 || {{ echo GATE_SKIP; exit 3; }}
exec {py} {OBSERVER} --output {job}/runtime.json --snapshot {SNAP} --expected-sdk-sha256 {sdk} --class-id {cid} --request-sha256 {request} --phase generation -- bash {SNAP}/scripts/trainfast/job_fast.sh {job}
"""
    (job / 'run.sh').write_text(run); os.chmod(job / 'run.sh', 0o775)
    source = {'owner': 'train_gen_20261114', 'workstream': 'train', 'key': key, 'ordinal': o, 'class_id': cid, 'tag': tag,
              'groups': groups, 'job_dir': str(job), 'seconds': seconds, 'start': start, 'index_count': 100000,
              'release': ref(RELEASE), 'native_commit': release['candidate_commit'], 'snapshot': str(SNAP),
              'settings': ref(job / 'settings.json'), 'campaign_config': ref(job / 'campaign_config.json'),
              'job_env': ref(job / 'job.env'), 'wrapper': ref(job / 'run.sh'),
              'preparation_binding': {**match[0], 'binding': ref(job / 'preparation_binding.json'),
                                      'manifest_revision': latest['revision'], 'manifest_sha256': latest['sha256']},
              'proposal': ref(PROPOSAL), 'limits': {'group_limit_s': L, 'member_limit_s': Lm, 'demo_safety_cap_s': cap,
                                                    'subtemplate_giveup_s': int(row['subtemplate_giveup_s'])},
              'eval_source_cloned': {'job_dir': str(src), 'campaign_config': ref(src / 'campaign_config.json')},
              'kcollect': int(env['KCOLLECT']), 'gpu2080': gpu2080, 'authored_at': time.time()}
    wjson(job / 'source.json', source)
    return source


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--class', dest='cid', required=True); ap.add_argument('--groups', type=int, default=1)
    ap.add_argument('--kcollect', type=int, default=2); ap.add_argument('--tag', default='canon')
    ap.add_argument('--lighting-json'); ap.add_argument('--gpu2080', action='store_true')
    ap.add_argument('--test-limits', nargs=2, type=int, help='group_limit_s member_limit_s override (bounded tests only)')
    a = ap.parse_args()
    s = author(a.cid, a.groups, a.kcollect, a.tag, lighting=json.loads(a.lighting_json) if a.lighting_json else None,
               gpu2080=a.gpu2080, limits=a.test_limits)
    print(json.dumps({k: s[k] for k in ('key', 'class_id', 'job_dir', 'seconds', 'start')}))
