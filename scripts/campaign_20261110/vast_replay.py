"""Immutable transport for campaign-owned, real physics-action replay jobs.

Preparing a transport never changes its source or activates GPU work. Native
replay outputs remain beneath the transported job directory; an empty canonical
gate directory may point at that tree only after scheduler ownership is settled.
"""
from pathlib import Path
import hashlib
import json
import os
import re
import shlex
import time
import uuid


SCHEMA = 'isg-vast-physics-replay-inputs-v1'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b''): h.update(part)
    return h.hexdigest()


def require(value, message):
    if not value: raise ValueError(message)


def read_env(path):
    result = {}
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith('#'): continue
        key, value = line.split('=', 1)
        words = shlex.split(value)
        require(len(words) <= 1, 'invalid source environment value')
        result[key] = words[0] if words else ''
    return result


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def prepare(root, source_job, snapshot, helper_snapshot, canonical_output, token):
    root, source = Path(root).resolve(), Path(source_job).resolve()
    snapshot, helper = Path(snapshot).resolve(), Path(helper_snapshot).resolve()
    output = Path(canonical_output).absolute()
    require(re.fullmatch(r'[A-Za-z0-9_-]+', token), 'unsafe replay token')
    require(source.is_relative_to(root / 'out') and (source / 'sampling.done').is_file(), 'source sampling is not closed inside campaign')
    require(snapshot.is_relative_to(root / 'code') and helper.is_relative_to(root / 'code'), 'replay runtimes must be owned snapshots')
    require(output.is_relative_to(root / 'validation') and output.parent.resolve().is_relative_to(root / 'validation')
            and not output.is_symlink(), 'canonical replay output must be owned and not redirected')
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())), 'canonical replay output already has evidence')
    environment = read_env(source / 'job.env')
    require(Path(environment['ISGE_CODE']).resolve() == snapshot, 'source runtime differs from pinned replay runtime')
    selected = None
    for meta in sorted((source / 'demo').glob('*/meta.json')):
        report = json.loads(meta.read_text()); demo = meta.with_name('demo.hdf5')
        tros = list((source / 'inst').rglob(meta.parent.name + '-tro_state.json'))
        if report.get('success') is True and demo.is_file() and len(tros) == 1:
            selected = meta, report, demo, tros[0]; break
    require(selected is not None, 'no unambiguous successful source demo')
    meta, report, demo, tro = selected
    require(Path(report['provenance']['isg_root']).resolve() == snapshot, 'demo runtime differs from pinned replay runtime')
    template = Path(report['template_dir']).resolve()
    require(template.is_relative_to(source / 'inst'), 'source template lies outside source job')
    from scripts.campaign_20261110.infra_replay_scene import prepare as prepare_scene
    scene_identity = prepare_scene(demo, source)
    paths = [source/'job.env', source/'sampling.done', meta, demo, tro,
             snapshot/'.snapshot_commit', helper/'scripts/campaign_20261110/job_replay.sh',
             helper/'scripts/campaign_20261110/infra_open_loop.sh', helper/'scripts/campaign_20261110/infra_replay_scene.py',
             helper/'scripts/campaign_20261110/vast_replay.py']
    paths += [p for p in template.iterdir() if p.is_file()]
    files = {str(p.resolve()): sha(p) for p in paths}
    job = root / 'out/vast_replay' / token
    require(not job.exists(), 'replay transport token already exists')
    job.mkdir(parents=True)
    additions = {'ISGF_REPLAY_BACKEND':'vast', 'ISGF_REPLAY_SOURCE_JOB':str(source),
                 'ISGF_REPLAY_SOURCE_SNAPSHOT':str(snapshot), 'ISGF_REPLAY_HELPER_SNAPSHOT':str(helper)}
    require(not set(additions) & set(environment), 'source environment already contains replay transport keys')
    (job/'job.env').write_bytes((source/'job.env').read_bytes() + b'\n' +
        ''.join(k+'='+shlex.quote(v)+'\n' for k,v in additions.items()).encode())
    files[str(job/'job.env')] = sha(job/'job.env')
    manifest = {'schema':SCHEMA, 'campaign_root':str(root), 'source_job':str(source), 'snapshot':str(snapshot),
                'helper_snapshot':str(helper), 'canonical_output':str(output), 'job_dir':str(job),
                'source_demo':str(demo), 'source_tro':str(tro), 'source_template':str(template),
                'source_scene_identity':scene_identity, 'files':files, 'prepared_at_unix':time.time()}
    write(job/'replay_inputs.json', manifest)
    return job


def verify(job, root=None):
    job = Path(job).resolve(); value = json.loads((job/'replay_inputs.json').read_text())
    owned = Path(root or value['campaign_root']).resolve()
    require(value['schema'] == SCHEMA and value['campaign_root'] == str(owned), 'replay manifest identity differs')
    require(job.is_relative_to(owned/'out/vast_replay') and value['job_dir'] == str(job), 'replay job lies outside its owned transport')
    for name, expected in value['files'].items():
        path = Path(name).resolve()
        require(path.is_relative_to(owned), 'replay source input lies outside campaign')
        require(path.is_file() and sha(path) == expected, 'replay source bytes changed: ' + name)
    return value


def transport_inputs(job, root):
    value = verify(job, root)
    return list(value['files']) + [value['source_template']]


def bind_output(job, root):
    value = verify(job, root)
    output = Path(value['canonical_output']); target = Path(job).resolve()/'gate'
    require(output.absolute().is_relative_to(Path(root).resolve()/'validation'), 'gate output outside campaign')
    require(output.parent.resolve().is_relative_to(Path(root).resolve()/'validation'), 'gate output parent escapes campaign')
    if output.is_symlink():
        require(output.resolve() == target.resolve(), 'canonical gate already points elsewhere')
        return
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())), 'canonical gate has existing evidence')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists(): output.rmdir()  # Only an explicitly checked empty attempt directory.
    output.symlink_to(target)


def enqueue(job, root, *, gpu='RTX 4090'):
    """Publish one typed queue record; separate allowlist activation is required."""
    root, job = Path(root).resolve(), Path(job).resolve(); value = verify(job, root)
    receipt = job/'vast_submission.json'
    if receipt.exists():
        previous = json.loads(receipt.read_text()); record = root/'vast/queue'/(str(previous['id'])+'.json')
        require(record.is_file() and sha(record) == previous['record_sha256'], 'ambiguous replay submission requires reconciliation')
        return str(previous['id'])
    ids = root/'vast/queue/ids'; ids.mkdir(parents=True, exist_ok=True)
    jid = max([900000000] + [int(p.name) for p in ids.iterdir() if p.name.isdigit()]) + 1
    while True:
        try: (ids/str(jid)).mkdir(); break
        except FileExistsError: jid += 1
    record = {'id':jid, 'job_kind':'physics_replay', 'name':'isg261110_replay_vast',
              'script':str(Path(value['helper_snapshot'])/'scripts/campaign_20261110/job_replay.sh'),
              'job_dir':str(job), 'time_s':2400, 'cpus':9, 'mem':'48G', 'acct':'vision', 'part':'svl',
              'log':str(root/'slurm'/('isg261110_replay_vast_'+str(jid)+'.log')), 'cwd':str(root),
              'submitted':time.time(), 'submitted_by':'campaign_physics_replay', 'gpu':gpu,
              'source_job':value['source_job'], 'replay_inputs_sha256':sha(job/'replay_inputs.json')}
    destination = root/'vast/queue'/(str(jid)+'.json'); write(destination, record)
    write(receipt, {'id':jid, 'record':str(destination), 'record_sha256':sha(destination), 'activated':False})
    return str(jid)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('--verify', type=Path, required=True)
    result = verify(parser.parse_args().verify)
    print(json.dumps({'verified':True, 'source_job':result['source_job'], 'source_demo':result['source_demo']}))
