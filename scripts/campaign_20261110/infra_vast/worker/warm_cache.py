"""Production-only immutable Kit/texture cache exchange; no asset/prep writes."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import tarfile
import tempfile
import time

PREFIX = 's3://isg-demos/jobs/og-isg261110/'
SCHEMA = 'isg-warm-cache-v1'


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''): h.update(b)
    return h.hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    with tmp.open('wb') as f:
        f.write(canonical(value)); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)


class Store:
    def __init__(self, root):
        self.root = root.rstrip('/')
        self.remote = self.root.startswith('s3://')
        if self.remote:
            assert (self.root + '/').startswith(PREFIX), 'outside production prefix'
            self.cmd = ['s5cmd', '--endpoint-url', os.environ['S3_ENDPOINT_URL'], '--retry-count', '5']
        else:
            Path(root).mkdir(parents=True, exist_ok=True)

    def url(self, key):
        assert key and not key.startswith('/') and '..' not in PurePosixPath(key).parts
        return self.root + '/' + key

    def list(self, prefix):
        if not self.remote:
            return [str(p.relative_to(self.root)) for p in (Path(self.root) / prefix).rglob('*.json')]
        r = subprocess.run(self.cmd + ['ls', self.url(prefix.rstrip('/') + '/*')], capture_output=True, text=True, timeout=60)
        if r.returncode:
            if any(x in r.stderr.lower() for x in ('no object found', 'nosuchkey', '404')): return []
            raise RuntimeError('cache manifest listing failed')
        keys=[]
        for line in r.stdout.splitlines():
            if not line.split():continue
            name=line.split()[-1]
            if not name.endswith('.json'):continue
            if name.startswith(self.root+'/'):name=name[len(self.root)+1:]
            elif not name.startswith(prefix.rstrip('/')+'/'):name=prefix.rstrip('/')+'/'+name
            keys.append(name)
        return keys

    def get(self, key, path):
        if not self.remote:
            shutil.copyfile(self.url(key), path); return
        subprocess.run(self.cmd + ['cp', self.url(key), str(path)], check=True, capture_output=True, timeout=600)

    def put_verified(self, path, key, *, immutable=True):
        expected = sha(path)
        if not self.remote:
            target = Path(self.url(key)); target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and immutable: assert sha(target) == expected, 'immutable cache object conflict'
            else: shutil.copyfile(path, target)
            assert sha(target) == expected
            return expected
        subprocess.run(self.cmd + ['cp', str(path), self.url(key)], check=True, capture_output=True, timeout=600)
        # Verify returned bytes, not a multipart ETag or merely upload exit code.
        h = hashlib.sha256()
        with tempfile.TemporaryFile() as errors:
            p = subprocess.Popen(self.cmd + ['cat', self.url(key)], stdout=subprocess.PIPE, stderr=errors)
            for block in iter(lambda: p.stdout.read(4 * 1024 * 1024), b''): h.update(block)
            assert p.wait(timeout=60) == 0 and h.hexdigest() == expected, 'cache readback hash mismatch'
        return expected


def safe_kit_files(root):
    """Only reusable caches/data from a helper-published complete template.

    Excludes user configuration, logs, sockets, texture symlinks, decrypted
    assets and secret/key files. Sources outside this exact root are never read.
    """
    root = Path(root)
    if not (root / '.complete').is_file(): return []
    found = []
    for p in root.rglob('*'):
        if not p.is_file() or p.is_symlink(): continue
        name = p.relative_to(root).as_posix()
        if name == '.complete': continue  # recreated only after verified restore
        if not name.startswith(('global/cache/', 'global/data/', 'global/warp_cache/', 'local/cache/')): continue
        if 'texturecache' in p.parts or any(x in name.lower() for x in ('credential', 'secret', 'token')): continue
        if p.suffix.lower() in ('.usd', '.usda', '.usdc', '.key', '.pem') or '.encrypted' in name: continue
        found.append(p)
    return sorted(found)


def texture_files(root, baseline):
    root = Path(root); groups = {}
    if not root.exists(): return []
    for p in root.iterdir():
        stem, sep, _ = p.name.partition('.ovtex.')
        if sep and p.is_file() and not p.is_symlink(): groups.setdefault(stem, []).append(p)
    result = []
    for stem, files in groups.items():
        if not any(p.name == stem + '.ovtex.desc' for p in files): continue
        # Descriptor last in the local helper guarantees a completed conversion.
        if any(p.name not in baseline for p in files): result.extend(files)
    return sorted(result)


def package(store, root, files, *, kind, key, abi, state_dir, instance):
    if not files: return None
    root = Path(root)
    records = []
    for p in files:
        stat = p.stat()
        records.append({'path': p.relative_to(root).as_posix(), 'sha256': sha(p), 'bytes': stat.st_size})
    identity = {'schema': SCHEMA, 'kind': kind, 'key': key, 'abi': abi, 'files': records}
    mid = hashlib.sha256(canonical(identity)).hexdigest()
    receipt_path = Path(state_dir) / 'uploaded' / (mid + '.json')
    if receipt_path.exists(): return json.loads(receipt_path.read_text())
    with tempfile.TemporaryDirectory(dir=state_dir) as temp:
        archive = Path(temp) / 'cache.tar.gz'
        with tarfile.open(archive, 'w:gz', compresslevel=1) as tar:
            for p in files: tar.add(p, arcname=p.relative_to(root).as_posix(), recursive=False)
        # Files are immutable helper publications. Check source stability before
        # making a discoverable descriptor; incomplete uploads cannot be reused.
        assert all(sha(root / r['path']) == r['sha256'] for r in records), 'cache changed during packing'
        blob = 'blobs/' + sha(archive) + '.tar.gz'
        store.put_verified(archive, blob)
        manifest = {**identity, 'archive': {'key': blob, 'sha256': sha(archive), 'bytes': archive.stat().st_size},
                    'instance': instance, 'created_at': time.time()}
        mp = Path(temp) / 'manifest.json'; write(mp, manifest)
        mkey = 'manifests/' + abi + '/' + kind + '/' + key + '/' + sha(mp) + '.json'
        store.put_verified(mp, mkey)
        receipt = {'manifest': mkey, 'manifest_sha256': sha(mp), 'archive_sha256': sha(archive),
                   'files': len(records), 'bytes': sum(r['bytes'] for r in records), 'verified_at': time.time()}
        write(receipt_path, receipt)
        return receipt


def export(store, kit_root, texture_root, abi, state_dir, instance):
    state_dir = Path(state_dir); state_dir.mkdir(parents=True, exist_ok=True)
    baseline_path = state_dir / 'baseline.json'
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {'textures': []}
    receipts = []
    for directory in sorted(Path(kit_root).glob('*')):
        if directory.is_dir() and not directory.is_symlink():
            receipt = package(store, directory, safe_kit_files(directory), kind='kit', key=directory.name,
                              abi=abi, state_dir=state_dir, instance=instance)
            if receipt: receipts.append(receipt)
    texture = package(store, texture_root, texture_files(texture_root, set(baseline['textures'])),
                      kind='textures', key='shared', abi=abi, state_dir=state_dir, instance=instance)
    if texture: receipts.append(texture)
    report = {'schema': SCHEMA, 'instance': instance, 'abi': abi, 'verified': True,
              'verified_at': time.time(), 'receipts': receipts}
    write(state_dir / 'latest.json', report)
    store.put_verified(state_dir / 'latest.json', 'receipts/' + instance + '.json', immutable=False)
    return report


def unpack_verified(store, manifest, dest):
    dest = Path(dest); dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest.parent) as tmp:
        archive = Path(tmp) / 'cache.tar.gz'; store.get(manifest['archive']['key'], archive)
        assert sha(archive) == manifest['archive']['sha256'], 'download hash mismatch'
        expected = {r['path']: r for r in manifest['files']}
        with tarfile.open(archive, 'r:gz') as tar:
            members = tar.getmembers()
            assert len(members) == len(expected) and len({m.name for m in members}) == len(members)
            for m in members:
                assert m.isfile() and m.name in expected and not PurePosixPath(m.name).is_absolute()
                assert '..' not in PurePosixPath(m.name).parts and m.size == expected[m.name]['bytes']
            # Never extract links, devices, or absolute/escaping names.
            for m in members:
                target = Path(tmp) / 'unpacked' / m.name; target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(m) as source, target.open('wb') as out: shutil.copyfileobj(source, out)
                assert sha(target) == expected[m.name]['sha256'], 'cache member hash mismatch'
        # Texture data first, descriptors last. Existing immutable files are
        # verified and retained; nothing already in use gets overwritten.
        for name in sorted(expected, key=lambda n: (n.endswith('.ovtex.desc'), n)):
            source = Path(tmp) / 'unpacked' / name; target = dest / name
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists(): assert target.is_file() and sha(target) == expected[name]['sha256'], 'local cache conflict'
            else:
                temporary = target.with_name(target.name + '.import.' + str(os.getpid()))
                shutil.copyfile(source, temporary); temporary.chmod(0o444)
                try: os.link(temporary, target)
                except FileExistsError: assert sha(target) == expected[name]['sha256']
                finally: temporary.unlink()


def restore(store, kit_root, texture_root, abi, kit_key, state_dir):
    state_dir = Path(state_dir); state_dir.mkdir(parents=True, exist_ok=True)
    restored = []
    for kind, key, destination in [('kit', kit_key, Path(kit_root) / kit_key), ('textures', 'shared', Path(texture_root))]:
        manifests = []
        for name in store.list('published/' + abi + '/' + kind + '/' + key):
            with tempfile.TemporaryDirectory(dir=state_dir) as tmp:
                mp = Path(tmp) / 'm.json'; store.get(name, mp); m = json.loads(mp.read_text()); assert Path(name).stem == sha(mp), 'manifest hash mismatch'
            assert m['schema'] == SCHEMA and m['abi'] == abi and m['kind'] == kind and m['key'] == key
            manifests.append((name, m))
        if kind == 'kit':
            if (destination / '.complete').is_file(): continue
            manifests = sorted(manifests, key=lambda row: len(row[1]['files']), reverse=True)[:1]
        for name, m in manifests:
            unpack_verified(store, m, destination); restored.append(name)
        if kind == 'kit' and manifests: (destination / '.complete').touch()
    # A baseline distinguishes boot/restored files from later local conversions.
    baseline = {'textures': [p.name for p in Path(texture_root).glob('*') if p.is_file()], 'at': time.time()}
    write(state_dir / 'baseline.json', baseline)
    report = {'restored': restored, 'kit_key': kit_key, 'abi': abi, 'at': time.time()}
    write(state_dir / 'restored.json', report)
    return report


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['restore', 'export'])
    p.add_argument('--store', default=os.environ.get('ISG_S3', '').rstrip('/') + '/warm_cache/v1')
    p.add_argument('--kit-root', required=True); p.add_argument('--textures', required=True)
    p.add_argument('--abi', required=True); p.add_argument('--kit-key')
    p.add_argument('--state', default='/work/warm_cache_exchange')
    p.add_argument('--instance', default=os.environ.get('ISG_INSTANCE') or os.environ.get('CONTAINER_ID'))
    a = p.parse_args(); assert a.abi and all(c.isalnum() or c in '-_.' for c in a.abi)
    if a.action == 'restore': result = restore(Store(a.store), a.kit_root, a.textures, a.abi, a.kit_key, a.state)
    else:
        assert a.instance and str(a.instance).isdigit()
        result = export(Store(a.store), a.kit_root, a.textures, a.abi, a.state, a.instance)
    print(json.dumps(result, sort_keys=True))


def capture(appdata, texture_root, state, *, final=False):
    """Keep completed conversions and a retired process's fresh Kit template.

    Active Kit folders are never used as shader templates. Only the owner calls
    final after its sampler and collectors have stopped.
    """
    appdata=Path(appdata);state=Path(state);state.mkdir(parents=True,exist_ok=True)
    texture_root=Path(texture_root);texture_root.mkdir(parents=True,exist_ok=True)
    source=appdata/'global/cache/texturecache';copied=[]
    files=texture_files(source,set())
    for path in sorted(files,key=lambda p:(p.name.endswith('.ovtex.desc'),p.name)):
        target=texture_root/path.name
        if target.exists():
            if sha(target)!=sha(path):raise ValueError('texture cache name has conflicting content')
            continue
        before=path.stat();tmp=state/(path.name+'.capture')
        shutil.copyfile(path,tmp);after=path.stat()
        if (before.st_size,before.st_mtime_ns)!=(after.st_size,after.st_mtime_ns):
            tmp.unlink();raise RuntimeError('texture conversion changed during capture')
        tmp.chmod(0o444)
        # Copy before linking: scratch and durable storage can be different FS.
        staged=target.with_name('.'+target.name+'.'+str(os.getpid()))
        shutil.copyfile(tmp,staged)
        try:os.link(staged,target)
        except FileExistsError:
            if sha(target)!=sha(tmp):raise ValueError('texture publication conflict')
        finally:staged.unlink();tmp.unlink()
        copied.append(path.name)
    kit=None
    if final and appdata.is_dir():
        kit=state/'kit';kit.mkdir(exist_ok=True)
        # Filter by the same allowlist used to upload published templates.
        for path in appdata.rglob('*'):
            if path.is_symlink()or not path.is_file():continue
            name=path.relative_to(appdata).as_posix()
            if not name.startswith(('global/cache/','global/data/','global/warp_cache/','local/cache/')):continue
            if 'texturecache'in path.parts or any(x in name.lower()for x in('credential','secret','token')):continue
            if path.suffix.lower()in('.usd','.usda','.usdc','.key','.pem')or'.encrypted'in name:continue
            target=kit/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(path,target)
        (kit/'.complete').touch()
    return {'textures':copied,'kit':str(kit)if kit else None}
