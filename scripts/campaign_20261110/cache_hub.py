"""CPU-only publication bridge between production-owned local and R2 caches."""
from __future__ import annotations
import argparse, hashlib, json, subprocess, tempfile, time
from pathlib import Path
from scripts.campaign_20261110 import cache_continuity as C
from scripts.campaign_20261110.infra_vast.worker.warm_cache import Store, sha, write


def publish(store, path, key, state, *, immutable=True):
    h = sha(path)
    marker = state / (hashlib.sha256((store.root+'|'+key).encode()).hexdigest()+'.json')
    if marker.exists() and json.loads(marker.read_text()).get('sha256') == h:
        return
    store.put_verified(path, key, immutable=immutable)
    write(marker, {'key': key, 'sha256': h, 'at': time.time()})


def bridge(local, remote, durable, state, warm=False):
    stores = [Store(local), Store(remote)]
    durable, state = Path(durable), Path(state)
    state.mkdir(parents=True, exist_ok=True)
    reports = []
    for store in stores:
        reports.extend((C.warm_reconcile if warm else C.reconcile)(store.root, durable))

    def fetch(reference):
        target = durable/'objects'/(reference['sha256']+Path(reference['key']).suffix)
        for store in stores:
            try:
                return C.get(store, reference, target)
            except (OSError, RuntimeError, subprocess.CalledProcessError):
                continue
        raise ValueError('published cache object unavailable: '+reference['key'])

    if warm:
        for receipt in (durable/'receipts').glob('*.json'):
            r = json.loads(receipt.read_text())
            mp = Path(r['durable_cluster'])/'manifest.json'
            m = json.loads(mp.read_text())
            archive = durable/'archives'/(m['archive']['sha256']+'.tar.gz')
            archive.parent.mkdir(exist_ok=True)
            if not archive.exists():
                for source in stores:
                    try:
                        source.get(m['archive']['key'], archive)
                        if sha(archive) != m['archive']['sha256']:
                            archive.unlink(); raise ValueError('warm archive hash differs')
                        break
                    except (OSError, RuntimeError, subprocess.CalledProcessError):
                        archive.unlink(missing_ok=True)
                if not archive.exists(): raise ValueError('warm archive unavailable')
            for store in stores:
                publish(store, archive, m['archive']['key'], state)
                publish(store, mp, r['published'], state)
                publish(store, receipt, 'acks/'+receipt.name, state)
    else:
        for merged in (durable/'merged').glob('*'):
            versions = list((merged/'versions').glob('*.json'))
            if not versions: continue
            vp = max(versions, key=lambda p: json.loads(p.read_text())['at'])
            v = json.loads(vp.read_text())
            refs = v['ik_bundles']+list(v['overlay_files'].values())
            for store in stores:
                for ref in refs: publish(store, fetch(ref), ref['key'], state)
                prefix = 'published/'+v['compatibility_id']+'/'
                publish(store, vp, prefix+v['version']+'.json', state)
                publish(store, vp, prefix+'latest.json', state, immutable=False)
        for ack in (durable/'acks').glob('*.json'):
            r = json.loads(ack.read_text())
            key = 'acks/'+r['owner']+'/'+r['upload']['sha256']+'.json'
            for store in stores: publish(store, ack, key, state)
    return reports


def main():
    p = argparse.ArgumentParser(); p.add_argument('--config', required=True); p.add_argument('--once', action='store_true')
    a = p.parse_args(); cfg = json.loads(Path(a.config).read_text())
    while True:
        report = {'at': time.time(), 'streams': {}, 'errors': {}}
        for name, spec in cfg['streams'].items():
            try: report['streams'][name] = bridge(**spec)
            except Exception as exc: report['errors'][name] = repr(exc)
        write(cfg['status'], report)
        print(json.dumps({'at': report['at'], 'receipts': {k: len(v) for k,v in report['streams'].items()}, 'errors': report['errors']}), flush=True)
        if a.once: break
        time.sleep(30)


if __name__ == '__main__': main()
