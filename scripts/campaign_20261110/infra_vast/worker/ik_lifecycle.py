"""Bounded exact-IK transport, separate from job artifacts and all other caches.

The supplied helper validates typed JSON witnesses. Missing/invalid exchange
state never blocks a native solve. No live code changes or remote query reads.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import fcntl
import subprocess
import threading
import time

CAMPAIGN = 'campaign_20261110'
HEX = re.compile(r'^[0-9a-f]{64}$')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.' + str(os.getpid()) + '.tmp')
    tmp.write_text(json.dumps(value, sort_keys=True) + '\n'); tmp.replace(path)


class ExactIKLifecycle:
    def __init__(self, spec, s5, remote_root, instance, *, local_root='/work/campaign_20261110/exact_ik', runner=subprocess.run):
        self.spec, self.s5, self.remote, self.instance = spec, s5, remote_root.rstrip('/'), str(instance)
        self.runner = runner; self.lock = threading.Lock(); self.stop = threading.Event()
        self.job = str(spec['id']); self.config = spec.get('exact_ik_exchange')
        self.root = Path(local_root); self.job_root = self.root/'jobs'/self.job
        self.report = {'campaign_id': CAMPAIGN, 'job_id': self.job, 'instance': self.instance, 'events': []}
        self.enabled = False

    def event(self, stage, **values):
        self.report['events'].append({'observed_at': time.time(), 'stage': stage, **values})
        self.report['events'] = self.report['events'][-100:]
        atomic(self.job_root/'lifecycle.json', self.report)

    def call(self, action, *args, timeout=25):
        env = dict(os.environ); env['PYTHONPATH'] = self.config['helper_code']
        command = ['python3', '-m', 'scripts.campaign_20261110.ik_exchange', action,
                   '--manifest', self.config['manifest_path'], *map(str,args)]
        result = self.runner(command, env=env, capture_output=True, text=True, timeout=timeout)
        if result.returncode: raise RuntimeError('validated IK helper failed with exit ' + str(result.returncode))
        return json.loads(result.stdout)

    def prepare(self):
        if not self.config: return {}
        start = time.monotonic()
        try:
            c = self.config
            if c.get('campaign_id') != CAMPAIGN or not HEX.fullmatch(c.get('compatibility_id','')): raise ValueError('foreign compatibility')
            if digest(c['manifest_path']) != c['manifest_sha256']: raise ValueError('manifest transport mismatch')
            if CAMPAIGN not in str(self.root.resolve()): raise ValueError('foreign local cache root')
            self.job_root.mkdir(parents=True, exist_ok=True)
            self.enabled = True
            atomic(self.root/'contexts'/(self.job+'.json'), {'spec': {k:self.spec[k] for k in ('id','job_dir','exact_ik_exchange')},
                   'instance':self.instance,'remote_root':self.remote,'local_root':str(self.root)})
            seed = c.get('seed')
            if seed:
                if not HEX.fullmatch(seed.get('sha256','')) or seed['key'] != 'ik_exchange/seeds/'+seed['sha256']+'.json': raise ValueError('unsafe seed identity')
                path = self.job_root/'seed.json'
                self.s5('cp', self.remote+'/'+seed['key'], str(path), timeout=45)
                if path.stat().st_size > 20*1024*1024 or digest(path) != seed['sha256']: raise ValueError('seed checksum/size differs')
                report = self.call('ingest', '--bundle', path, '--cache', self.root/'cache', '--expected-sha256', seed['sha256'],
                                   '--receipt', self.job_root/'seed_receipt.json', '--deadline-seconds', '10')
                self.event('import', complete=report['complete'], entries=report['entries'], branches=report['branches'],
                           bytes=report['bytes'], elapsed_seconds=time.monotonic()-start)
            else: self.event('import', complete=True, entries=0, branches=0, bytes=0, reason='no compatible seed yet')
            environment = {'ISPATIALGYM_EXACT_IK_MANIFEST': c['manifest_path'],
                    'ISPATIALGYM_EXACT_IK_LOCAL_CACHE': str(self.root/'cache'),
                    'ISPATIALGYM_EXACT_IK_PRODUCER': 'vast:'+self.instance+':'+self.job,
                    'ISPATIALGYM_EXACT_IK_METRICS': str(Path(self.spec['job_dir'])/'exact_ik_metrics')}
            env_path=Path(self.spec['job_dir'])/'job.env'
            if env_path.exists():
                before=env_path.read_text(); lines=before.splitlines(); keys=set(environment)
                lines=[line for line in lines if line.split('=',1)[0] not in keys]
                lines.extend(k+'='+shlex.quote(v) for k,v in environment.items())
                backup=env_path.with_name('job.env.before_exact_ik'); backup.write_text(before)
                temp=env_path.with_name('job.env.exact_ik.tmp');temp.write_text('\n'.join(lines)+'\n');temp.replace(env_path)
                self.event('cache_environment',before_sha256=digest(backup),after_sha256=digest(env_path),values=environment)
            return environment
        except Exception as exc:
            self.enabled = False; self.event('import_failed', error=type(exc).__name__, elapsed_seconds=time.monotonic()-start)
            return {}

    def flush(self, reason='periodic'):
        if not self.enabled or not self.lock.acquire(blocking=False): return
        start = time.monotonic()
        # /work is local instance storage. This lock also protects boot's bounded
        # final-flush subprocess, and is released automatically on interruption.
        handle = None
        try:
            handle=(self.job_root/'flush.lock').open('a')
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            bundle = self.job_root/'delta.json'; cursor = self.job_root/'cursor.json'
            args = ['--cache',self.root/'cache','--bundle',bundle,'--cursor',cursor,'--max-entries','256',
                    '--max-bytes',str(8*1024*1024),'--deadline-seconds','5','--source','vast:'+self.instance+':'+self.job]
            packed = self.call('pack', *args, timeout=15)
            if not packed['entry_count']:
                self.event('export', reason=reason, entries=0, scan_complete=packed['scan_complete'], failures=packed['failures'])
                return
            sha = packed['sha256']; key = 'ik_exchange/published/'+self.instance+'/'+self.job+'/'+sha+'.json'
            self.s5('cp', str(bundle), self.remote+'/'+key, timeout=45)
            receipt = {k:packed[k] for k in ('sha256','entry_count','branch_count','bytes','compatibility_id','elapsed_seconds')}
            receipt.update(campaign_id=CAMPAIGN, job_id=self.job, instance=self.instance, bundle_key=key,
                           manifest_path=self.config['manifest_path'], manifest_sha256=self.config['manifest_sha256'],
                           reason=reason, uploaded_at=time.time())
            self.s5('pipe', self.remote+'/'+key+'.receipt.json', inp=json.dumps(receipt), timeout=30)
            # The immutable typed payload and descriptor both reached durable R2.
            # Coordinator separately validates/merges and publishes its acknowledgement.
            atomic(cursor, packed['cursor'])
            self.event('export', **receipt, durable_object_uploaded=True, durable_merge_pending=True)
        except Exception as exc:
            self.event('export_failed', reason=reason, error=type(exc).__name__, elapsed_seconds=time.monotonic()-start)
        finally:
            if handle is not None: handle.close()
            self.lock.release()
            # An empty postjob/drain delta still closes the lifecycle scan. Keep
            # its evidence visible even when the branch above returns early.
            try: self.s5('cp', str(self.job_root/'lifecycle.json'), self.remote+'/ik_exchange/status/'+self.instance+'/'+self.job+'.json', timeout=15)
            except Exception: pass

    def periodic(self):
        while not self.stop.wait(180): self.flush('periodic')

    def begin(self):
        if self.enabled:
            threading.Thread(target=self.periodic, daemon=True).start()

    def finish(self):
        self.stop.set(); self.flush('postjob')


def flush_stored():
    """Boot calls this under an outer60s timeout before normal destruction."""
    root=Path('/work/campaign_20261110/exact_ik')
    def transfer(*args,check=True,inp=None,timeout=30):
        result=subprocess.run(['s5cmd','--endpoint-url',os.environ['S3_ENDPOINT_URL'],*args],
                              input=inp,text=True,capture_output=True,timeout=timeout)
        if check and result.returncode:raise RuntimeError('bounded final cache upload failed')
        return result
    deadline=time.monotonic()+50
    for path in sorted((root/'contexts').glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True)[:8]:
        if time.monotonic()>deadline:break
        try:
            value=json.loads(path.read_text())
            if value['instance']!=os.environ.get('ISG_INSTANCE'):continue
            context=ExactIKLifecycle(value['spec'],transfer,os.environ['ISG_S3'],value['instance'])
            if digest(context.config['manifest_path'])!=context.config['manifest_sha256']:continue
            context.enabled=True;context.flush('bounded_boot_final_flush')
        except Exception:continue


if __name__=='__main__':
    flush_stored()
