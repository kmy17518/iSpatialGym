"""Immutable observed attempt records, including failures before sampling."""
import hashlib
import json
import os
from pathlib import Path
import uuid


def retain_attempt(root, job, done):
    attempt = str(done.get('attempt_id') or (str(job.get('instance'))+'_'+str(int(float(job.get('assigned') or 0)*1000000))))
    if not attempt.replace('_','').replace('-','').isalnum():raise ValueError('unsafe attempt identity')
    path=Path(root)/str(job['id'])/(attempt+'.json');path.parent.mkdir(parents=True,exist_ok=True)
    record={'schema':'isg-vast-attempt-v1','job_id':str(job['id']),'attempt_id':attempt,
            'instance':job.get('instance'),'assigned_unix':job.get('assigned'),'try_number':job.get('tries'),'done':done}
    payload=(json.dumps(record,sort_keys=True,indent=2)+'\n').encode()
    if path.exists():
        if path.read_bytes()!=payload:raise ValueError('immutable attempt evidence changed')
    else:
        temporary=path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
        with temporary.open('xb') as stream:stream.write(payload);stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path);temporary.unlink()
    return {'path':str(path),'sha256':hashlib.sha256(payload).hexdigest()}
