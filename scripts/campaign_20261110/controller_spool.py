"""Durable handoff of prepared controller work to the shared capacity router."""
from pathlib import Path

from scripts.campaign_20261110.contracts import file_digest, require


def enqueue_record(root, record, save, *, priority=50, backends=None):
    from scripts.campaign_20261110.infra_dispatch import BACKENDS, enqueue
    require(record.get('dry_run_passed') is True,'spooled work requires the exact command dry-run')
    token=Path(record['directory']).name
    options={'priority':priority}
    backends=tuple(backends or record.get('dispatch_backends',BACKENDS))
    command=record['command']
    if '--code' in command:
        runtime=Path(command[command.index('--code')+1])/'ispatialgym/sampling/campaign_runtime.py'
        if 'fcntl.flock(' in runtime.read_text():
            backends=tuple(backend for backend in backends if backend.startswith('vast_'))
            require(backends,'legacy NFS journal runtime requires a local-storage Vast backend')
            record['routing_constraint']={'reason':'known_unsupported_nfs_failure_journal_lock',
                'runtime_source':str(runtime),'runtime_sha256':file_digest(runtime)}
    options['backends']=backends
    path=enqueue(root,token,record['command'],**options)
    record.update(status='external_submission',dispatch_intent=str(path),
                  dispatch_receipt=str(Path(root)/'dispatch/receipts'/(token+'.json')))
    save()


def candidate_budget_used(records):
    return sum(row['count'] for row in records if not row.get('migration',{}).get('never_started'))
