"""Durable handoff of prepared controller work to the shared capacity router."""
from pathlib import Path

from scripts.campaign_20261110.contracts import require


def enqueue_record(root, record, save, *, priority=50):
    from scripts.campaign_20261110.infra_dispatch import enqueue
    require(record.get('dry_run_passed') is True,'spooled work requires the exact command dry-run')
    token=Path(record['directory']).name
    options={'priority':priority}
    if record.get('dispatch_backends'):options['backends']=tuple(record['dispatch_backends'])
    path=enqueue(root,token,record['command'],**options)
    record.update(status='external_submission',dispatch_intent=str(path),
                  dispatch_receipt=str(Path(root)/'dispatch/receipts'/(token+'.json')))
    save()


def candidate_budget_used(records):
    return sum(row['count'] for row in records if not row.get('migration',{}).get('never_started'))
