"""Backend identity from campaign-owned allocation/state records, never ID prefixes."""
from pathlib import Path


def vast_job_ids(root):
    root = Path(root)
    return {path.stem for folder in (root/'vast/queue',root/'vast/state/jobs')
            for path in folder.glob('*.json') if path.stem.isdigit()}
