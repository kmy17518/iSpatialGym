#!/usr/bin/env python3
"""Hash actual reserved ready asset bytes without changing reservation identities."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reservations',type=Path,required=True)
    p.add_argument('--dataset-root',type=Path,required=True)
    p.add_argument('--generated-readiness',type=Path)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    d=json.loads(a.reservations.read_text())
    generated=json.loads(a.generated_readiness.read_text())['objects'] if a.generated_readiness else {}
    ready=set(d['model_holdout_models'])|set(d['category_holdout_models'])
    all_reserved=set(d['train_exclude_models'])
    pending=all_reserved-ready
    rows={}
    for key in sorted(all_reserved):
        if key in ready:
            category,model=key.split('/')
            path=(a.dataset_root/'objects'/category/model/'usd'/(model+'.encrypted.usd')).resolve()
            rows[key]={'asset_sha256':sha(path),'path':str(path),'ready':True,
                       'readiness_provenance':'Exact ready-only split reservation manifest; file existence does not grant readiness.'}
        elif key in generated and generated[key]['ready']:
            item=generated[key];path=Path(item['usd']).resolve()
            if sha(path)!=item['asset_sha256']:
                raise ValueError(f'generated asset changed since admission: {key}')
            rows[key]={'asset_sha256':item['asset_sha256'],'path':str(path),'ready':True,
                       'readiness_provenance':item['readiness_source']}
            pending.discard(key)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps({'schema':'campaign-reserved-asset-hashes-v1','reservation_sha256':sha(a.reservations),
        'reserved_model_keys':sorted(all_reserved),'ready_assets':rows,'reserved_asset_hashes':sorted({r['asset_sha256'] for r in rows.values()}),
        'pending_unready_model_keys':sorted(pending),'rule':'Exclude both model keys and exact asset hashes globally across every task role; structural walls/floors exempt.'},indent=2,sort_keys=True)+'\n')
    print(json.dumps({'reserved_keys':len(all_reserved),'ready_asset_hashes':len(rows),'pending_unready':sorted(pending),'output':str(a.output)}))


if __name__=='__main__':
    main()
