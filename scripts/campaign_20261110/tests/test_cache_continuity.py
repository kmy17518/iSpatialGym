import json
from pathlib import Path
import pytest
from scripts.campaign_20261110 import cache_continuity as C
from scripts.campaign_20261110.infra_vast.worker import warm_cache as W
from scripts.campaign_20261110.tests.test_ik_exchange import manifest,entry
from ispatialgym.sampling import exact_exchange as E
from ispatialgym.sampling.v2 import map_tilt as M


def producer(tmp_path):
    root=tmp_path/'campaign_20261110';job=root/'A';context=job/'cache_continuity';context.mkdir(parents=True)
    store=root/'store';m=manifest();mp=context/'manifest.json';C.write(mp,m)
    settings=root/'settings.json';C.write(settings,{'owner':'workerA','store':str(store)})
    C.write(context/'context.json',{'settings_path':str(settings),'owner':'workerA','manifest':str(mp),'compatibility_id':m['compatibility_sha256'],
                                   'overlay_shards':str(job/'overlay'),'base_overlay_release':0,'legacy_roots':{}})
    key,value=entry(m);E.install(context/'exact'/m['compatibility_sha256'],value,key,m)
    shard=job/'overlay/map1/a.jsonl';shard.parent.mkdir(parents=True)
    event={'schema':M.SCHEMA,'map_key':'map1','entry':'cell','outcome':'ok'}
    shard.write_text(json.dumps(event)+'\n')
    return root,job,store,m,mp,shard,event


def test_upload_merge_publication_and_consumer_are_separate(tmp_path,monkeypatch):
    root,job,store,m,mp,shard,event=producer(tmp_path)
    C.flush(job)
    assert not (store/'published').exists()
    receipts=C.reconcile(str(store),root/'durable')
    assert len(receipts)==1 and receipts[0]['checksums_verified']
    latest=store/'published'/m['compatibility_sha256']/'latest.json'
    assert latest.exists()and C.read(latest)['entries']==1
    B=root/'B';B.mkdir();ref=root/'refs.json';C.write(ref,{'provenance':{'robot_refs':{}}})
    settings=root/'Bsettings.json';C.write(settings,{'owner':'workerB','store':str(store),'robot_reference_manifest':str(ref)})
    monkeypatch.setenv('ISPATIALGYM_CAMPAIGN_CONFIG',str(root/'cfg.json'))
    monkeypatch.setattr(C.compat,'build_manifest',lambda *args:m)
    loaded=C.prepare(settings,B)
    assert loaded['loaded_bundles']==1
    receipt=C.read(B/'cache_continuity/consumed_version.json')
    assert receipt['loaded'][0]['entries']==1 and receipt['actual_native_hits']=='see exact_ik_metrics'
    assert C.reconcile(str(store),root/'durable')==[]
    # A growing shard sends only its new complete line, not the old prefix.
    with shard.open('a')as f:f.write(json.dumps(event)+'\n'+json.dumps(event)[:15])
    C.flush(job);C.reconcile(str(store),root/'durable')
    merged=root/'durable/merged'/m['compatibility_sha256']/'overlay'
    values,version=M.load_merged(str(merged),'map1')
    assert values['cell']['ok']==2


def test_corrupt_payload_never_gets_cluster_ack(tmp_path):
    root,job,store,m,mp,shard,event=producer(tmp_path)
    C.flush(job)
    upload=C.read(job/'cache_continuity/upload.json')
    (store/upload['ik']['key']).write_text('{}')
    with pytest.raises(ValueError,match='checksum'):C.reconcile(str(store),root/'durable')
    assert not list((root/'durable/acks').glob('*.json'))


def test_warm_upload_not_consumed_until_cluster_verification(tmp_path):
    root=tmp_path/'warm';source=root/'A/kits/GPU_driver_b1k';source.mkdir(parents=True)
    (source/'.complete').touch();f=source/'global/cache/shader.bin';f.parent.mkdir(parents=True);f.write_bytes(b'compiled')
    texture=root/'A/textures';texture.mkdir();(texture/'key.ovtex.0').write_bytes(b'texture');(texture/'key.ovtex.desc').write_bytes(b'desc')
    state=root/'state';state.mkdir();store=W.Store(str(root/'store'))
    report=W.export(store,source.parent,texture,'abi',state,'workerA')
    assert len(report['receipts'])==2
    before=W.restore(store,root/'B/kits',root/'B/textures','abi','GPU_driver_b1k',root/'B/state')
    assert not before['restored']
    receipts=C.warm_reconcile(str(root/'store'),root/'durable')
    assert len(receipts)==2 and all(r['checksums_verified']for r in receipts)
    after=W.restore(store,root/'B/kits',root/'B/textures','abi','GPU_driver_b1k',root/'B/state')
    assert len(after['restored'])==2
    assert (root/'B/kits/GPU_driver_b1k/global/cache/shader.bin').read_bytes()==b'compiled'
    assert (root/'B/textures/key.ovtex.0').read_bytes()==b'texture'
    assert C.warm_reconcile(str(root/'store'),root/'durable')==[]
