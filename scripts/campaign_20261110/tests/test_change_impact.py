import json
from pathlib import Path

from scripts.campaign_20261110 import change_impact as impact
from scripts.campaign_20261110.contracts import file_digest, write_json_atomic


def fixture(tmp_path,monkeypatch):
    from scripts.campaign_20261110 import export_release,controller_families
    tro=tmp_path/'out/job/inst/alias/template/instances/instance__1-tro_state.json'
    write_json_atomic(tro,{'ispatialgym':{'instance_id':'instance__1','campaign':{
        'class_id':'test','task':'test','split':'eval_standard','validation_key':{'code_sha256':'a'*64}}}})
    accepted={'instance_id':'instance__1','group_id':'group','split':'eval_standard','sampling_split':'eval_standard',
              'class_id':'test','task':'test','tro':str(tro),'tro_sha256':file_digest(tro)}
    monkeypatch.setattr(export_release,'snapshot_allocations',lambda root:([accepted],{'group':{}},[]))
    monkeypatch.setattr(controller_families,'allocations',lambda root:[])
    cat=tmp_path/'catalog.json';write_json_atomic(cat,{'schema':'isg-change-impact-catalog-v1','baseline_commit':'a'*40,
        'current_commit':'b'*40,'changes':[{'change_id':'b'*40,'review_state':'impact_scope_pending'}]})
    return tro,accepted,cat


def test_empty_decisions_are_never_complete_and_diagnostic_copy_not_counted_accepted(tmp_path,monkeypatch):
    tro,_,cat=fixture(tmp_path,monkeypatch)
    copy=tmp_path/'diagnostics/compare/inst/alias/template/instances'/tro.name;copy.parent.mkdir(parents=True);copy.write_bytes(tro.read_bytes())
    out=tmp_path/'validation/change_impact/run1';report=impact.inventory(tmp_path,cat,out)
    assert report['accepted_indexed']==report['generated_unique_ids']==1
    assert report['native_artifact_records']==2 and not report['review_complete']
    rows=[json.loads(l) for l in (out/'instances.jsonl').read_text().splitlines()]
    assert sum(r['accepted'] for r in rows)==1
    assert next(r for r in rows if not r['accepted'])['acceptance_relation'].startswith('retained copy')
    ledger=json.loads((tmp_path/'validation/change_impact/ledger.json').read_text())
    assert ledger['decisions']==[] and ledger['coverage']['complete'] is False
    assert ledger['summary']['changes_pending']==1 and ledger['summary']['verified']==0


def test_accepted_hash_mismatch_is_explicit_inventory_gap(tmp_path,monkeypatch):
    tro,accepted,cat=fixture(tmp_path,monkeypatch);accepted['tro_sha256']='0'*64
    out=tmp_path/'validation/change_impact/run1';report=impact.inventory(tmp_path,cat,out)
    assert report['accepted_indexed']==1
    assert any(g['reason']=='accepted native hash differs' for g in report['gaps'])
    row=json.loads((out/'instances.jsonl').read_text());assert row['accepted_source_integrity'] is False


def test_missing_native_does_not_silently_drop_accepted_identity(tmp_path,monkeypatch):
    tro,_,cat=fixture(tmp_path,monkeypatch);tro.unlink()
    report=impact.inventory(tmp_path,cat,tmp_path/'validation/change_impact/run1')
    assert report['accepted_records']==1 and report['accepted_indexed']==0
    assert any(g.get('instance_id')=='instance__1' for g in report['gaps'])
