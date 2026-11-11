import json
from pathlib import Path

import pytest

from scripts.campaign_20261110.controller_groups import source_registry_context, prepare_source
from scripts.campaign_20261110.contracts import file_digest
from ispatialgym.sampling.interiors import InteriorSpec


def measured(tmp_path):
    path=tmp_path/'registry/interiors.json';path.parent.mkdir()
    spec=InteriorSpec('measured',.8,.3,.4,.35)
    path.write_text(json.dumps({'schema':'isg-interior/1','interiors':{'measured':spec.to_json()}}))
    proof=path.with_name('registry_lookup_proof.json')
    proof.write_text(json.dumps({'runtime_registry':str(path),'runtime_registry_sha256':file_digest(path),
        'geometry_changed':False,'selected_name':'measured','selected_digest':spec.digest(),
        'original_geometry_digest':spec.digest()}))
    return path,proof


def test_unchanged_source_registry_and_proof_are_retained(tmp_path,monkeypatch):
    path,proof=measured(tmp_path)
    env={'ISPATIALGYM_INTERIORS':str(path)}
    actual=source_registry_context(env,tmp_path)
    assert actual['environment']==env
    assert actual['registry']=={'path':str(path),'sha256':file_digest(path)}
    assert actual['lookup_proof']=={'path':str(proof),'sha256':file_digest(proof)}
    from scripts.campaign_20261110 import planning_inputs, goal_groups
    monkeypatch.setattr(goal_groups,'read_env',lambda p:env)
    def check(*a,**kw):
        assert kw['extra_paths'][-2:]==[str(path),str(proof)]
        return 'exact byte audit before quota lock'
    monkeypatch.setattr(planning_inputs,'prepare_anchor',check)
    assert prepare_source({'anchor_job':str(tmp_path),'anchor_ledger':'ledger','seed_config':'config',
        'snapshot':'native'},{'started_at':1},tmp_path)=='exact byte audit before quota lock'


@pytest.mark.parametrize('change',['bytes','geometry','foreign'])
def test_registry_context_rejects_changed_or_foreign_inputs(tmp_path,change):
    path,proof=measured(tmp_path)
    root=tmp_path
    if change=='bytes':path.write_text(path.read_text()+'\n')
    if change=='geometry':
        value=json.loads(proof.read_text());value['selected_digest']='wrong';proof.write_text(json.dumps(value))
    if change=='foreign':root=tmp_path/'other_root'
    with pytest.raises(ValueError):source_registry_context({'ISPATIALGYM_INTERIORS':str(path)},root)


def test_sources_without_registry_keep_default_behavior(tmp_path):
    assert source_registry_context({},tmp_path) is None
