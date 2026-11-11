import json
from pathlib import Path
import pytest
from scripts.campaign_20261110.contracts import file_digest
from scripts.campaign_20261110.training_eligibility import immutable_evidence_json,_EVIDENCE_CACHE


def test_cached_evidence_still_detects_changed_bytes_and_is_not_mutable(tmp_path,monkeypatch):
    p=tmp_path/'proof.json';p.write_text('{"success":true}')
    expected=file_digest(p);_EVIDENCE_CACHE.clear();actual=Path.read_bytes;calls=[]
    def measured(path):calls.append(str(path));return actual(path)
    monkeypatch.setattr(Path,'read_bytes',measured)
    first=immutable_evidence_json(p,expected);first['success']=False
    assert immutable_evidence_json(p,expected)=={'success':True} and len(calls)==1
    p.write_text('{"success":false}')
    with pytest.raises(ValueError,match='hash changed'):immutable_evidence_json(p,expected)
    assert len(calls)==2


def test_changed_expected_hash_is_not_served_from_cache(tmp_path):
    p=tmp_path/'proof.json';p.write_text('{}');immutable_evidence_json(p,file_digest(p))
    with pytest.raises(ValueError,match='hash changed'):immutable_evidence_json(p,'0'*64)
