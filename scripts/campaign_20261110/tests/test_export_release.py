from dataclasses import asdict, replace
from pathlib import Path
import json

import pytest

from scripts.campaign_20261110 import contracts as c, export_release as ex
from scripts.campaign_20261110.tests.test_group_lane import candidate, claim, reservations


def exported_fixture(root):
    f, review = candidate(root, 1)
    claim(root, f, review)
    rows, groups, inputs = ex.snapshot_allocations(root, root / "dataset/quotas.sqlite3")
    auditor = ex.ArtifactAudit(root, f.since)
    hashes = {"schema": "campaign-reserved-asset-hashes-v1", "reserved_model_keys": [],
              "ready_assets": {}, "reserved_asset_hashes": [], "pending_unready_model_keys": []}
    normalized = [ex.normalize_record(row, root=root, started_at=f.since,
                  templates={f.template.fingerprint: f.template}, reservations=reservations(),
                  reservation_hashes=hashes, auditor=auditor)[0] for row in rows]
    return f, normalized, groups, inputs


def test_export_reads_only_committed_state_and_preserves_source_tros(tmp_path):
    f, rows, groups, inputs = exported_fixture(tmp_path)
    # A publication orphan is never authority for acceptance.
    c.write_json_atomic(tmp_path / "dataset/groups/orphan.json", {"accepted": {"bogus": "eval_standard"}})
    before = {m["artifacts"]["tro"]["path"]: Path(m["artifacts"]["tro"]["path"]).read_bytes() for m in rows}
    ex.audit_assignments(rows, groups, [f.template] * len(rows))
    dest, result = ex.publish(tmp_path, "review01", rows, groups, inputs)
    assert result["counts"] == {"eval_standard": 1, "train": 1}
    assert result["render_shards"] == 1
    assert result["goal_reuse_policy"] == ex.GOAL_REUSE_POLICY
    assert next(row["policy"] for row in inputs if row.get("kind") == "goal_reuse_policy") == ex.GOAL_REUSE_POLICY
    assert all(Path(path).read_bytes() == data for path, data in before.items())
    persisted = [json.loads(line) for line in (dest / "instances.jsonl").read_text().splitlines()]
    assert persisted[1]["split"] == "train" and persisted[1]["sampling_split"] == "eval_standard"
    assert set(groups) == {"g1_group"}
    with pytest.raises(c.ContractError, match="already exists"):
        ex.publish(tmp_path, "review01", rows, groups, inputs)


def test_export_catches_missing_sibling_duplicate_and_modified_demo(tmp_path):
    f, rows, groups, _ = exported_fixture(tmp_path)
    with pytest.raises(c.ContractError, match="duplicate instance"):
        ex.audit_assignments(rows + rows, groups, [f.template])
    altered = {gid: {**record, "accepted": {rows[0]["instance_id"]: "eval_standard"}} for gid, record in groups.items()}
    with pytest.raises(c.ContractError, match="sibling"):
        ex.audit_assignments(rows[:1], altered, [f.template])
    item = rows[0]["artifacts"]["demo"]
    Path(item["path"]).write_bytes(b"changed")
    with pytest.raises(c.ContractError, match="artifact changed"):
        ex.ArtifactAudit(tmp_path, f.since).check(item)


def test_snapshot_never_opens_shared_database_with_sqlite(tmp_path, monkeypatch):
    f, review = candidate(tmp_path, 1)
    claim(tmp_path, f, review)
    shared = tmp_path / "dataset/quotas.sqlite3"
    original = ex.sqlite3.connect
    def guarded(path, *args, **kwargs):
        assert str(tmp_path) not in str(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(ex.sqlite3, "connect", guarded)
    rows, _, _ = ex.snapshot_allocations(tmp_path, shared)
    assert len(rows) == 2


def test_group_export_does_not_admit_cross_allocator_duplicate(tmp_path):
    f, rows, groups, _ = exported_fixture(tmp_path)
    duplicate = {**rows[0], "group_id": None, "task": "camera_relocalization-standard"}
    with pytest.raises(c.ContractError, match="duplicate instance"):
        ex.audit_assignments(rows + [duplicate], groups, [f.template])


def test_ready_contract_required_even_when_source_artifacts_are_valid(tmp_path):
    f, rows, _, _ = exported_fixture(tmp_path)
    source = rows[0]
    source["contract"] = asdict(f.template)
    source["contract"]["objects"][0]["ready"] = False
    with pytest.raises(c.ContractError, match="confirmed readiness"):
        ex.normalize_record(source, root=tmp_path, started_at=f.since, templates={}, reservations={},
                            reservation_hashes={}, auditor=ex.ArtifactAudit(tmp_path, f.since))


def test_pool_count_variants_share_eval_pin_but_keep_physical_render_identity(tmp_path):
    from scripts.campaign_20261110 import binding_pools as bp
    target = c.ObjectRef('target', 'target', 'apple', 'a', c.digest('a'), ready=True, readiness_source='catalog')
    distractor = c.ObjectRef('d1', 'distractor', 'pear', 'b', c.digest('b'), ready=True, readiness_source='catalog')
    unused = replace(distractor, instance='inactive', model='unused', asset_sha256=c.digest('unused'))
    support = c.ObjectRef('support', 'support', 'table', 's', c.digest('s'), ready=True, readiness_source='catalog')
    pool = bp.BindingPool('disemb_pool', 'disemb_class', 'eval_standard', 'house_single_floor', 'living_room_0',
        (support,), (bp.PoolCandidate('target', target), bp.PoolCandidate('distractor', distractor, 2), bp.PoolCandidate('unused', unused)),
        {'target': (1, 1), 'distractor': (1, 2)}, c.digest('source'))
    manifest = bp.write_pool(tmp_path / 'pool.json', pool)
    rows, templates, groups = [], [], {}
    for count in (1, 2):
        template = c.Template('count_'+str(count), 'disembedding', 'disemb_class', 'eval_standard', pool.scene, pool.room,
            (support, target)+tuple(replace(distractor, instance='d'+str(i)) for i in range(1, count+1)), c.digest(count))
        template = bp.bind_variant(template, manifest, {'target': 'target', **{'d'+str(i): 'distractor' for i in range(1, count+1)}})
        gid = 'group'+str(count)
        members = []
        for split in ('eval_standard', 'train'):
            iid = gid+'_'+split
            row = {'instance_id': iid, 'campaign_id': 'fresh', 'split': split, 'task': 'disembedding', 'class_id': 'disemb_class',
                   'group_id': gid, 'bucket': count-1, 'perturbations': {}, 'template': asdict(template),
                   'physical_template_sha256': ex.physical_template(template), 'conceptual_template_sha256': ex.conceptual_template(template)}
            members.append(row)
            rows.append(row)
            templates.append(template)
        groups[gid] = {'accepted': {r['instance_id']: r['split'] for r in members}}
    ex.audit_assignments(rows, groups, templates)
    assert len({r['conceptual_template_sha256'] for r in rows}) == 1
    assert len({r['physical_template_sha256'] for r in rows}) == 2
    rows[0]['conceptual_template_sha256'] = c.digest('forged')
    with pytest.raises(c.ContractError, match='conceptual template identity changed'):
        ex.audit_assignments(rows, groups, templates)
    rows[0]['conceptual_template_sha256'] = ex.conceptual_template(templates[0])
    # A maximum pool member excluded from this physical variant must still be
    # covered by the global split audit.
    held = replace(unused, instance='held_target', role='target')
    held_template = c.Template('held', 'object_scaling', 'held', 'eval_model', pool.scene, pool.room,
                              (held,), c.digest('held_definition'))
    with pytest.raises(c.ContractError, match='global split leakage'):
        ex.audit_assignments(rows[:2], {key: groups[key] for key in ('group1',)}, templates[:2]+[held_template])
