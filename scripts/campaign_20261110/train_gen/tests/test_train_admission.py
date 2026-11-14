import importlib.util, json, pathlib

import pytest

spec = importlib.util.spec_from_file_location('ta', pathlib.Path(__file__).parents[1] / 'train_admission.py')
ta = importlib.util.module_from_spec(spec); spec.loader.exec_module(ta)

CAP = 1800.0


def member(pub=100.0, dl=400.0, cap=CAP, iid='m1'):
    return {'instance_id': iid, 'published_at': pub, 'deadline': dl, 'demo_deadline': pub + cap, 'status': 'accepted'}


def group(gdl=400.0):
    return {'id': 'online-' + '0' * 32, 'started_at': 0.0, 'deadline': gdl}


def og(m, g, gate='ungated_train_safety_cap'):
    return {'demo_gate': gate, 'demo_deadline': m['demo_deadline'], 'group_deadline': g['deadline'], 'member_deadline': m['deadline']}


# ----------------------------------------------------------- ungated timing
def test_demo_after_sampling_deadline_is_accepted_and_flagged():
    m, g = member(), group()
    t = ta.ungated_timing(m, g, {'finished_at_unix': 1500.0}, og(m, g), CAP)
    assert t['demo_finished_after_group_sampling_deadline'] is True
    assert t['seconds_to_publish'] == 100.0 and t['seconds_to_finish'] == 1500.0 and t['demo_seconds'] == 1400.0


def test_demo_within_sampling_deadline_not_flagged():
    m, g = member(), group()
    assert ta.ungated_timing(m, g, {'finished_at_unix': 300.0}, og(m, g), CAP)['demo_finished_after_group_sampling_deadline'] is False


def test_published_after_group_sampling_deadline_rejected():
    m, g = member(pub=450.0, dl=500.0), group(gdl=400.0)
    with pytest.raises(ValueError, match='group sampling deadline'):
        ta.ungated_timing(m, g, {'finished_at_unix': 600.0}, og(m, g), CAP)


def test_published_after_member_deadline_rejected():
    m, g = member(pub=350.0, dl=300.0), group()
    with pytest.raises(ValueError, match='member sampling deadline'):
        ta.ungated_timing(m, g, {'finished_at_unix': 600.0}, og(m, g), CAP)


def test_demo_past_safety_cap_rejected():
    m, g = member(), group()
    with pytest.raises(ValueError, match='safety cap'):
        ta.ungated_timing(m, g, {'finished_at_unix': 100.0 + CAP + 1}, og(m, g), CAP)


def test_eval_style_member_without_demo_deadline_rejected():
    m, g = member(), group(); m.pop('demo_deadline')
    with pytest.raises(ValueError, match='not ungated'):
        ta.ungated_timing(m, g, {'finished_at_unix': 300.0}, {}, CAP)


def test_gate_and_cap_consistency():
    m, g = member(), group()
    with pytest.raises(ValueError, match='demo_gate'):
        ta.ungated_timing(m, g, {'finished_at_unix': 300.0}, og(m, g, gate=None), CAP)
    with pytest.raises(ValueError, match='published_at \\+ demo_safety_cap_s'):
        ta.ungated_timing(m, g, {'finished_at_unix': 300.0}, og(m, g), 900.0)


# ------------------------------------------------- member exclusion / decision
def _ck(ok, err=None):
    c = ta.Checks()
    c.run('verify_result', 'frozen:online_groups.verify_result', (lambda: True) if ok else (lambda: ta.need(False, err)))
    return c


def test_duplicate_start_member_excluded_individually():
    g = {'members': [member(iid='a'), member(iid='b'), member(iid='c'), {'instance_id': 'd', 'status': 'expired'}]}
    seen = []

    def verify(m, row, kept):
        seen.append((m['instance_id'], [k['instance_id'] for k in kept]))
        if m['instance_id'] == 'b':
            return _ck(False, 'no meaningful saved-state difference from accepted member'), {}, None
        return _ck(True), {}, {'timing': {'seconds_to_publish': 1.0}}
    out, states = ta.collect_members(g, {}, verify)
    assert seen == [('a', []), ('b', ['a']), ('c', ['a'])]       # b never enters kept
    assert [m['eligible'] for m in out] == [True, False, True]
    assert out[1]['excluded_category'] == 'duplicate_start'
    rec = {'worker_checks': [], 'group_checks': [], 'members': out, 'eligible_members': ['a', 'c']}
    assert ta.decide_group(rec)[0] == 'admitted'


def test_single_member_group_admitted():
    rec = {'worker_checks': [], 'group_checks': [], 'members': [{'instance_id': 'a'}], 'eligible_members': ['a']}
    assert ta.decide_group(rec) == ('admitted', None, [])


def test_zero_accepted_and_failures_rejected():
    assert ta.decide_group({'members': [], 'timing': {'native_reason': 'budget_or_indices_exhausted'}})[1] == 'no_accepted_member'
    bad = {'check': 'runtime_provenance', 'origin': 'own', 'ok': False, 'error': 'native returncode 1'}
    rec = {'worker_checks': [bad], 'group_checks': [], 'members': [{'instance_id': 'a'}], 'eligible_members': ['a']}
    assert ta.decide_group(rec)[1] == 'verification_failed'
    rec['worker_checks'] = [{**bad, 'missing': True}]
    assert ta.decide_group(rec)[1] == 'blocked_unverifiable'
    rec = {'worker_checks': [], 'group_checks': [], 'members': [{'instance_id': 'a', 'excluded_reason': 'x'}], 'eligible_members': []}
    assert ta.decide_group(rec)[1] == 'no_eligible_member'


# ------------------------------------------------------- records + idempotence
def receipt(cls, gid, cfg, tpl, members, late=()):
    b = {'class_id': cls, 'group_id': gid, 'source_key': 't2611_1', 'allocation': {m: 'train' for m in members},
         'excluded_members': [], 'preparation_binding': {'configuration_id': cfg, 'template_key': tpl},
         'timing': {'members_demo_finished_after_sampling_deadline': list(late)}}
    b['content_sha256'] = ta.digest(b)
    return b


@pytest.fixture
def out(tmp_path, monkeypatch):
    monkeypatch.setattr(ta, 'OUT', tmp_path / 'admission_r1')
    monkeypatch.setattr(ta, 'EVAL_TREE', tmp_path / 'vast')
    return tmp_path / 'admission_r1'


def test_receipts_write_once_idempotent_and_conflict(out):
    b = receipt('cls_a', 'online-' + '1' * 32, 'cfg1', 'tpl1', ['m1'])
    assert ta.apply_receipts([b])[0]['status'] == 'created'
    assert ta.apply_receipts([b])[0]['status'] == 'identical'
    changed = dict(b, allocation={'m1': 'train', 'm2': 'train'}); changed['content_sha256'] = 'x'
    assert ta.apply_receipts([changed])[0]['status'] == 'conflict'
    assert json.loads((out / 'receipts/cls_a' / (b['group_id'] + '.json')).read_text()) == b   # never overwritten


def test_writes_refused_outside_out_and_under_eval_tree(out, tmp_path):
    with pytest.raises(ValueError, match='outside admission root'):
        ta.write_once_bytes(tmp_path / 'elsewhere.json', b'{}')
    ta.EVAL_TREE = out.parent        # an OUT inside the eval tree must also be refused
    with pytest.raises(ValueError, match='eval tree'):
        ta.write_once_bytes(out / 'receipts/x.json', b'{}')


def test_ledger_counts_and_idempotent_rerun(out):
    bs = [receipt('cls_a', 'online-' + '1' * 32, 'cfg1', 'tpl1', ['m1', 'm2'], late=['m2']),
          receipt('cls_a', 'online-' + '2' * 32, 'cfg2', 'tpl1', ['m3']),
          receipt('cls_b', 'online-' + '3' * 32, 'cfg3', 'tpl3', ['m4'])]
    ta.apply_receipts(bs)
    rel = {'release': {'path': '/r.json', 'sha256': 'r'}}
    led = ta.build_ledger(ta.receipts_on_disk(), {'sources': {}}, rel, {'sha256': 't'})
    assert led['per_class']['cls_a'] == {'groups': 2, 'members': 3, 'sub_templates': 2, 'templates': 1,
                                         'members_demo_after_sampling_deadline': 1, 'excluded_members': 0}
    assert led['totals']['groups'] == 3 and led['totals']['members'] == 4 and led['totals']['classes'] == 2
    first = ta.write_ledger(led)
    assert first['ledger_status'] == 'created' and first['latest_status'] == 'written'
    again = ta.write_ledger(ta.build_ledger(ta.receipts_on_disk(), {'sources': {}}, rel, {'sha256': 't'}))
    assert again == {**first, 'ledger_status': 'identical', 'latest_status': 'unchanged'}
    assert json.loads((out / 'ledger/LATEST.json').read_text())['ledger']['sha256'] == first['sha256']
