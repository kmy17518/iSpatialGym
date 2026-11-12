from types import SimpleNamespace

from scripts.campaign_20261110.controller_groups import next_work, reserve_pending_counts, ranges


def test_briefcase_refill_audits_source_context_before_dispatch(tmp_path):
    import copy
    import pytest
    from scripts.campaign_20261110 import contracts as c
    from scripts.campaign_20261110.controller_groups import audit_authored_source_context
    from ispatialgym.sampling.campaign_floor_push_proposals import author_hint, CLASS_ID, FIELD
    summary = {'manipulation_stance': {'base_pose2d': [1., 2., .3], 'focus': [2., 2., .8],
               'translate': .5, 'tilt': -1.2}, 'arm': {'arm': 'right'}, 'grasp': {'name': 'top_0~r180'}}
    source = tmp_path/'source_summary.json'; c.write_json_atomic(source, summary)
    anchor = {'summary': summary, 'sources': {'summary': {'path': str(source), 'sha256': c.file_digest(source)}}}
    template = SimpleNamespace(class_id=CLASS_ID)
    plan = {'class_id': CLASS_ID, 'anchor_sources': anchor['sources'], 'goal_selection': 'fresh_anchor',
            FIELD: author_hint(anchor, CLASS_ID)}
    path = tmp_path/'groups'/'one'/'goal_plan.json'; queue = tmp_path/'queues'/'one'/'queue.json'
    def publish(value):
        c.write_json_atomic(path, value)
        c.write_json_atomic(queue, {'entries': [{'path': str(path), 'sha256': c.file_digest(path)}]})
    publish(plan)
    evidence = audit_authored_source_context({'queue': str(queue)}, anchor, template, tmp_path)
    assert c.file_digest(evidence['path']) == evidence['sha256']
    for change in ('missing', 'wrong_stance', 'stale_hash', 'different_goal'):
        bad = copy.deepcopy(plan)
        if change == 'missing': del bad[FIELD]
        if change == 'wrong_stance': bad[FIELD]['tilt'] = 0.
        if change == 'stale_hash': bad[FIELD]['source_summary_sha256'] = '0'*64
        if change == 'different_goal': bad['goal_selection'] = 'changed'
        publish(bad)
        with pytest.raises(Exception, match='proposal|hash'):
            audit_authored_source_context({'queue': str(queue)}, anchor, template, tmp_path)
    assert audit_authored_source_context({}, {}, SimpleNamespace(class_id='unrelated'), tmp_path) is None


def test_shelf_refill_rejects_drawn_start_as_fixed_release_reference(tmp_path):
    import copy
    import pytest
    from scripts.campaign_20261110 import contracts as c
    from scripts.campaign_20261110.controller_groups import audit_authored_source_context
    from ispatialgym.sampling.campaign_shelf_goals import author_context, CLASS
    parent = {'mode':'native_shelf_tip','support_name':'bookcase','grasp_key':'book@lean_left','edge_name':'+u',
        'target_board':'left','arm':'left','row_z':.45,'bay_m':0.,'target_lean_deg':20.,'partner_lean_deg':15.,
        'spine_behind_edge_m':.06,'source_target_start':{'position':[0.,0.,.6],'orientation':[0.,0.,0.,1.]},
        'source_summary_sha256':'a'*64,'source_sidecar_sha256':'b'*64}
    release = {'position':[.1,-.122109,.6],'orientation':[0.,0.,0.,1.]}
    summary = {'shelf_start':{'board_draw':{},'spine_behind_edge_m':.0443,'target_depth_proposal':{
        'source_depth_m':.06,'target_depth_m':.044274,'partner_depth_unchanged':True,'goal_unchanged':True}},
        'touching':{'tip_release':True},'campaign_goal_group':{'phase':'settled_goal_verified','relation':'touching',
        'target_name':'target','plan_sha256':'c'*64,'shelf_tip_context':parent,'requested_release':release,'requested_place':release}}
    source = tmp_path/'summary.json'; sidecar = tmp_path/'sidecar.json'
    c.write_json_atomic(source,summary); c.write_json_atomic(sidecar,{'place_kind':'shelf_tip'})
    anchor = {'summary':summary,'sources':{k:{'path':str(p),'sha256':c.file_digest(p)}for k,p in [('summary',source),('sidecar',sidecar)]},
        'planned_start_poses':{'target':[[0.,.015726,.6],[0.,0.,0.,1.]]},
        'planned_goal_poses':{'target':[release['position'],release['orientation']]},
        'template_meta':{'bindings':[{'role':'target','name':'target'},{'role':'support','category':'bookcase','scene_object_name':'bookcase'}]}}
    plan = {'goal_selection':'fresh_anchor','shelf_tip_context':author_context(anchor,target_name='target',support_name='bookcase')}
    path = tmp_path/'groups/g/goal_plan.json'; queue = tmp_path/'queues/g/queue.json'
    def publish(value):
        c.write_json_atomic(path,value); c.write_json_atomic(queue,{'entries':[{'path':str(path),'sha256':c.file_digest(path)}]})
    publish(plan)
    result = audit_authored_source_context({'queue':str(queue)},anchor,SimpleNamespace(class_id=CLASS),tmp_path)
    assert c.file_digest(result['path']) == result['sha256']
    for mutation in ('drawn_start','partner_depth','missing'):
        bad = copy.deepcopy(plan)
        if mutation == 'drawn_start': bad['shelf_tip_context']['source_target_start']['position'] = [0.,.015726,.6]
        elif mutation == 'partner_depth': bad['shelf_tip_context']['spine_behind_edge_m'] = .044274
        else: bad.pop('shelf_tip_context')
        publish(bad)
        with pytest.raises(c.ContractError,match='fixed geometry reference'):
            audit_authored_source_context({'queue':str(queue)},anchor,SimpleNamespace(class_id=CLASS),tmp_path)
    source.write_text('{}')
    with pytest.raises(c.ContractError,match='summary changed'):
        audit_authored_source_context({'queue':str(queue)},anchor,SimpleNamespace(class_id=CLASS),tmp_path)


def test_explicit_own_group_replay_never_uses_previous_template_gate(tmp_path):
    import pytest
    from scripts.campaign_20261110 import contracts as c
    from scripts.campaign_20261110.controller_held import claim_gate_for_record
    allocation = {'key':'class','jobid':'123','job_dir':str(tmp_path/'out/current')}
    record = {'phase':'eval_standard'}
    previous = {'eval_standard':str(tmp_path/'older/gate.json')}
    assert claim_gate_for_record({},record,allocation,previous,tmp_path) == previous['eval_standard']
    spec = {'require_own_group_gate':True}
    assert claim_gate_for_record(spec,record,allocation,previous,tmp_path) is None
    own = tmp_path/'validation/open_loop/class/123/gate.json'
    c.write_json_atomic(own,{'source_job':str(tmp_path/'out/older')})
    with pytest.raises(c.ContractError,match='another job'):
        claim_gate_for_record(spec,record,allocation,previous,tmp_path)
    c.write_json_atomic(own,{'source_job':allocation['job_dir']})
    assert claim_gate_for_record(spec,record,allocation,previous,tmp_path) == str(own)


def counts(**kwargs):
    return {"evaluation": {}, "train_buckets": {}, "class_remaining": 1000, "task_remaining": 4000, **kwargs}


def test_unbucketed_pair_eval_continues_to_training_with_new_phase_gate():
    template = SimpleNamespace(task="inter_object_relation_satisfaction-ontop_small_scale", split="eval_standard")
    assert next_work(template, counts())["groups"] == 6
    assert next_work(template, counts(evaluation={0: 10})) is None
    assert next_work(template, counts(evaluation={0: 10}), training_validated=True) is None
    assert next_work(template, counts(), mode="train_only") is None


def test_holdouts_never_spill_into_training_and_count_three_members_per_group():
    model = SimpleNamespace(task="inter_object_relation_satisfaction-under_small_scale", split="eval_model")
    assert next_work(model, counts())["groups"] == 3
    assert next_work(model, counts(evaluation={0: 9})) is None
    assert next_work(model, counts(), mode="train_only") is None
    category = SimpleNamespace(task=model.task, split="eval_category")
    assert next_work(category, counts())["groups"] == 2
    assert next_work(category, counts(evaluation={0: 4})) is None


def test_eval_only_preserves_pending_capacity_and_never_launches_train_only():
    template = SimpleNamespace(task="inter_object_relation_satisfaction-inside_small_scale",
                               class_id="inside", split="eval_standard")
    assert next_work(template, counts(evaluation={0: 9}), mode="eval_only")["groups"] == 1
    reserved = reserve_pending_counts(counts(evaluation={0: 9}), template,
        [{"task": template.task, "class_id": "inside", "phase": "eval_standard", "bucket": 0, "groups": 1}])
    assert next_work(template, reserved, mode="eval_only") is None
    assert next_work(template, counts(evaluation={0: 10}), mode="eval_only", training_validated=True) is None


def test_strict_joint_buckets_and_soft_training_use_their_own_weights():
    template = SimpleNamespace(task="configuration_matching-articulation_close_small_scale", split="eval_standard")
    assert next_work(template, counts(evaluation={0: 5, 1: 2}))["groups"] == 3
    assert next_work(template, counts(evaluation={0: 10})) is None
    result = next_work(template, counts(evaluation={0: 5, 1: 5}, train_buckets={0: 100, 1: 0}), training_validated=True)
    assert result is None
    assert next_work(template, counts(class_remaining=0), mode="train_only") is None
    assert ranges(template) == ((0, .5), (.5, 1))


def test_pending_generic_groups_reserve_global_task_and_class_capacity():
    template = SimpleNamespace(task="disembedding", class_id="class", split="eval_standard")
    pending = [{"task": "disembedding", "class_id": "class", "phase": "eval_standard", "bucket": 1, "groups": 4},
               {"task": "disembedding", "class_id": "another", "phase": "train", "bucket": 0, "groups": 6}]
    result = reserve_pending_counts(counts(), template, pending)
    assert result["evaluation"] == {1: 4}
    assert result["class_remaining"] == 988 and result["task_remaining"] == 3988
    assert next_work(template, result)["bucket"] == 0


def test_training_cannot_consume_unfinished_standard_evaluation_siblings():
    template = SimpleNamespace(task="inter_object_relation_satisfaction-ontop_small_scale", split="eval_standard")
    assert next_work(template, counts(class_remaining=4, evaluation_sibling_reserve=2),
                     training_validated=True, mode="train_only") is None
    result = next_work(template, counts(class_remaining=5, evaluation_sibling_reserve=2),
                       training_validated=True, mode="train_only")
    assert result is None  # Standalone production is blocked.
    assert next_work(template, counts(class_remaining=1, evaluation_sibling_reserve=0),
                     training_validated=True, mode="train_only") is None


def test_retired_runtime_reconciles_existing_work_but_never_authors_more(tmp_path, monkeypatch):
    import pytest
    from dataclasses import asdict
    from scripts.campaign_20261110 import controller_held as held, controller_migration, contracts as c
    from scripts.campaign_20261110.tests.test_goal_groups import GoalGroupControllerTests
    _,_,_,template,_=GoalGroupControllerTests().fixture(tmp_path)
    config=tmp_path/'config.json';gate=tmp_path/'gate.json'
    c.write_json_atomic(config,{'template':asdict(template),'validation_key':{}})
    c.write_json_atomic(gate,{'success':True})
    spec={'class_id':template.class_id,'seed_config':str(config),'seed_config_sha256':c.file_digest(config),
          'source_gate':str(gate),'source_gate_sha256':c.file_digest(gate),'snapshot':'/native'}
    record={'queue':'q','phase':template.split,'status':'external_submission','index':1,'count':3}
    state={'queues':[record]};claims=[]
    monkeypatch.setattr(controller_migration,'controller_check',lambda *a,**k:None)
    monkeypatch.setattr(held,'reconcile',lambda *a,**k:{'key':template.class_id,'jobid':'job','job_dir':'/owned/job'})
    monkeypatch.setattr(held,'job_terminal',lambda *a:True)
    monkeypatch.setattr(held,'claim_finished_groups',lambda *a,**k:claims.append(k['job_terminal']))
    monkeypatch.setattr(held,'queue_closures',lambda *a:{})
    monkeypatch.setattr(held,'record_runtime_failure',lambda *a:None)
    adapter=SimpleNamespace(supports=lambda task:True,read_counts=lambda *a:pytest.fail('retired runtime must not plan'))
    assert held.advance(spec,state,{'retired_classes':[template.class_id]},tmp_path,[],{},1,lambda:None,adapter=adapter)==0
    assert claims==[True] and record['status']=='complete'
    assert state['status']=='retired_runtime_no_new_groups'


def test_fresh_alignment_anchor_confirms_frozen_policy_without_rekeying(tmp_path):
    from dataclasses import replace
    import pytest
    from scripts.campaign_20261110.tests.test_goal_groups import GoalGroupControllerTests
    from scripts.campaign_20261110.goal_groups import with_alignment_buckets
    from scripts.campaign_20261110.contracts import ContractError
    _, _, _, original, _ = GoalGroupControllerTests().fixture(tmp_path)
    template = replace(original, task="alignment-world_vertical")
    first = with_alignment_buckets(template, [15, 25], source={"path": "first", "field": "range", "sha256": "a" * 64})
    confirmed = with_alignment_buckets(first, [15, 25], source={"path": "fresh", "field": "range", "sha256": "b" * 64})
    assert confirmed is first and confirmed.fingerprint == first.fingerprint
    with pytest.raises(ContractError, match="policy changed"):
        with_alignment_buckets(first, [10, 25], source={"field": "range", "sha256": "c" * 64})


def test_category_queue_reconciles_canonical_submit_key_without_changing_category_class(tmp_path, monkeypatch):
    from dataclasses import asdict, replace
    from scripts.campaign_20261110 import controller_held as held, controller_migration, contracts as c
    from scripts.campaign_20261110.tests.test_goal_groups import GoalGroupControllerTests
    _, _, _, original, _ = GoalGroupControllerTests().fixture(tmp_path)
    template = replace(original, task='inter_object_relation_satisfaction-ontop_small_scale',
                       class_id='inter_object_relation_satisfaction-ontop_small_scale-rubber_eraser-notepad', split='eval_category')
    config = tmp_path/'category_config.json'; gate = tmp_path/'gate.json'
    c.write_json_atomic(config, {'template': asdict(template), 'validation_key': {}})
    c.write_json_atomic(gate, {'success': True})
    key = 'inter_object_relation_satisfaction-ontop_small_scale_01'
    record = {'queue': 'queue', 'phase': 'eval_category', 'status': 'external_submission',
              'command': ['python', 'submit_fast.py', 'submit', key, '3'], 'index': 100, 'count': 3}
    spec = {'class_id': template.class_id, 'seed_config': str(config), 'seed_config_sha256': c.file_digest(config),
            'source_gate': str(gate), 'source_gate_sha256': c.file_digest(gate), 'snapshot': '/snapshot'}
    observed = []
    monkeypatch.setattr(controller_migration, 'controller_check', lambda *args, **kwargs: None)
    monkeypatch.setattr(held, 'reconcile', lambda rows, **kwargs: observed.append(kwargs) or None)
    state = {'queues': [record]}
    assert held.advance(spec, state, {}, tmp_path, [], {}, 1, lambda: None,
                        adapter=SimpleNamespace(supports=lambda task: True)) == 0
    assert observed[0]['key'] == key and spec['class_id'] == template.class_id
