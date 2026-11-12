import json
from pathlib import Path

from scripts.campaign_20261110.inventory_specialized_consumers import inventory


def test_specialized_jobs_keep_scale_constraint_and_only_same_model_rest(tmp_path):
    root = tmp_path / 'campaign'; runtime = root / 'runtime'; output = root / 'inventory'
    customs = runtime / 'ispatialgym/assets/task_specs/isg_customs.json'; customs.parent.mkdir(parents=True)
    customs.write_text(json.dumps({'inter_object_relation_satisfaction-inside_small_scale-sunflower-vase-table': {
        'roles': {'target': {'category': 'sunflower', 'require': {'grasps': 'insert'},
                             'grasp_keys': {'sunflower/old': 'sunflower__old__r50'}}}}}))
    legacy = runtime / 'runlogs/grasp_library_20260905'; (legacy / 'results').mkdir(parents=True); (legacy / 'cands').mkdir()
    (legacy / 'results/sunflower__old__r50.json').write_text(json.dumps({'model': 'old', 'constraint': 'flower_stem_top'}))
    (legacy / 'cands/sunflower__old__r50.yaml').write_text('ispatialgym:\n  rest_pose: designated\n  rest_quat_wxyz: [0.9, 0.1, 0.2, 0.3]\n')
    rows = []
    for model in ['old', 'new']:
        rows.append({'id': model, 'kind': 'tabletop_grasp_library', 'required_by_classes': ['inter_object_relation_satisfaction-inside_03'],
                     'inputs': {'target': {'category': 'sunflower', 'model': model, 'dataset': 'assets', 'asset_sha256': 'a',
                                            'applied_scale': [1.1, 1.1, 1.1]}}, 'consumer_scopes': {'required': ['config_' + model]}})
    graph = root / 'graph.json'; graph.write_text(json.dumps({'rows': rows}))
    inventory(root, runtime, graph, output)
    wanted = list(json.loads((output / 'wanted.json').read_text()).values())
    assert len(wanted) == 2
    for row in wanted:
        assert row['job']['constraint'] == 'flower_stem_top'
        assert row['job']['scale'] == [1.1, 1.1, 1.1]
        assert row['consumer_binding']['registration_required']
        assert row['job']['legacy_key'] is None
        assert bool(row['job']['rest_hint_wxyz']) == (row['job']['model'] == 'old')
