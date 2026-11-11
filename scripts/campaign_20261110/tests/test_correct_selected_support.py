from copy import deepcopy
import unittest

from scripts.campaign_20261110.correct_selected_support import replace_support


class ExactSupportCorrectionTest(unittest.TestCase):
    def fixture(self):
        old = dict(id='old', category='table', model='m', asset_sha256='asset',
                   default_scale=[1, 1, 1], sampled_uniform_factors=[.8, 1.2],
                   absolute_scales=[[1, 1, .8], [1, 1, 1.2]])
        new = {**old, 'id': 'new', 'default_scale': [.99, 1, 1],
               'absolute_scales': [[.99, 1, .8], [.99, 1, 1.2]]}
        support = dict(scene='s', object='table_0', height_slot=0,
                       support_profile='old', selected_scale=[1, 1, .8])
        role = dict(role='support', model_key='table/m', asset_sha256='asset',
                    scale_rule='z_only_support_profile', applied_scale=[1, 1, .8])
        request = dict(id='old_request', kind='map', inputs={'support': support}, status='blocked')
        untouched = dict(id='other', kind='geometry', inputs={'model': 'different'}, status='verified')
        config = dict(id='slot0', configuration_sha256=None, dependencies=['old_request', 'other'], roles=[role])
        return dict(artifact_requests=[request, untouched], configurations=[config]), dict(
            old_profile=old, new_profile=new, binding={'scene': 's', 'name': 'table_0'},
            correction_ref={'path': '/proof', 'sha256': 'proof'})

    def test_new_request_retains_slot_and_unrelated_proof(self):
        source, kwargs = self.fixture()
        preserved = deepcopy(source)
        result, receipt = replace_support(source, **kwargs)
        self.assertEqual(source, preserved)
        self.assertEqual(result['configurations'][0]['id'], 'slot0')
        self.assertNotEqual(result['configurations'][0]['dependencies'][0], 'old_request')
        self.assertEqual(result['artifact_requests'][1], source['artifact_requests'][1])
        self.assertEqual(receipt['physical_artifacts_reinterpreted'], 0)

    def test_materialized_and_verified_inputs_fail_closed(self):
        for kind in ('config', 'proof'):
            source, kwargs = self.fixture()
            if kind == 'config': source['configurations'][0]['configuration_sha256'] = 'existing'
            else: source['artifact_requests'][0]['status'] = 'verified'
            with self.assertRaises(ValueError): replace_support(source, **kwargs)

    def test_changed_draw_or_slot_is_rejected(self):
        source, kwargs = self.fixture()
        kwargs['new_profile']['sampled_uniform_factors'] = [.9, 1.2]
        with self.assertRaises(ValueError): replace_support(source, **kwargs)
        source, kwargs = self.fixture()
        source['artifact_requests'][0]['inputs']['support']['selected_scale'] = [1, 1, .9]
        with self.assertRaises(ValueError): replace_support(source, **kwargs)


if __name__ == '__main__': unittest.main()
