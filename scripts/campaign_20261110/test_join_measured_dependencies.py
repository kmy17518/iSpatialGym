import copy
import unittest

from scripts.campaign_20261110.join_measured_dependencies import join, map_matches, support_matches


class MeasuredJoinTests(unittest.TestCase):
    def setUp(self):
        self.support = {'scene': 'room', 'object': 'table', 'height_slot': 0, 'support_profile': 'profile',
                        'selected_scale': [1, 1, .9], 'source_geometry_sha256': 'geometry'}
        self.proof = {'probe': {'geometry_admitted': True, 'conservative_overlap_blockers': [],
                               'scene': 'room', 'object': 'table', 'slot': 0, 'profile_id': 'profile',
                               'absolute_scale': [1, 1, .9], 'actual_height_key': '0.7182',
                               'measured_height_above_floor': .718225},
                      'binding': {'source_geometry_sha256': 'geometry'},
                      'probe_ref': {'path': 'probe', 'sha256': 'probehash'},
                      'stable_scene': {'path': 'scene', 'sha256': 'scenehash'}}
        self.request = {'id': 'map', 'kind': 'tabletop_reach_map', 'required_by_classes':
                        ['inter_object_relation_satisfaction-ontop_small_scale_01'],
                        'inputs': {'support': self.support, 'arm': 'right',
                                   'target': {'category': 'food', 'model': 'model', 'dataset': 'dataset',
                                              'asset_sha256': 'asset', 'applied_scale': [1.1] * 3}},
                        'status': 'blocked', 'physical_artifact_id': None}
        self.row = {'status': 'verified', 'support': self.proof,
                    'job': {'category': 'food', 'model': 'model', 'dataset': 'dataset',
                            'asset_sha256': 'asset', 'scale': [1.1] * 3},
                    'map': {'native_reader_passed': True, 'positive_cells': 10, 'arm': 'right',
                            'object_key': 'native', 'rest_id': 'r0', 'height': .7182},
                    'native_library_identity_key': 'native', 'positive_rest_ids': ['r0'],
                    'consumer': {'classes': self.request['required_by_classes']},
                    'library': {'path': 'lib', 'sha256': 'libhash'}, 'generator': {'sha256': 'generator'}}

    def test_exact_match_and_each_identity_mismatch(self):
        self.assertTrue(map_matches(self.request, self.row))
        for key, value in [('scene', 'other'), ('object', 'other'), ('height_slot', 1),
                           ('selected_scale', [1, 1, .90001]), ('source_geometry_sha256', 'other')]:
            changed = copy.deepcopy(self.support); changed[key] = value
            self.assertFalse(support_matches(changed, self.proof), key)
        for path, value in [(('job', 'asset_sha256'), 'other'), (('job', 'scale'), [1.10001] * 3),
                            (('map', 'arm'), 'left'), (('map', 'height'), .7183),
                            (('map', 'positive_cells'), 0), (('map', 'object_key'), 'other')]:
            changed = copy.deepcopy(self.row); changed[path[0]][path[1]] = value
            self.assertFalse(map_matches(self.request, changed), path)

    def test_table_height_cannot_clear_tray_or_attachment_query(self):
        for cls in ['disembedding_09', 'configuration_matching-block_design_01',
                    'configuration_matching-attachment_01']:
            request = copy.deepcopy(self.request); request['required_by_classes'] = [cls]
            row = copy.deepcopy(self.row); row['consumer']['classes'] = [cls]
            self.assertFalse(map_matches(request, row))

    def test_all_positive_rests_required_and_layout_stays_pending(self):
        support = {'id': 'support', 'kind': 'native_support_geometry_and_height', 'inputs': self.support,
                   'status': 'blocked', 'physical_artifact_id': None}
        layout = {'id': 'layout', 'kind': 'native_maximum_layout_fit', 'status': 'blocked',
                  'physical_artifact_id': None}
        manifest = {'artifact_requests': [self.request, support, layout],
                    'configurations': [{'class_id': 'class', 'dependencies': ['map', 'support', 'layout'],
                                        'status': 'blocked'}], 'per_class': [{'class_id': 'class'}], 'summary': {}}
        evidence = {'observed_at': 'now', 'supports': [self.proof], 'maps': [self.row]}
        result = join(manifest, evidence, {'path': 'audit', 'sha256': 'auditsha'})
        self.assertEqual(result['summary']['dependency_statuses'], {'verified': 2, 'blocked': 1})
        self.assertEqual(result['summary']['fully_served_configurations'], 0)
        self.assertEqual(manifest['artifact_requests'][0]['status'], 'blocked')
        evidence['maps'][0]['positive_rest_ids'] = ['r0', 'r1']
        result = join(manifest, evidence, {'path': 'audit', 'sha256': 'auditsha'})
        self.assertEqual(result['artifact_requests'][0]['status'], 'blocked')


if __name__ == '__main__':
    unittest.main()
