import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ispatialgym.sampling.model_pool import ModelPool, active_scene, check_policy, signature


class Backend:
    def __init__(self):
        self.spawned = {}; self.preloads = 0; self.parked = set(); self.reset = []
    def preload_pool_objects(self, specs):
        self.preloads += 1
        self.parked = {s['name'] for s in specs}
        return {s['name']: s['name'] for s in specs}
    def reset_pool_object(self, obj):
        self.parked.remove(obj); self.reset.append(obj)
    def park_pool_object(self, obj):
        self.parked.add(obj)
    def observe_model_pool(self, members, active):
        return {'parked': sorted(self.parked)}


class PoolContractTests(unittest.TestCase):
    """Contract failures before expensive native runs; these are not physics proof."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.manifest = {'entries': [dict(category='bottle', model=m, dataset='assets', scale=[1., 1., 1.], copies=1)
                                     for m in ['a', 'b']], 'batch_id': 'two_models'}
        self.backend = Backend()
        self.pool = ModelPool(self.backend, self.manifest, {'sha256': 'exact'}, Path(self.tmp.name)/'evidence.json')

    def test_two_model_switch_reuses_objects_and_never_lazy_spawns(self):
        a = self.pool.acquire('target', 'bottle', 'a', 'assets', [1., 1., 1.])
        self.pool.release_template()
        b = self.pool.acquire('target', 'bottle', 'b', 'assets', [1., 1., 1.])
        self.pool.release_template()
        again = self.pool.acquire('target', 'bottle', 'a', 'assets', [1., 1., 1.])
        self.assertNotEqual(a, b); self.assertEqual(a, again)
        self.assertEqual(self.backend.preloads, 1)
        self.assertEqual(set(self.backend.spawned), {a}); self.assertEqual(self.backend.parked, {b})
        with self.assertRaisesRegex(ValueError, 'no unused exact'):
            self.pool.acquire('partner', 'bottle', 'a', 'assets', [1., 1., 1.])
        with self.assertRaisesRegex(ValueError, 'no unused exact'):
            self.pool.acquire('partner', 'bottle', 'b', 'assets', [.99, 1., 1.])
        evidence = json.loads((Path(self.tmp.name)/'evidence.json').read_text())
        self.assertFalse(evidence['physical_validation_passed']); self.assertEqual(evidence['benchmark_credit'], 0)

    def test_export_preserves_context_and_rejects_inactive_cross_reference(self):
        scene = {'objects_info': {'init_info': {'campaign_pool_a': {}, 'campaign_pool_b': {}, 'wall': {'physical': 'unchanged'}}},
                 'state': {'registry': {'object_registry': {'campaign_pool_a': {}, 'campaign_pool_b': {}, 'wall': {'pose': [1, 2, 3]}}}},
                 'metadata': {'task': {'inst_to_name': {'target': 'campaign_pool_a'}}}}
        result = active_scene(scene, ['campaign_pool_a', 'campaign_pool_b'], ['campaign_pool_a'])
        self.assertEqual(set(result['objects_info']['init_info']), {'campaign_pool_a', 'wall'})
        self.assertEqual(result['state']['registry']['object_registry']['wall'], {'pose': [1, 2, 3]})
        self.assertIn('campaign_pool_b', scene['objects_info']['init_info'])
        scene['metadata']['task']['attachment'] = 'campaign_pool_b'
        with self.assertRaisesRegex(ValueError, 'remains referenced'):
            active_scene(scene, ['campaign_pool_a', 'campaign_pool_b'], ['campaign_pool_a'])

    def test_missing_default_scale_cannot_be_guessed(self):
        with self.assertRaisesRegex(ValueError, 'unique measured'):
            self.pool.acquire('target', 'bottle', 'a', 'assets', None)
        with self.assertRaisesRegex(ValueError, 'explicit exact'):
            signature('bottle', 'a', 'assets', [1., float('nan'), 1.])

    def test_new_reservation_invalidates_inactive_pool_member(self):
        value = {'campaign_root': '/tmp', 'reservation_policy_sha256': 'rev', 'scene': 'S', 'rooms': ['R'],
                 'entries': [{'category': 'bottle', 'model': 'b', 'asset_sha256': 'bytes'}]}
        active = ({'policy_sha256': 'rev'}, {'category_holdout_rooms': [], 'train_exclude_models': ['bottle/b'],
                  'category_holdout_categories': []}, {'reserved_asset_hashes': ['bytes']})
        with patch('scripts.campaign_20261110.reservation_policy.active_policy', return_value=active):
            with self.assertRaisesRegex(ValueError, 'held-out model'):
                check_policy(value)


if __name__ == '__main__': unittest.main()
