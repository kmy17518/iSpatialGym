import unittest

from scripts.campaign_20261110.audit_contact_dependencies import verify_object_identity


class NativeContactIdentityTest(unittest.TestCase):
    def setUp(self):
        self.obj = dict(category='box_of_cereal', model='iucshm', scale=[.8] * 3)

    def test_float32_and_serialized_native_scale_are_identical(self):
        proof = verify_object_identity(self.obj, {**self.obj, 'scale': [.800000011920929] * 3})
        self.assertEqual(proof['canonical_scale'], [.8] * 3)
        self.assertNotEqual(proof['library_scale'], proof['requested_scale'])

    def test_different_scale_or_asset_remains_rejected(self):
        for change in ({'scale': [.81] * 3}, {'scale': [.8, .8, .81]},
                       {'category': 'box_of_rice'}, {'model': 'dkhgxn'}):
            with self.subTest(change=change), self.assertRaisesRegex(AssertionError, 'differs'):
                verify_object_identity(self.obj, {**self.obj, **change})

    def test_invalid_scale_is_rejected(self):
        for scale in ([.8, .8], [float('nan')] * 3, [float('inf')] * 3):
            with self.subTest(scale=scale), self.assertRaises(AssertionError):
                verify_object_identity(self.obj, {**self.obj, 'scale': scale})


if __name__ == '__main__':
    unittest.main()
