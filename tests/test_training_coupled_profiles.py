import copy
import hashlib
import random
import unittest

from scripts.campaign_20261110.training_coupled_profiles import candidate_profile, coupled_profile, digest


class CoupledProfilesTests(unittest.TestCase):
    def setUp(self):
        self.target = dict(category="lid", model="one", asset_sha256="a" * 64, default_scale=[1., .9, .8])
        self.partner = dict(category="pot", model="two", asset_sha256="b" * 64, default_scale=[.7, .6, .5])

    def test_attachment_preserves_existing_pair_seed_and_scales(self):
        identity = {"target": self.target, "partner": self.partner}
        key = digest(identity)
        rng = random.Random(int(digest({"campaign": "campaign_20261110", "kind": "attachment_coupled_scale", "pair": key}), 16))
        factors = [rng.uniform(.8, 1.2), rng.uniform(.8, 1.2)]
        actual = coupled_profile("attachment", identity)
        self.assertEqual(actual["id"], key)
        self.assertEqual(actual["sampled_uniform_factors"], factors)
        self.assertEqual(actual["absolute_scales"], {r: [[round(v * f, 4) for v in obj["default_scale"]] for f in factors] for r, obj in identity.items()})

    def test_every_board_role_uses_same_factor_and_scale_rounding(self):
        profile = coupled_profile("block_design", {"target": self.target, "board": self.partner})
        for role, obj in profile["objects"].items():
            for slot, factor in enumerate(profile["sampled_uniform_factors"]):
                self.assertTrue(all(abs(s - d * factor) <= .00005000001 for s, d in zip(profile["absolute_scales"][role][slot], obj["default_scale"])))
        self.assertEqual(profile["physical_validation"], "unverified")

    def test_context_identity_changes_puzzle_profile(self):
        first = coupled_profile("puzzle", {"target": self.target, "distractors:piece/two": self.partner})
        changed = copy.deepcopy(self.partner)
        changed["asset_sha256"] = "c" * 64
        second = coupled_profile("puzzle", {"target": self.target, "distractors:piece/two": changed})
        self.assertNotEqual(first["id"], second["id"])

    def test_unknown_native_defaults_do_not_produce_a_profile(self):
        candidate = dict(task="configuration_matching-attachment", target=dict(category="lid", model="one"), partner=dict(category="pot", model="two"))
        assets = {"lid/one": dict(self.target, native_default_geometry={"default_scale": self.target["default_scale"]}),
                  "pot/two": dict(self.partner, native_default_geometry=None)}
        self.assertEqual(candidate_profile(candidate, assets), (None, {}))

    def test_puzzle_contexts_couple_but_tray_does_not(self):
        target = dict(category="jigsaw_puzzle_piece", model="target")
        piece = dict(role="distractors", category="jigsaw_puzzle_piece", model="context")
        candidate = dict(task="disembedding", target=target, extra_roles=[piece, dict(role="source", category="tray", model="one")])
        assets = {"jigsaw_puzzle_piece/target": dict(self.target, category="jigsaw_puzzle_piece", model="target", native_default_geometry={"default_scale": [1., 1., 1.]}),
                  "jigsaw_puzzle_piece/context": dict(self.partner, category="jigsaw_puzzle_piece", model="context", native_default_geometry={"default_scale": [1., 1., 1.]})}
        profile, roles = candidate_profile(candidate, assets)
        self.assertEqual(len(profile["objects"]), 2)
        self.assertNotIn(("source", "tray/one"), roles)
        self.assertEqual(profile["absolute_scales"]["target"], profile["absolute_scales"]["distractors:jigsaw_puzzle_piece/context"])

    def test_nonfinite_default_rejected(self):
        broken = dict(self.partner, default_scale=[1., float("nan"), 1.])
        with self.assertRaises(ValueError):
            coupled_profile("attachment", {"target": self.target, "partner": broken})


if __name__ == "__main__":
    unittest.main()
