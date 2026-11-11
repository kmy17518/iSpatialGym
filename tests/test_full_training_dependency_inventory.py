import copy
import unittest

from scripts.campaign_20261110.training_dependency_inventory import build, slots


class FullTrainingInventoryTests(unittest.TestCase):
    def setUp(self):
        self.asset = dict(model_key="cup/model", category="cup", model="model", dataset="behavior-1k-assets",
                          ready=True, readiness_source="catalog", asset_sha256="a" * 64, blockers=[],
                          native_default_geometry=dict(default_scale=[1., 1., 1.], source={"path": "native.json", "sha256": "b" * 64}))
        self.candidate = dict(candidate_id="candidate", class_id="class", task="object_scaling",
                              target={"category": "cup", "model": "model"}, partner=None, extra_roles=[],
                              anchor={"scene": "scene", "rooms": ["room"]})

    def inventory(self, candidates=None, asset=None):
        return build(candidates or [self.candidate], {"cup/model": asset or self.asset},
                     {"bindings": {}, "profiles": {}}, {"category_holdout_rooms": []},
                     sources={}, generator_inputs={})

    def test_two_exact_scale_slots_and_all_consumers_retained(self):
        result = self.inventory()
        self.assertEqual(result["summary"]["selected_configuration_slots"], 2)
        roles = [r["roles"][0] for r in result["configurations"]]
        self.assertNotEqual(roles[0]["applied_scale"], roles[1]["applied_scale"])
        geometry = [r for r in result["artifact_requests"] if r["kind"] == "native_asset_default_geometry"]
        self.assertEqual(len(geometry), 1)
        self.assertEqual(len(geometry[0]["consumers"]), 2)

    def test_excluded_proposals_stay_in_required_denominator(self):
        asset = copy.deepcopy(self.asset)
        asset["blockers"] = ["current_global_holdout_exclusion"]
        result = self.inventory(asset=asset)
        self.assertEqual(len(result["configurations"]), 2)
        self.assertTrue(all("current_global_holdout_exclusion" in r["blockers"] for r in result["configurations"]))
        self.assertFalse(any(r["status"] == "verified" for r in result["artifact_requests"]))

    def test_unknown_scale_never_becomes_a_physical_library_identity(self):
        asset = copy.deepcopy(self.asset)
        asset["native_default_geometry"] = None
        result = self.inventory(asset=asset)
        libraries = [r for r in result["artifact_requests"] if "grasp_library" in r["kind"]]
        self.assertTrue(libraries)
        self.assertTrue(all(r["physical_artifact_id"] is None for r in libraries))

    def test_camera_has_no_grasp_or_map_requirement(self):
        candidate = copy.deepcopy(self.candidate)
        candidate["task"] = "camera_relocalization-obstructed"
        result = self.inventory([candidate])
        self.assertEqual(len(result["configurations"]), 1)
        self.assertEqual(sum(r["status"] == "not_applicable" for r in result["artifact_requests"]), 1)
        self.assertFalse(any("grasp_library" in r["kind"] or r["kind"].endswith("reach_map") for r in result["artifact_requests"]))

    def test_axial_table_has_two_height_slots_structural_wall_does_not(self):
        candidate = copy.deepcopy(self.candidate)
        candidate["task"] = "alignment-axial"
        candidate["anchor"]["category"] = "table"
        self.assertEqual(len(slots(candidate)), 4)
        candidate["task"] = "alignment-world_vertical"
        candidate["anchor"]["category"] = "walls"
        self.assertEqual(len(slots(candidate)), 2)

    def test_coupled_scale_is_not_independent_per_model_draw(self):
        candidate = copy.deepcopy(self.candidate)
        candidate["task"] = "configuration_matching-attachment"
        candidate["partner"] = candidate["target"]
        result = self.inventory([candidate])
        self.assertTrue(all(r["applied_scale"] is None for c in result["configurations"] for r in c["roles"]))
        self.assertFalse(result["summary"]["denominator_complete"])


if __name__ == "__main__":
    unittest.main()
