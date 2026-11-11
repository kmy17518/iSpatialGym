import unittest

from scripts.campaign_20261110.infra_vast.fleet_policy import rental_ceiling


class FleetPolicyTests(unittest.TestCase):
    def test_ready_work_can_exceed_old_fleet_cap(self):
        self.assertEqual(rental_ceiling({'fleet_scaling_mode': 'ready_budget', 'max_instances': 45}, 80, 350), 430)

    def test_no_uncovered_work_adds_no_rental(self):
        self.assertEqual(rental_ceiling({'fleet_scaling_mode': 'ready_budget'}, 80, 0), 80)

    def test_fixed_mode_is_preserved_for_other_campaigns(self):
        self.assertEqual(rental_ceiling({'max_instances': 3}, 1, 20), 3)

    def test_invalid_mode_or_demand_fails_closed(self):
        for cfg, live, need in [({'fleet_scaling_mode': 'unknown'}, 0, 1), ({'fleet_scaling_mode': 'ready_budget'}, 0, -1), ({'fleet_scaling_mode': 'ready_budget'}, 0, True)]:
            with self.assertRaises(ValueError):
                rental_ceiling(cfg, live, need)


if __name__ == '__main__':
    unittest.main()
