import unittest

from scripts.campaign_20261110.infra_vast.fleet_policy import rental_ceiling, activated_queue_reserve


class FleetPolicyTests(unittest.TestCase):
    def test_queue_commitments_retain_assigned_work_but_not_terminal_or_inactive(self):
        config={'dispatch_job_ids':['1',2,'3'],'max_dph_gpu':2}
        states={'1':{'status':'queued','time_s':1800},'2':{'status':'assigned','time_s':3600},
                '3':{'status':'synced','time_s':7200},'4':{'status':'queued','time_s':10000}}
        self.assertEqual(activated_queue_reserve(config,states),3)
        states['1']['status']='failed'
        self.assertEqual(activated_queue_reserve(config,states),2)
        for invalid in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                activated_queue_reserve(config, {'1':{'status':'queued','time_s':invalid}})

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
