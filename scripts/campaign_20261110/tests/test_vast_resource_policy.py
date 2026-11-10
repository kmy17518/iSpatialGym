import unittest
from scripts.campaign_20261110.infra_vast.worker.resource_policy import machine_reservation, select_cpus, duplicate_worker_gpus, bad_machines, host_startup_failure


class ResourcePolicyTests(unittest.TestCase):
    def test_account_draining_does_not_poison_healthy_host_eligibility(self):
        historical = 'account funding drain; no assigned/staging or heartbeat-running jobs'
        events=[{'ev':'bad_machine','machine_id':1,'reason':historical},
                {'ev':'bad_machine','machine_id':2,'reason':'no heartbeat after 55 min (running)'},
                {'ev':'bad_machine','machine_id':3,'reason':historical},
                {'ev':'bad_machine','machine_id':3,'reason':'no NVIDIA graphics device'}]
        self.assertEqual(bad_machines(events),{2,3})
        self.assertFalse(host_startup_failure(historical))
        self.assertFalse(host_startup_failure('booting with an empty queue for 15 min'))
        self.assertTrue(host_startup_failure('boot not finished after 55 min'))
        self.assertTrue(host_startup_failure('no heartbeat after 55 min (running)'))

    def setUp(self):
        self.config = {'cpus_per_job':9, 'ram_per_job_gb':40, 'max_contracts_per_machine':4}
        self.offer = {'machine_id':1, 'num_gpus':1, 'gpu_ids':[22], 'cpu_cores_effective':16, 'cpu_ram':64000}
        self.live = {'a':{'machine_id':1, 'gpu_ids':[21]}}
        self.hb = {'a':{'cpus':15, 'allowed_cpu_affinity':list(range(64))}}

    def test_quota_only_peers_receive_disjoint_pinned_cpus(self):
        value, reason = machine_reservation(self.offer,self.live,self.hb,{},self.config)
        self.assertIsNone(reason); self.assertEqual(value['cpu_affinity'],list(range(15,25)))
        self.live['b'] = {'machine_id':1,'gpu_ids':[22]}
        next_offer = dict(self.offer,gpu_ids=[23])
        newer, reason = machine_reservation(next_offer,self.live,self.hb,{'b':value},self.config)
        self.assertIsNone(reason); self.assertEqual(newer['cpu_affinity'],list(range(25,35)))

    def test_duplicate_gpu_cap_missing_identity_and_cpu_exhaustion_fail_closed(self):
        for modification, reason in [({'gpu_ids':[21]},'duplicate_physical_gpu'), ({'gpu_ids':[]},'single_gpu_identity_required'),
                                     ({'cpu_cores_effective':9},'insufficient_guaranteed_resources')]:
            with self.subTest(reason=reason):
                self.assertEqual(machine_reservation(dict(self.offer,**modification),self.live,self.hb,{},self.config)[1],reason)
        self.assertEqual(machine_reservation(self.offer,self.live,self.hb,{},dict(self.config,max_contracts_per_machine=1))[1],'machine_contract_cap')
        self.hb['a']['allowed_cpu_affinity'] = list(range(16))
        self.assertEqual(machine_reservation(self.offer,self.live,self.hb,{},self.config)[1],'insufficient_disjoint_cpus')
        self.hb = {}
        self.assertEqual(machine_reservation(self.offer,self.live,self.hb,{},self.config)[1],'peer_cpu_reservation_unknown')

    def test_worker_never_replaces_unavailable_requested_cpus_with_overlapping_defaults(self):
        self.assertEqual(select_cpus(range(64),16,range(15,25)),list(range(15,25)))
        self.assertEqual(select_cpus(range(16),16,range(15,25)),[])
        self.assertEqual(select_cpus(range(64),8,range(15,25)),[])
        self.assertEqual(select_cpus(range(64),16,[]),list(range(16)))

    def test_uuid_duplicate_and_multiple_visible_gpu_are_not_assignable(self):
        live = {'a':{'machine_id':1},'b':{'machine_id':1},'c':{'machine_id':2}}
        hbs = {'a':{'gpu_uuid':'same','n_gpus':1},'b':{'gpu_uuid':'same','n_gpus':1},'c':{'n_gpus':2}}
        self.assertEqual(duplicate_worker_gpus(live,hbs),{'a','b','c'})


if __name__ == '__main__': unittest.main()
