import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from scripts.campaign_20261110 import dispatch_cluster_first as C
from scripts.campaign_20261110 import infra_dispatch as D


class ClusterFirstTests(unittest.TestCase):
    def setUp(self):
        self.payload={'backends':['minyeong_svl','shiyuc_svl','vast_4090']}
        self.workers={b:{'observed_at':1000,'phase':'polling'}for b in C.CLUSTER_BACKENDS}
        self.demand={'cpus':9,'memory_mb':48*1024,'excluded_nodes':['svl9']}
        self.observation={'state':'available','observed_at':1000,'jobs':[],
          'nodes':[{'NodeName':'svl8','State':'MIXED','Partitions':'svl',
            'CfgTRES':'cpu=72,gres/gpu=8,gres/gpu:titanrtx=8','AllocTRES':'cpu=63,gres/gpu=7',
            'CPUEfctv':'72','CPUAlloc':'63','RealMemory':'500000','AllocMem':'440000'}]}
    def assess(self):return C.assess(self.payload,self.observation,self.workers,self.demand,now=1001)
    def test_free_compatible_cluster_prevents_vast_claim(self):
        d=self.assess();self.assertFalse(d['allow_vast'])
        self.assertEqual(d['preferred_cluster_backends'],['minyeong_svl','shiyuc_svl'])
        self.assertEqual(d['accounts_checked'],['minyeong','shiyuc'])
    def test_secondary_remains_available_when_primary_at_cap(self):
        self.observation['jobs']=[['minyeong','svl','RUNNING','gres/gpu:titanrtx:1','normal','svl7',str(i)]for i in range(48)]
        self.assertEqual(self.assess()['preferred_cluster_backends'],['shiyuc_svl'])
    def test_real_independent_pending_work_does_not_get_queue_age_grace(self):
        self.observation['jobs']=[[u,'svl','PENDING','gres/gpu:1','normal','(Priority)',u+'-pending']for u in ('minyeong','shiyuc')]
        d=self.assess();self.assertTrue(d['allow_vast']);self.assertEqual(d['minimum_pending_seconds'],0)
    def test_dependencies_are_not_misclassified_as_resource_waits(self):
        self.observation['jobs']=[[u,'svl','PENDING','gres/gpu:1','normal','(Dependency)',u+'-pending']for u in ('minyeong','shiyuc')]
        self.assertFalse(self.assess()['allow_vast'])
    def test_gpu_inventory_requires_cpu_memory_model_state_and_exclusions(self):
        for key,value in [('CPUAlloc','70'),('AllocMem','490000'),('NodeName','svl9'),
                          ('State','IDLE+DRAIN'),('CfgTRES','gres/gpu=8,gres/gpu:2080ti=8')]:
            with self.subTest(key=key):
                n=self.observation['nodes'][0];old=n[key];n[key]=value
                self.assertTrue(self.assess()['allow_vast']);n[key]=old
    def test_unavailable_probe_and_owner_are_explicit_not_free_capacity(self):
        self.observation['state']='unavailable'
        d=self.assess();self.assertTrue(d['allow_vast']);self.assertEqual(d['reason'],'cluster_availability_unavailable')
        self.observation['state']='available';self.workers={}
        d=self.assess();self.assertTrue(d['allow_vast'])
        self.assertEqual(d['backends'][0]['reason'],'cluster_dispatcher_unobserved')
    def test_transport_restrictions_are_owned_not_silently_overridden(self):
        self.payload['backends']=['vast_4090'];d=self.assess();self.assertTrue(d['allow_vast'])
        self.assertTrue(all(r['reason']=='intent_cluster_transport_not_enabled'for r in d['backends']))
    def test_no_new_receipt_or_submission_when_cluster_can_claim(self):
        with tempfile.TemporaryDirectory()as tmp:
            root=Path(tmp);path=root/'intent.json'
            payload={'token':'test','command':['unused'],'identity':{}}
            path.write_text(json.dumps(payload))
            with patch('scripts.campaign_20261110.production_gate.require_dispatch'),patch.object(D,'find_allocation',return_value=None),patch.object(C,'check',return_value=self.assess()),patch.object(D,'routed_command')as route:
                D.execute(root,path,'vast_4090')
            route.assert_not_called();self.assertFalse((root/'dispatch/receipts/test.json').exists())
            self.assertFalse(json.loads((root/'dispatch/cluster_first/test.json').read_text())['allow_vast'])
    def test_exact_resource_request_and_native_exclusions_are_read(self):
        with tempfile.TemporaryDirectory()as tmp:
            code=Path(tmp);submit=code/'scripts/trainfast/submit_fast.py';submit.parent.mkdir(parents=True)
            submit.write_text('EXCLUDE="private-node"\n')
            d=C.requirements({'identity':{'code':str(code)},'command':['submit','--kcollect','3','--mem','64G']})
            self.assertEqual(d['cpus'],12);self.assertEqual(d['memory_mb'],65536)
            self.assertIn('private-node',d['excluded_nodes']);self.assertIn('svl9',d['excluded_nodes'])
    def test_stale_shared_observation_is_refreshed_at_claim(self):
        with tempfile.TemporaryDirectory()as tmp, patch.object(C.time,'time',return_value=1100), patch.object(C,'probe',return_value={**self.observation,'observed_at':1100})as probe, patch.object(C,'requirements',return_value=self.demand):
            result=C.check(tmp,self.payload,self.observation)
        probe.assert_called_once();self.assertEqual(result['observation_age_seconds'],0)
    def test_unknown_exact_requirements_are_owned_and_not_dispatched(self):
        with tempfile.TemporaryDirectory()as tmp, patch.object(C.time,'time',return_value=1001), patch.object(C,'requirements',side_effect=ValueError('dynamic exclusion')):
            result=C.check(tmp,self.payload,self.observation)
        self.assertFalse(result['allow_vast']);self.assertEqual(result['reason'],'unverified_resource_requirements')
        self.assertTrue(result['next_action']);self.assertEqual(result['owner'],'/root/compute')


if __name__=='__main__':unittest.main()
