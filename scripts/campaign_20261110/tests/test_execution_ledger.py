import json
from pathlib import Path
import tempfile
import unittest
from scripts.campaign_20261110.execution_ledger import build, SOURCES


class ExecutionLedgerTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup);self.root=Path(self.temp.name)
    def put(self,key,value):
        p=self.root/SOURCES[key];p.parent.mkdir(parents=True,exist_ok=True)
        p.write_text(json.dumps({'campaign_id':'campaign_20261110','observed_at':1000,**value}))
    def test_missing_and_stale_sources_do_not_become_zero_or_pass(self):
        self.put('canonical',{'validated':97,'total':100,'classes':[]})
        x=build(self.root,1700)
        self.assertEqual(x['sources']['canonical']['state'],'stale')
        self.assertEqual(x['sources']['configurations']['state'],'unavailable')
        self.assertIsNone(x['preparation']['required_configurations'])
        self.assertIsNone(x['accepted']['counts']);self.assertFalse(x['release_decisions']['ledger_alone_authorizes_release'])
    def test_configuration_coverage_requires_exact_artifact_evidence(self):
        self.put('configurations',{'configurations':[{'id':'C','dependencies':['D']}],
            'artifact_requests':[{'id':'D','status':'verified'}], 'scope_gaps':['unresolved joint expansion']})
        x=build(self.root,1001)
        self.assertEqual(x['preparation']['required_configurations'],1)
        self.assertEqual(x['preparation']['verified_configurations'],0)
        self.assertFalse(x['preparation']['denominator_complete'])
        self.assertTrue(x['indexed_sources']['configurations']['sha256'])
    def test_completed_native_never_implies_demo_replay_or_acceptance(self):
        self.put('pipeline',{'records':[{'candidate_id':'A','native':{'written':1,'sha256':'native'},'status':'native_passed'}]})
        x=build(self.root,1001);r=x['rows'][0]
        self.assertEqual(r['stages']['native']['status'],'source_evidence')
        for s in ('demo','physics_replay','rendering'):self.assertEqual(r['stages'][s]['status'],'unobserved')
        self.assertEqual(r['stages']['accepted']['status'],'excluded_zero_credit')
        self.assertTrue(r['sources'][0]['sha256'])

    def test_native_submission_is_visible_without_inventing_execution(self):
        receipt={'job_id':'18444227','request_sha256':'exact-request',
                 'scheduler_state':'unavailable','receipt':{'path':'submission.json','sha256':'exact-receipt'}}
        self.put('pipeline',{'records':[{'candidate_id':'pool','status':'native_submitted',
                                       'native_submission':receipt}]})
        r=build(self.root,1001)['rows'][0]
        self.assertEqual(r['active_work'],[receipt])
        self.assertEqual(r['stages']['submitted']['evidence'],receipt)
        for name in ('native','demo','physics_replay','rendering'):
            self.assertEqual(r['stages'][name]['status'],'unobserved')
        self.assertEqual(r['stages']['accepted']['status'],'excluded_zero_credit')
        self.put('pipeline',{'records':[{'candidate_id':'missing'}]})
        self.assertEqual(build(self.root,1001)['rows'][0]['active_work'],[])


if __name__=='__main__':unittest.main()
