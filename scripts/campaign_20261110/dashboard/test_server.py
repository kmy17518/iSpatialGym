import hashlib
import http.client
import json
from pathlib import Path
import tempfile
import threading
import unittest

from scripts.campaign_20261110.dashboard.server import Handler, ThreadingHTTPServer, snapshot


class ReadOnlyDashboardTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        for name in ('infra', 'dashboard', 'inventory'):
            (self.root / name).mkdir()
        self.write('live.json', {'campaign_id': 'campaign_20261110', 'campaign_vast_metrics':
            {'running_gpus': 7, 'api_key': 'SHOULD_NEVER_APPEAR', 'instances': [{'_key_name': 'PRIVATE_LABEL'}]}})
        self.write('dashboard/dashboard_evidence.json', {'campaign_id': 'campaign_20261110',
            'counts': {'eval_standard': 11, 'eval_model': 2, 'eval_category': 1, 'train': 31}})
        self.write('infra/cluster_snapshot.json', {'jobs': [], 'validation': {'gates': {}}})
        (self.root / 'LIVE.md').write_text('Campaign report\n')
        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.server.data_root, self.server.inventory = self.root, self.root / 'inventory'
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def write(self, name, data):
        (self.root / name).write_text(json.dumps(data))

    def request(self, method, path, host=None):
        conn = http.client.HTTPConnection(*self.server.server_address)
        conn.request(method, path, headers={'Host': host or '127.0.0.1'})
        response = conn.getresponse()
        result = response.status, response.read()
        conn.close()
        return result

    def hashes(self):
        return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in self.root.rglob('*') if p.is_file()}

    def test_deadline_checkpoint_is_read_only_scoped_and_sanitized(self):
        value={'campaign_id':'campaign_20261110','observed_at':'2026-11-12T01:02:41Z',
               'complete_by_deadline_supported':False,'forecasts':{'full_selected_preparation':
               {'status':'incomplete_denominator_and_runtime_model'}},'api_key':'HIDDEN'}
        self.write('dashboard/streaming_deadline_checkpoint.json',value)
        before=self.hashes();data=json.loads(self.request('GET','/api/status')[1])
        self.assertFalse(data['deadline_checkpoint']['complete_by_deadline_supported'])
        code,raw=self.request('GET','/reports/deadline-checkpoint')
        self.assertEqual(code,200);self.assertNotIn(b'HIDDEN',raw);self.assertEqual(before,self.hashes())
        self.write('dashboard/streaming_deadline_checkpoint.json',{'campaign_id':'old'})
        self.assertFalse(snapshot(self.root,self.root/'inventory')['deadline_checkpoint_source']['available'])
        self.assertEqual(self.request('GET','/reports/deadline-checkpoint')[0],404)

    def test_dependency_readiness_requires_current_exact_preparation_join(self):
        source={'path':'/campaign/training_preparation/frozen/dependency_kind_readiness.json','sha256':'abc'}
        self.write('dashboard/preparation_status.json',{'campaign_id':'campaign_20261110',
            'execution':{'dependency_kind_readiness':source}})
        self.write('dashboard/dependency_readiness.json',{'campaign_id':'campaign_20261110','source':source,
            'data':{'campaign_id':'campaign_20261110','required_logical':100,'kinds':[
                {'kind':'reach','verified_logical':4,'required_logical':20,'exact_unresolved_request_ids':['large-list']}]}})
        before=self.hashes();data=snapshot(self.root,self.root/'inventory')
        self.assertEqual(data['dependency_readiness']['required_logical'],100)
        self.assertNotIn('exact_unresolved_request_ids',data['dependency_readiness']['kinds'][0])
        self.assertEqual(self.request('GET','/reports/dependency-readiness')[0],200)
        self.assertEqual(before,self.hashes())
        self.write('dashboard/preparation_status.json',{'campaign_id':'campaign_20261110',
            'execution':{'dependency_kind_readiness':{**source,'sha256':'new-proof'}}})
        data=snapshot(self.root,self.root/'inventory')
        self.assertEqual(data['dependency_readiness'],{})
        self.assertFalse(data['dependency_readiness_source']['available'])

    def test_runtime_suite_keeps_historical_and_fresh_results_separate(self):
        value={'campaign_id':'campaign_20261110','historical_accepted_classes':100,
               'historical_runtime_identities':40,'validation_passed':0,'required_classes':100,
               'rows':[{'class_id':'one','state':'failed_dispatch'}],'api_key':'HIDDEN'}
        self.write('dashboard/runtime_validation.json',value)
        before=self.hashes();data=json.loads(self.request('GET','/api/status')[1])
        self.assertEqual(data['runtime_validation']['validation_passed'],0)
        self.assertEqual(data['runtime_validation']['historical_accepted_classes'],100)
        code,raw=self.request('GET','/reports/runtime-validation')
        self.assertEqual(code,200);self.assertNotIn(b'HIDDEN',raw);self.assertEqual(before,self.hashes())
        self.write('dashboard/runtime_validation.json',{'campaign_id':'old','validation_passed':100})
        self.assertFalse(snapshot(self.root,self.root/'inventory')['runtime_validation_source']['available'])
        self.assertEqual(self.request('GET','/reports/runtime-validation')[0],404)

    def test_capacity_ledger_is_read_only_and_campaign_scoped(self):
        value={'campaign_id':'campaign_20261110','observed_at':123,
               'unused_capacity_blockers':[{'owner':'root','missing_input':'exact native map','next_action':'audit'}]}
        self.write('dashboard/capacity_recovery.json',value)
        before=self.hashes()
        data=snapshot(self.root,self.root/'inventory')
        self.assertEqual(data['capacity_recovery'],value)
        self.assertEqual(self.request('GET','/reports/capacity-recovery')[0],200)
        self.assertEqual(self.hashes(),before)
        value['campaign_id']='previous_campaign'
        self.write('dashboard/capacity_recovery.json',value)
        self.assertEqual(snapshot(self.root,self.root/'inventory')['capacity_recovery'],{})
        self.assertNotEqual(self.request('GET','/reports/capacity-recovery')[0],200)

    def test_pipeline_attempts_are_scoped_read_only_and_zero_credit(self):
        value={'campaign_id':'campaign_20261110','observed_at':123,
               'benchmark_credit':0,'quota_consumed':0,
               'records':[{'candidate_id':'augmentation_a','status':'validated_canary_exported'}],
               'batch_stages':[{'status':'prepared_unvalidated'}],'api_key':'HIDDEN'}
        self.write('dashboard/pipeline_status.json',value)
        before=self.hashes();data=json.loads(self.request('GET','/api/status')[1])
        self.assertEqual(data['pipeline_status']['benchmark_credit'],0)
        self.assertEqual(data['pipeline_status']['batch_stages'][0]['status'],'prepared_unvalidated')
        code,raw=self.request('GET','/reports/pipeline-status')
        self.assertEqual(code,200);self.assertNotIn(b'HIDDEN',raw);self.assertEqual(before,self.hashes())
        self.write('dashboard/pipeline_status.json',{'campaign_id':'previous_campaign'})
        self.assertEqual(snapshot(self.root,self.root/'inventory')['pipeline_status'],{})
        self.assertEqual(self.request('GET','/reports/pipeline-status')[0],404)

    def test_reads_preserve_all_monitoring_files_and_counts(self):
        before = self.hashes()
        for _ in range(3):
            code, raw = self.request('GET', '/api/status')
            self.assertEqual(code, 200)
            data = json.loads(raw)
            self.assertEqual(data['evidence']['counts'], {'eval_standard': 11, 'eval_model': 2, 'eval_category': 1, 'train': 31})
            self.assertEqual(sum(data['targets'].values()), 102000)
        self.request('GET', '/reports/live')
        self.assertEqual(before, self.hashes())

    def test_execution_and_configuration_pages_preserve_denominator_and_isolation(self):
        self.write('dashboard/execution_ledger.json',{'campaign_id':'campaign_20261110','rows':[],
                   'preparation':{'required_configurations':101,'verified_configurations':0}})
        self.write('dashboard/configuration_index.json',{'campaign_id':'campaign_20261110','rows':[
            {'id':str(i),'class_id':'C','model_keys':['bottle/A'],'state':'blocked','api_key':'HIDDEN'} for i in range(101)]})
        before=self.hashes();code,raw=self.request('GET','/api/configurations?q=bottle&offset=100')
        self.assertEqual(code,200);value=json.loads(raw)
        self.assertEqual(value['required_configurations_in_index'],101);self.assertEqual(len(value['rows']),1)
        self.assertNotIn(b'HIDDEN',raw)
        self.assertEqual(self.request('GET','/reports/execution-ledger')[0],200)
        self.assertEqual(before,self.hashes())
        self.write('dashboard/configuration_index.json',{'campaign_id':'old','rows':[]})
        self.assertEqual(self.request('GET','/api/configurations')[0],404)

    def test_mutating_http_methods_are_rejected_without_writes(self):
        before = self.hashes()
        for method in ('POST', 'PUT', 'PATCH', 'DELETE'):
            self.assertEqual(self.request(method, '/api/status')[0], 405)
        self.assertEqual(before, self.hashes())

    def test_video_ranges_and_allowlist_are_read_only(self):
        directory=self.root/'dashboard/batch_restoration_videos';directory.mkdir()
        video=directory/'same_augmentation_episode_2_h264.mp4';video.write_bytes(b'0123456789')
        before=self.hashes();conn=http.client.HTTPConnection(*self.server.server_address)
        conn.request('GET','/videos/'+video.name,headers={'Host':'127.0.0.1','Range':'bytes=2-5'})
        response=conn.getresponse()
        self.assertEqual(response.status,206);self.assertEqual(response.getheader('Content-Range'),'bytes 2-5/10')
        self.assertEqual(response.read(),b'2345');conn.close()
        self.assertEqual(self.request('GET','/videos/../../live.json')[0],404)
        self.assertEqual(self.request('GET','/videos/local_videos.json')[0],404)
        self.assertEqual(self.request('POST','/videos/'+video.name)[0],405)
        self.assertEqual(self.request('GET','/videos/'+video.name,host='external.example')[0],403)
        self.assertEqual(before,self.hashes())
        video.unlink();video.symlink_to(self.root/'LIVE.md')
        self.assertEqual(self.request('GET','/videos/'+video.name)[0],404)

    def test_restoration_report_requires_campaign_identity(self):
        self.write('dashboard/batch_restoration_status.json',{'campaign_id':'previous_campaign'})
        self.assertEqual(self.request('GET','/reports/batch-restoration')[0],404)
        self.write('dashboard/batch_restoration_status.json',{'campaign_id':'campaign_20261110','api_key':'HIDDEN','scope':'same augmentation'})
        before=self.hashes();code,raw=self.request('GET','/reports/batch-restoration')
        self.assertEqual(code,200);self.assertNotIn(b'HIDDEN',raw);self.assertEqual(before,self.hashes())

    def test_credentials_and_unapproved_files_are_not_served(self):
        raw = self.request('GET', '/api/status')[1]
        self.assertNotIn(b'SHOULD_NEVER_APPEAR', raw)
        self.assertNotIn(b'PRIVATE_LABEL', raw)
        for path in ('/etc/passwd', '/../../live.json', '/logs', '/live.json'):
            self.assertEqual(self.request('GET', path)[0], 404)
        self.assertEqual(self.request('GET', '/api/status', 'public.example')[0], 403)

    def test_other_campaign_identity_is_rejected(self):
        self.write('live.json', {'campaign_id': 'old_campaign'})
        self.write('dashboard/dashboard_evidence.json', {'campaign_id': 'old_campaign', 'counts': {'train': 99999}})
        data = snapshot(self.root, self.root / 'inventory')
        self.assertEqual(data['live'], {})
        self.assertEqual(data['evidence'], {})
        self.assertFalse(data['sources']['evidence']['available'])

    def test_containment_receipt_is_scoped_read_only(self):
        value={'campaign_id':'campaign_20261110','status':'verified_blocked','api_key':'HIDDEN'}
        self.write('dashboard/training_containment.json',value)
        before=self.hashes()
        code,raw=self.request('GET','/reports/training-containment')
        self.assertEqual(code,200);self.assertNotIn(b'HIDDEN',raw)
        self.assertEqual(snapshot(self.root,self.root/'inventory')['training_containment']['status'],'verified_blocked')
        self.assertEqual(before,self.hashes())
        self.write('dashboard/training_containment.json',{'campaign_id':'older'})
        self.assertEqual(self.request('GET','/reports/training-containment')[0],404)
        self.assertEqual(snapshot(self.root,self.root/'inventory')['training_containment'],{})

    def test_group_yield_isolated_read_only_and_full_evidence_route(self):
        value={'campaign_id':'campaign_20261110','summary':{'standard_eval':{'completed_groups':4}},
               'sources':[{'path':'proof','sha256':'abc'}],'groups':[{'group_id':'one'}],'api_key':'HIDDEN'}
        self.write('dashboard/goal_group_yield.json',value)
        before=self.hashes();data=json.loads(self.request('GET','/api/status')[1])
        self.assertEqual(data['group_yield']['summary'],value['summary'])
        self.assertNotIn('sources',data['group_yield'])
        status,raw=self.request('GET','/reports/goal-yield')
        self.assertEqual(status,200);self.assertEqual(json.loads(raw)['groups'],value['groups'])
        self.assertNotIn(b'HIDDEN',raw);self.assertEqual(before,self.hashes())
        self.write('dashboard/goal_group_yield.json',{'campaign_id':'older','summary':{}})
        data=snapshot(self.root,self.root/'inventory')
        self.assertEqual(data['group_yield'],{});self.assertFalse(data['sources']['group_yield']['available'])

    def test_resource_jobs_are_scoped_to_this_campaign(self):
        self.write('infra/cluster_snapshot.json', {'jobs': [
            {'id': '101', 'name': 'isg261110_production'},
            {'id': '102', 'name': 'isgof_previous_campaign'},
            {'id': '103', 'name': 'isg_category_unrelated'},
            {'id': '18423963', 'name': 'catwardrobe'}]})
        data = snapshot(self.root, self.root / 'inventory')
        self.assertEqual([j['id'] for j in data['cluster']['jobs']], ['101', '18423963'])

    def test_heldout_ledger_is_campaign_scoped_read_only_and_sanitized(self):
        value={'campaign_id':'campaign_20261110','owner':'/root/templates_assets',
            'rows':[{'split':'eval_model','remaining':3,'api_key':'HIDDEN','active_jobs':[{'job_id':'123','state':'PENDING'}]}]}
        self.write('dashboard/heldout_recovery.json',value)
        before=self.hashes();code,raw=self.request('GET','/reports/heldout-recovery')
        self.assertEqual(code,200);self.assertNotIn(b'HIDDEN',raw)
        self.assertEqual(snapshot(self.root,self.root/'inventory')['heldout_recovery']['rows'][0]['remaining'],3)
        self.assertEqual(self.hashes(),before)
        self.write('dashboard/heldout_recovery.json',{'campaign_id':'older'})
        self.assertEqual(self.request('GET','/reports/heldout-recovery')[0],404)
        self.assertFalse(snapshot(self.root,self.root/'inventory')['sources']['heldout_recovery']['available'])

    def test_standard_recovery_is_read_only_sanitized_and_campaign_scoped(self):
        value={'campaign_id':'campaign_20261110','zero_classes':59,
               'classes':[{'class_id':'inside_01','accepted_standard':0,'api_key':'HIDDEN'}]}
        self.write('dashboard/standard_recovery.json',value)
        before=self.hashes()
        code,raw=self.request('GET','/reports/standard-recovery')
        self.assertEqual(code,200);self.assertNotIn(b'HIDDEN',raw)
        data=json.loads(self.request('GET','/api/status')[1])
        self.assertEqual(data['standard_recovery']['zero_classes'],59)
        self.assertEqual(before,self.hashes())
        self.write('dashboard/standard_recovery.json',{'campaign_id':'older'})
        self.assertEqual(self.request('GET','/reports/standard-recovery')[0],404)
        self.assertEqual(snapshot(self.root,self.root/'inventory')['standard_recovery'],{})

    def test_audit_report_arrays_are_readable_and_sanitized(self):
        self.write('dashboard/vast_runtime_resources.json', [{'cpu_affinity': '1-9', 'api_key': 'HIDDEN'}])
        status, raw = self.request('GET', '/reports/vast-runtime')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw), [{'cpu_affinity': '1-9'}])

    def test_training_eligibility_preserves_history_without_fallback_and_is_read_only(self):
        data=snapshot(self.root,self.root/'inventory')
        self.assertEqual(data['training_eligibility'],{})
        self.assertEqual(data['evidence']['counts']['train'],31)
        self.assertFalse(data['sources']['training_eligibility']['available'])
        value={'campaign_id':'campaign_20261110','historically_accepted':31,
               'eligible_accepted':7,'excluded':24,'unverified':0,'api_key':'HIDDEN'}
        self.write('dashboard/training_eligibility.json',value)
        before=self.hashes();code,raw=self.request('GET','/reports/training-eligibility')
        self.assertEqual(code,200);self.assertNotIn(b'HIDDEN',raw)
        data=json.loads(self.request('GET','/api/status')[1])
        self.assertEqual(data['training_eligibility']['eligible_accepted'],7)
        self.assertEqual(data['evidence']['counts']['train'],31)
        self.assertEqual(before,self.hashes())
        self.write('dashboard/training_eligibility.json',{'campaign_id':'older','eligible_accepted':99999})
        self.assertEqual(self.request('GET','/reports/training-eligibility')[0],404)
        self.assertEqual(snapshot(self.root,self.root/'inventory')['training_eligibility'],{})

    def test_training_gate_and_requirements_routes_are_scoped_and_never_write(self):
        for name,route in [('training_production_gate','training-production-gate'),('training_requirements','training-requirements'),('training_stream','training-stream'),('existing_native_reviews','existing-native-reviews'),('ik_observation','ik-cache'),('change_impact','change-impact'),('current_eligibility','current-eligibility')]:
            self.write('dashboard/'+name+'.json',{'campaign_id':'campaign_20261110','status':'closed','command':'DO NOT EXECUTE'})
            before=self.hashes();code,raw=self.request('GET','/reports/'+route)
            self.assertEqual(code,200);self.assertNotIn(b'DO NOT EXECUTE',raw)
            self.assertEqual(before,self.hashes())
            self.write('dashboard/'+name+'.json',{'campaign_id':'older'})
            self.assertEqual(self.request('GET','/reports/'+route)[0],404)

    def test_new_quality_and_denominator_reports_are_scoped_sanitized_and_read_only(self):
        for route,name in [('/reports/replay-quality','gate_quality_audit.json'),
                           ('/reports/gpu-denominator-coverage','cohort_rental_denominator_coverage.json')]:
            self.write('dashboard/'+name,{'campaign_id':'campaign_20261110','observed_at':'dated','api_key':'HIDDEN'})
            before=self.hashes();status,raw=self.request('GET',route)
            self.assertEqual(status,200);self.assertNotIn('HIDDEN',raw.decode());self.assertEqual(before,self.hashes())
            self.write('dashboard/'+name,{'campaign_id':'older'})
            self.assertEqual(self.request('GET',route)[0],404)

    def test_accepted_release_is_read_only_and_rejects_wrong_campaign(self):
        self.write('dashboard/accepted_release.json',{'campaign_id':'campaign_20261110','release':{'instances':3704}})
        before=self.hashes();status,raw=self.request('GET','/reports/accepted-release')
        self.assertEqual(status,200);self.assertEqual(json.loads(raw)['release']['instances'],3704)
        self.assertEqual(before,self.hashes())
        for invalid in ({'campaign_id':'old','release':{}},[]):
            self.write('dashboard/accepted_release.json',invalid)
            self.assertEqual(self.request('GET','/reports/accepted-release')[0],404)

    def test_fleet_gpu_time_preserves_scope_and_rejects_other_campaigns(self):
        value={'campaign_id':'campaign_20261110','fleet_window_estimated_gpu_seconds':7200,
               'fleet_window_measured_attributed_worker_gpu_seconds':3600,
               'rentals':[{'instance':'owned'}], 'sources':[{'path':'private proof'}]}
        self.write('dashboard/vast_fleet_gpu_time.json',value)
        before=self.hashes();data=json.loads(self.request('GET','/api/status')[1])
        self.assertEqual(data['fleet_gpu_time']['fleet_window_estimated_gpu_seconds'],7200)
        self.assertNotIn('rentals',data['fleet_gpu_time'])
        self.assertNotIn('accepted_per_gpu_hour',data['fleet_gpu_time'])
        code,raw=self.request('GET','/reports/fleet-gpu-time')
        self.assertEqual(code,200);self.assertEqual(json.loads(raw)['rentals'],value['rentals'])
        self.assertEqual(before,self.hashes())
        self.write('dashboard/vast_fleet_gpu_time.json',{'campaign_id':'old'})
        self.assertEqual(self.request('GET','/reports/fleet-gpu-time')[0],404)
        self.assertFalse(snapshot(self.root,self.root/'inventory')['sources']['fleet_gpu_time']['available'])

    def test_missing_or_invalid_sources_are_explicit(self):
        (self.root / 'dashboard/dashboard_evidence.json').write_text('{broken')
        (self.root / 'infra/cluster_snapshot.json').unlink()
        data = snapshot(self.root, self.root / 'inventory')
        self.assertFalse(data['sources']['evidence']['available'])
        self.assertFalse(data['sources']['cluster']['available'])
        self.assertIn('error', data['sources']['evidence'])

    def test_broader_cluster_time_is_separate_and_campaign_scoped(self):
        value={'campaign_id':'campaign_20261110','verified_allocated_gpu_seconds':10800,
               'records':[{'job_id':'123','gpu_count':1}], 'accepted_instance_rate':None}
        self.write('dashboard/cluster_fleet_gpu_time.json',value)
        before=self.hashes();data=json.loads(self.request('GET','/api/status')[1])
        self.assertEqual(data['cluster_fleet_gpu_time']['allocation_count'],1)
        self.assertEqual(data['cluster_fleet_gpu_time']['verified_allocated_gpu_seconds'],10800)
        self.assertIsNone(data['cluster_fleet_gpu_time']['accepted_instance_rate'])
        self.assertNotIn('records',data['cluster_fleet_gpu_time'])
        code,raw=self.request('GET','/reports/cluster-fleet-gpu-time')
        self.assertEqual(code,200);self.assertEqual(json.loads(raw)['records'],value['records'])
        self.assertEqual(before,self.hashes())
        self.write('dashboard/cluster_fleet_gpu_time.json',{'campaign_id':'older'})
        self.assertEqual(self.request('GET','/reports/cluster-fleet-gpu-time')[0],404)
        self.assertFalse(snapshot(self.root,self.root/'inventory')['sources']['cluster_fleet_gpu_time']['available'])


if __name__ == '__main__':
    unittest.main()
