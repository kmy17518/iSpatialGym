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

    def test_mutating_http_methods_are_rejected_without_writes(self):
        before = self.hashes()
        for method in ('POST', 'PUT', 'PATCH', 'DELETE'):
            self.assertEqual(self.request(method, '/api/status')[0], 405)
        self.assertEqual(before, self.hashes())

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

    def test_audit_report_arrays_are_readable_and_sanitized(self):
        self.write('dashboard/vast_runtime_resources.json', [{'cpu_affinity': '1-9', 'api_key': 'HIDDEN'}])
        status, raw = self.request('GET', '/reports/vast-runtime')
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(raw), [{'cpu_affinity': '1-9'}])

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


if __name__ == '__main__':
    unittest.main()
