import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

from scripts.campaign_20261110.ik_native_probe import (CAMPAIGN, SCHEMA, sha, validate_spec,
    compatibility_difference, write_observation_environment, observe_native_shutdown)


class ProbeBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        root = self.root = Path(self.tmp.name) / CAMPAIGN
        code = root / 'code/pinned'; code.mkdir(parents=True)
        source = root / 'groups/source'; source.mkdir(parents=True)
        for path in [code / '.snapshot_commit', source / 'template.json', source / 'template_meta.json', root / 'customs.json', root / 'manifest.json']:
            path.write_text('{}')
        config = {'split': 'eval_standard', 'validation_key': {'code_sha256': sha(code / '.snapshot_commit')},
                  'template': {'objects': [{'ready': True}], 'scene_setup': {'source_template': str(source),
                  'source_template_sha256': sha(source / 'template.json')}}}
        cp = root / 'config.json'; cp.write_text(json.dumps(config))
        self.spec = {'schema': SCHEMA, 'campaign_id': CAMPAIGN, 'diagnostic_only': True, 'benchmark_credit': 0,
                     'source_root': str(root), 'snapshot': str(code), 'config': str(cp), 'manifest': str(root / 'manifest.json'),
                     'customs': str(root / 'customs.json'), 'template': str(source), 'reference_index': 50265300,
                     'inputs': {str(p): sha(p) for p in [code / '.snapshot_commit', cp, root / 'manifest.json',
                         root / 'customs.json', source / 'template.json', source / 'template_meta.json']}}
        self.output = root / 'ik_exchange/probes/diagnostic1'

    def tearDown(self):
        self.tmp.cleanup()

    def test_valid_probe_retains_source_and_only_allows_diagnostic_output(self):
        before = {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        _, output = validate_spec(self.spec, self.output)
        self.assertEqual(output, self.output)
        self.assertEqual(before, {str(p): p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        with self.assertRaises(ValueError):
            validate_spec(self.spec, self.root / 'out/collection')

    def test_changed_runtime_or_scene_cannot_be_claimed_identical(self):
        Path(self.spec['snapshot'], '.snapshot_commit').write_text('changed')
        with self.assertRaisesRegex(ValueError, 'changed pinned input'):
            validate_spec(self.spec, self.output)

    def test_existing_evidence_is_not_overwritten(self):
        self.output.mkdir(parents=True); (self.output / 'probe_report.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'already has evidence'):
            validate_spec(self.spec, self.output)

    def test_production_credit_and_unbounded_queries_rejected(self):
        self.spec['benchmark_credit'] = 1
        with self.assertRaisesRegex(ValueError, 'nonaccepting'):
            validate_spec(self.spec, self.output)
        self.spec['benchmark_credit'] = 0; self.spec['total_query_limit'] = 501
        with self.assertRaisesRegex(ValueError, 'query bounds'):
            validate_spec(self.spec, self.output)

    def test_compatibility_report_distinguishes_interpreter_hashes_from_settings(self):
        expected = {'modules': {'ik.py': 'old'}, 'settings': {'ISPATIALGYM_FAST': '1'},
                    'objects': [{'applied_scale': [1., 1., 1.]}]}
        actual = {'modules': {'ik.py': 'new'}, 'settings': {'ISPATIALGYM_FAST': '1'},
                  'objects': [{'applied_scale': (1., 1., 1.)}]}
        self.assertEqual(compatibility_difference(expected, actual), {'modules': ['ik.py']})

    def test_environment_evidence_excludes_credentials_and_never_overwrites(self):
        self.output.mkdir(parents=True)
        result = write_observation_environment(self.output, self.spec['snapshot'],
            {'ISPATIALGYM_EXACT_IK_PRODUCER': 'primary:diagnostic', 'API_KEY': 'do-not-record'})
        text = Path(result['path']).read_text()
        self.assertIn('ISPATIALGYM_EXACT_IK_PRODUCER=primary:diagnostic', text)
        self.assertNotIn('API_KEY', text)
        self.assertNotIn('do-not-record', text)
        with self.assertRaises(FileExistsError):
            write_observation_environment(self.output, self.spec['snapshot'], {})

    def test_local_native_import_flushes_before_process_exiting_shutdown(self):
        events = []
        def original(*args, **kwargs):
            events.append(('native_shutdown', args, kwargs))
            raise SystemExit(0)
        module = SimpleNamespace(shutdown=original)
        with patch.dict('sys.modules', {'omnigibson': module}):
            restore = observe_native_shutdown(lambda: events.append('flushed'))
            # Same import scope as multiply_isg.main: no caller.og attribute.
            def native_main():
                import omnigibson as og
                og.shutdown('argument', flag=True)
            with self.assertRaises(SystemExit):
                native_main()
            self.assertEqual(events, ['flushed', ('native_shutdown', ('argument',), {'flag': True})])
            restore()
            self.assertIs(module.shutdown, original)


if __name__ == '__main__':
    unittest.main()
