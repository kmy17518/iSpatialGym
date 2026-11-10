from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest

from scripts.campaign_20261110.infra_vast.bounded_process import run_bounded


class BoundedStagingTests(unittest.TestCase):
    def test_success_and_failure_keep_logs(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'stage.log'
            run_bounded([sys.executable, '-c', "print('uploaded')"], log, timeout=2)
            with self.assertRaises(subprocess.CalledProcessError):
                run_bounded([sys.executable, '-c', "print('bad input');raise SystemExit(4)"], log, timeout=2)
            self.assertEqual(log.read_text(), 'uploaded\nbad input\n')

    def test_timeout_stops_upload_children_and_keeps_progress(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / 'unexpected_upload'
            child = "import time,pathlib;time.sleep(1);pathlib.Path(" + repr(str(marker)) + ").write_text('bad')"
            parent = "import subprocess,sys,time;subprocess.Popen([sys.executable,'-c'," + repr(child) + "]);time.sleep(10)"
            progress = []
            with self.assertRaises(subprocess.TimeoutExpired):
                run_bounded([sys.executable, '-c', parent], Path(directory) / 'stage.log',
                            timeout=.2, interval=.05, progress=progress.append)
            time.sleep(1)
            self.assertTrue(progress)
            self.assertFalse(marker.exists())

    def test_positive_timeout_required(self):
        with self.assertRaises(ValueError):
            run_bounded([], '/not-created', timeout=0)
