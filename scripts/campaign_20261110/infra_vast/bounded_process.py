"""Bound an owned staging process and all of its upload subprocesses."""
import os
from pathlib import Path
import signal
import subprocess
import time


def run_bounded(command, log_path, *, timeout, progress=None, interval=30):
    if timeout <= 0:
        raise ValueError('positive staging timeout required')
    started = time.monotonic()
    with Path(log_path).open('a') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            while True:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    code = process.wait(timeout=min(interval, remaining))
                except subprocess.TimeoutExpired:
                    if progress:
                        progress(time.monotonic() - started)
                    continue
                if code:
                    raise subprocess.CalledProcessError(code, command)
                return
        except BaseException:
            # This session was created solely for this invocation. Uploads must
            # not continue modifying a quarantined job after its timeout.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
            finally:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
            raise
