"""Retry transient observation I/O while retaining a real module identity."""
import argparse
import json
import runpy
import subprocess
import sys
import time


def execute(*, module=None, script=None, argv=()):
    if bool(module)==bool(script):
        raise ValueError('exactly one observer entry point required')
    sys.argv=[script or module,*argv]
    if script:
        return runpy.run_path(script,run_name='__main__')
    # Some adapters intentionally pass sys.modules[__name__] to the common
    # controller. It must refer to the executing adapter, not this wrapper.
    return runpy.run_module(module,run_name='__main__',alter_sys=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--script');p.add_argument('--module')
    p.add_argument('args',nargs=argparse.REMAINDER);a=p.parse_args()
    argv=a.args[1:] if a.args[:1]==['--'] else a.args
    while True:
        try:
            execute(module=a.module,script=a.script,argv=argv)
            break
        except (OSError,subprocess.TimeoutExpired) as error:
            print(json.dumps({'observation_failed':type(error).__name__+': '+str(error),
                'observed_at':time.time(),'retry_seconds':30,'acceptance_on_missing_evidence':False}),flush=True)
            time.sleep(30)


if __name__=='__main__':main()
