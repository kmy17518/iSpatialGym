"""Submit one prepared campaign validation with immutable provenance and private caches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root', 'group', 'snapshot', 'key', 'tag', 'v2-root'):
        p.add_argument('--'+name, required=True)
    p.add_argument('--index', required=True, type=int)
    p.add_argument('--ledger', default='canary_allocations.tsv')
    p.add_argument('--template-customs', action='store_true')
    p.add_argument('--bddl-root', type=Path)
    p.add_argument('--customs-file', type=Path)
    p.add_argument('--kcollect', type=int, default=2)
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--vast-gpu', choices=['RTX 4090','RTX 5090'])
    p.add_argument('--dispatch-spool', action='store_true')
    p.add_argument('--time-s', type=int)
    a = p.parse_args()
    root, group, snapshot = (Path(x).resolve() for x in (a.root, a.group, a.snapshot))
    assert group.is_relative_to(root) and snapshot.is_relative_to(root/'code')
    assert a.index >= 9000000 and Path(a.ledger).name == a.ledger
    config = json.loads((group/'campaign_config.json').read_text())
    source = json.loads((group/'source.json').read_text())
    assert config['purpose'] == 'validation' and not (group/'goal_plan.json').exists()
    assert hashlib.sha256((snapshot/'.snapshot_commit').read_bytes()).hexdigest() == config['validation_key']['code_sha256']
    assert not (group/'submit.txt').exists(), 'submission already attempted; reconcile ledger first'
    # Block design reserves 3600 s for a demo plus 420 s for shutdown. A 5400 s
    # allocation therefore stopped its collectors before slow native sampling
    # finished (fresh jobs 18423481 and 18423951). Preserve the sampler/oracle
    # budgets and provide a genuine collection window after sampling instead.
    block_design = config['template']['task'] == 'configuration_matching-block_design'
    time_s = a.time_s if a.time_s is not None else (9000 if block_design else 5400)
    assert time_s >= (2700 + 3600 + 420 if block_design else 2700), 'allocation cannot cover the collection window'
    env = {'TEMPLATE':source['template_dir'],'EVAL_INDEX':str(group/'empty_eval_index.json'),
           'ISPATIALGYM_CAMPAIGN_CONFIG':str(group/'campaign_config.json'),
           'ISGF_COLLECT_EXTRA':'--no-goal-override' + (' --max-steps 6000' if config['template']['task'] == 'configuration_matching-block_design' else ''),
           'ISGF_SAMPLING_CACHE':str(root/'cache'/a.tag/a.key/str(a.index)),
           'ISGF_ASSET_CACHE':str(root/'asset_cache'),'ISGF_JB_NFS':str(root/'jobs'),'ISGF_PREDEMO_ROOT':str(root),
           'KIT_TPL_NFS_ROOT':str(root/'kit_templates'),'KIT_TEX_STORE':str(root/'kit_textures'),
           'ISGF_EXP_PATH':str(root/'isaacsim_overlay/apps'),'ISPATIALGYM_FAST_BAND_SHARE':'0',
           'ISPATIALGYM_FAST_COLLECTOR_PREBUILD':'0'}
    if a.template_customs:
        env['START_PER_ATTEMPT'] = '0'
    if a.customs_file:
        customs = a.customs_file.resolve()
        assert a.template_customs and customs.is_file() and customs.is_relative_to(root), 'explicit private customs requires template mode and an owned file'
        metadata = json.loads((Path(source['template_dir'])/'template_meta.json').read_text())
        alias = metadata['variant']
        assert alias == config['template']['scene_setup']['customs_alias'], 'native customs alias differs from contract'
        assert Path(metadata['customs']).resolve() == customs, 'explicit customs differs from the native source'
        assert alias in json.loads(customs.read_text()), 'private customs lacks the native alias'
        env['ISG_CUSTOMS'] = str(customs)
        env['ISPATIALGYM_CAMPAIGN_CUSTOMS_SHA256'] = hashlib.sha256(customs.read_bytes()).hexdigest()
    if a.bddl_root:
        bddl = a.bddl_root.resolve()
        assert bddl.is_dir() and bddl.is_relative_to(root) and not any(c.isspace() for c in str(bddl)), 'private BDDL must be an owned whitespace-free directory'
        env['KEEP_SCENE'] = '--keep-template-scene --bddl-root ' + str(bddl)
    cmd = ['python3',str(snapshot/'scripts/trainfast/submit_fast.py'),'submit',a.key,'1','--flow','fast',
           '--code',str(snapshot),'--root',str(root),'--ledger',str(root/'manifests'/a.ledger),
           '--base-index',str(a.index),'--tag',a.tag,'--job-prefix','isg261110_','--grasps','v2only',
           '--v2-root',a.v2_root,'--alias-memory','off','--no-memory-aggregate','--kcollect',str(a.kcollect),
           '--sample-limit-s','2700','--time-s',str(time_s),'--customs','template' if a.template_customs else 'trunk']
    for name, value in env.items():
        cmd.extend(['--extra-env',name+'='+value])
    if a.dispatch_spool:
        assert not a.vast_gpu and not a.dry_run, 'spool owns backend routing and dry-run'
        from scripts.campaign_20261110.controller_spool import enqueue_record
        result = subprocess.run(cmd + ['--dry-run'], capture_output=True, text=True)
        (group/'dry.txt').write_text(result.stdout + result.stderr)
        assert result.returncode == 0, 'validation dry-run failed; inspect dry.txt'
        record = {'directory': str(group), 'command': cmd, 'dry_run_passed': True}
        enqueue_record(root, record, lambda: None, priority=25)
        (group/'dispatch_intent.json').write_text(json.dumps(record, indent=2) + '\n')
        print(json.dumps({'dispatch_intent': record['dispatch_intent'], 'dispatch_receipt': record['dispatch_receipt']}))
        return
    process_env = None
    if a.vast_gpu:
        submitter = root/'vast/tool/vsubmit'
        assert submitter.is_file(), 'campaign Vast submitter missing'
        cmd = [str(submitter)] + cmd
        process_env = dict(os.environ, VAST_GPU=a.vast_gpu, VAST_QUEUE_ROOT=str(root/'vast'))
    for dry in ([True] if a.dry_run else [True, False]):
        result = subprocess.run(cmd+(['--dry-run'] if dry else []),capture_output=True,text=True,env=process_env)
        (group/('dry.txt' if dry else 'submit.txt')).write_text(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError(result.stderr[-2000:])
        if not dry:
            print(result.stdout.strip())


if __name__ == '__main__':
    main()
