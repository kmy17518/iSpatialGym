"""Migrate manual/family GPU intents through a proved never-started boundary.

The primary account prepares fresh immutable inputs and publishes replacements.
Each scheduler owner alone holds/cancels its jobs. Held/camera queue controllers
retain their own migration ownership. All prior inputs and allocations survive.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import subprocess
import time

from scripts.campaign_20261110.contracts import file_digest, require
from scripts.campaign_20261110.controller_families import reserve_indices
from scripts.campaign_20261110.controller_migration import eligible, migrate_record, no_execution_artifacts, scheduler_proof
from scripts.campaign_20261110.infra_dispatch import enqueue, input_identity, set_option, shared_json
from scripts.campaign_20261110.storage import atomic_directory_lock


def read(path):
    return json.loads(Path(path).read_text())


def environment(command):
    pairs = [command[i + 1].split('=', 1) for i, word in enumerate(command) if word == '--extra-env']
    require(len({pair[0] for pair in pairs}) == len(pairs), 'migration requires unique environment keys')
    return dict(pairs)


def set_environment(command, name, value):
    matches = [i + 1 for i, word in enumerate(command) if word == '--extra-env' and command[i + 1].startswith(name + '=')]
    require(len(matches) <= 1, 'duplicate migration environment key')
    if matches: command[matches[0]] = name + '=' + str(value)
    else: command.extend(['--extra-env', name + '=' + str(value)])


def excluded_controller_work(root, payload):
    if environment(payload['command']).get('ISPATIALGYM_CAMPAIGN_GOAL_QUEUE'):
        return True
    token = payload['token']
    for path in (root / 'controllers').glob('*/state.json'):
        if not path.parent.name.startswith(('held', 'camera', 'generic')): continue
        # Exact dispatch paths or directory names are used, never substring
        # matching against an unrelated model/class name.
        def owns(value):
            if isinstance(value, dict):
                for key in ('dispatch_intent', 'dispatch_receipt', 'directory'):
                    if key in value and Path(str(value[key])).stem == token: return True
                return any(owns(child) for child in value.values())
            if isinstance(value, list): return any(owns(child) for child in value)
            return False
        if owns(read(path)): return True
    return False


def immutable_bytes(source, destination):
    data = Path(source).read_bytes()
    if destination.exists():
        require(destination.read_bytes() == data, 'prepared migration input differs')
    else:
        with destination.open('xb') as stream:
            stream.write(data); stream.flush(); os.fsync(stream.fileno())
        destination.chmod(0o640)


def fresh_replacement(root, payload, token, index):
    """Copy the contract exactly; change only nonphysical execution identities."""
    command = list(payload['command']); env = environment(command)
    require(not env.get('ISPATIALGYM_CAMPAIGN_GOAL_QUEUE'), 'queue migration belongs to its controller')
    require(input_identity(root, command) == payload['identity'], 'source intent changed')
    directory = root / 'groups/dispatch_migrations' / token
    directory.mkdir(parents=True, exist_ok=True)
    config = directory / 'campaign_config.json'
    immutable_bytes(env['ISPATIALGYM_CAMPAIGN_CONFIG'], config)
    set_environment(command, 'ISPATIALGYM_CAMPAIGN_CONFIG', config)
    old_index = Path(env['EVAL_INDEX'])
    index_data = read(old_index)
    require(index_data == {'schema':'isg-instance-index-v1', 'n':0, 'instances':[]}, 'source index is not empty')
    immutable_bytes(old_index, directory / 'empty_eval_index.json')
    set_environment(command, 'EVAL_INDEX', directory / 'empty_eval_index.json')
    set_environment(command, 'ISGF_SAMPLING_CACHE', root / 'cache/dispatch_migrations' / token)
    old_directory = Path(env.get('ISPATIALGYM_CAMPAIGN_GOAL_PLAN', env['ISPATIALGYM_CAMPAIGN_CONFIG'])).parent
    group_id = None
    if env.get('ISPATIALGYM_CAMPAIGN_GOAL_PLAN'):
        source_plan = read(env['ISPATIALGYM_CAMPAIGN_GOAL_PLAN'])
        require(source_plan.get('schema', '').startswith('isg-') and source_plan.get('schema', '').endswith('-goal-plan-v1'), 'unsupported goal plan schema')
        require(int(source_plan['max_members']) == payload['identity']['n'], 'plan member count differs')
        plan = dict(source_plan, group_id=token)
        target = directory / 'goal_plan.json'
        if target.exists(): require(read(target) == plan, 'prepared migration plan differs')
        else: shared_json(target, plan)
        set_environment(command, 'ISPATIALGYM_CAMPAIGN_GOAL_PLAN', target)
        group_id = token
    for name in ('source.json', 'contracts.json'):
        path = old_directory / name
        if path.exists(): immutable_bytes(path, directory / name)
    set_option(command, '--base-index', index)
    # The dispatcher assigns a new per-intent ledger. Keeping tag/code/key
    # unchanged permits authoritative family reconciliation and gate reuse.
    result = {'directory':str(directory), 'index':index, 'count':payload['identity']['n'],
              'dispatch_intent':str(root/'dispatch/ready'/(token+'.json')),
              'dispatch_receipt':str(root/'dispatch/receipts'/(token+'.json')), 'command':command}
    if group_id: result['group_id'] = group_id
    shared_json(directory/'migration_source.json', {'source_token':payload['token'],
        'source_command_sha256':payload['command_sha256'], 'source_files':payload['identity']['files'],
        'source_config_sha256':file_digest(env['ISPATIALGYM_CAMPAIGN_CONFIG']),
        'replacement_config_sha256':file_digest(config), 'changed_plan_fields':['group_id'] if group_id else []})
    return result


def prepare(root, limit=4, minimum_age=0):
    require(getpass.getuser() == 'minyeong', 'primary account must reserve and prepare replacements')
    response = subprocess.run(['squeue','-r','-h','-u','minyeong,shiyuc','-o','%i|%T|%u'],
                              capture_output=True, text=True, timeout=40, check=True)
    pending = {row[0]:row[2] for line in response.stdout.splitlines() if len(row:=line.split('|')) == 3 and row[1] == 'PENDING'}
    prepared = []; checked = 0
    for receipt_path in sorted((root/'dispatch/receipts').glob('*.json')):
        if checked >= limit: break
        receipt = read(receipt_path); allocation = receipt.get('allocation', {})
        if receipt.get('status') != 'submitted' or receipt.get('migration') or receipt.get('migration_pending'): continue
        jid = allocation.get('jobid')
        if jid not in pending: continue
        payload = read(root/'dispatch/ready'/receipt_path.name)
        if excluded_controller_work(root, payload): continue
        checked += 1
        try:
            with atomic_directory_lock(root/'dispatch/locks'/(payload['token']+'.lockdir')):
                proof = scheduler_proof(jid)
                if not eligible(proof, owner=pending[jid], minimum_age=minimum_age, now=time.time()): continue
                files = no_execution_artifacts(allocation['job_dir'], root)
                token = payload['token'] + '_pending_m01'
                start = reserve_indices(root, token, int(allocation['n']))
                replacement = fresh_replacement(root, payload, token, start)
                path = root/'dispatch/migrations'/payload['token']/'request.json'
                path.parent.mkdir(parents=True, exist_ok=True)
                for directory in (path.parent.parent, path.parent): directory.chmod(0o2770)
                request = {'schema':'isg-spool-migration-request-v1', 'owner':pending[jid],
                           'source_receipt':str(receipt_path), 'source_payload':payload,
                           'minimum_pending_seconds':minimum_age, 'prepared_at':time.time(),
                           'preparation_execution_proof':files, 'replacement':replacement,
                           'record':{'status':'submitted', 'allocation':allocation}}
                shared_json(path, request)
                receipt['migration_pending'] = {'request':str(path), 'replacement':replacement}
                shared_json(receipt_path, receipt)
                prepared.append(str(path))
        except FileExistsError: continue
        except Exception as error:
            print(json.dumps({'phase':'prepare','receipt':str(receipt_path),'error':str(error)}), flush=True)
    return prepared


def cancel(root, limit=2):
    owner = getpass.getuser(); results = []; checked = 0
    for path in sorted((root/'dispatch/migrations').glob('*/request.json')):
        if checked >= limit: break
        request = read(path)
        if request['owner'] != owner or request['record'].get('migration') or request.get('published_at') or request.get('aborted'): continue
        checked += 1
        try:
            with atomic_directory_lock(path.with_suffix('.lockdir')):
                request = read(path)
                require(input_identity(root, request['source_payload']['command']) == request['source_payload']['identity'], 'source input changed before cancellation')
                result = migrate_record(root, request['record'], minimum_age=request['minimum_pending_seconds'],
                    owner=owner, save=lambda:shared_json(path, request), execute=True)
                if result: results.append(str(path))
                else:
                    after = scheduler_proof(request['record']['allocation']['jobid'])
                    if after['scheduler']['JobState'] != 'PENDING':
                        request['aborted'] = {'reason':'job started or ended before migration cancellation','proof':after}
                        shared_json(path, request)
        except FileExistsError: continue
        except Exception as error:
            print(json.dumps({'phase':'cancel','request':str(path),'error':str(error)}), flush=True)
    return results


def publish(root, limit=4):
    require(getpass.getuser() == 'minyeong', 'primary account publishes replacement work')
    results = []; checked = 0
    for path in sorted((root/'dispatch/migrations').glob('*/request.json')):
        if checked >= limit: break
        request = read(path)
        if request.get('published_at') or not (request['record'].get('migration') or request.get('aborted')): continue
        checked += 1
        try:
            with atomic_directory_lock(path.with_suffix('.lockdir')):
                if request.get('aborted'):
                    receipt_path = Path(request['source_receipt']); receipt = read(receipt_path)
                    receipt.pop('migration_pending', None)
                    receipt['migration_aborted'] = {'request':str(path), **request['aborted']}
                    shared_json(receipt_path, receipt)
                    request['published_at'] = time.time(); shared_json(path, request)
                    continue
                proof = request['record']['migration']; replacement = request['replacement']
                require(proof.get('never_started') and proof['after']['accounting']['state'] == 'CANCELLED'
                        and proof['after']['accounting']['start'] in ('Unknown','None','N/A','')
                        and not proof['final_execution_proof']['nonempty_execution_artifacts'], 'incomplete never-started proof')
                require(proof['original_allocation'] == request['record']['allocation'], 'migration proof allocation differs')
                old_env = environment(request['source_payload']['command'])
                new_env = environment(replacement['command'])
                require(file_digest(new_env['ISPATIALGYM_CAMPAIGN_CONFIG']) ==
                        request['source_payload']['identity']['files'][old_env['ISPATIALGYM_CAMPAIGN_CONFIG']],
                        'replacement changed the exact physical contract')
                proof_path = path.parent/'never_started_proof.json'
                if proof_path.exists(): require(read(proof_path) == proof, 'published proof differs')
                else: shared_json(proof_path, proof)
                token = Path(replacement['directory']).name
                from scripts.campaign_20261110.migration_routes import source_routes
                source_path = root/'dispatch/ready'/(request['source_payload']['token']+'.json')
                require(read(source_path) == request['source_payload'], 'original migration intent changed')
                routing = source_routes(root, {'dispatch_intent':str(source_path)})
                routing['proof'] = {'path':str(proof_path), 'sha256':file_digest(proof_path)}
                enqueue(root, token, replacement['command'], priority=min(10,request['source_payload']['priority']),
                        backends=routing['backends'], migration_routing=routing)
                request['replacement_routing'] = routing
                receipt_path = Path(request['source_receipt']); receipt = read(receipt_path)
                receipt['migration'] = {'never_started':True, 'proof':{'path':str(proof_path),'sha256':file_digest(proof_path)},
                                        'replacement':replacement}
                receipt.pop('migration_pending', None); shared_json(receipt_path, receipt)
                request['published_at'] = time.time(); shared_json(path, request); results.append(token)
        except FileExistsError: continue
        except Exception as error:
            print(json.dumps({'phase':'publish','request':str(path),'error':str(error)}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--phase', required=True, choices=('prepare','cancel','publish'))
    parser.add_argument('--limit', type=int, default=2)
    args = parser.parse_args(); os.umask(0o007)
    require(1 <= args.limit <= 8, 'migration pass must be bounded')
    with atomic_directory_lock(args.root/'dispatch/migrations'/(args.phase+'_'+getpass.getuser()+'.lockdir')):
        print(json.dumps({'phase':args.phase, 'results':globals()[args.phase](args.root.resolve(), limit=args.limit)}), flush=True)


if __name__ == '__main__': main()
