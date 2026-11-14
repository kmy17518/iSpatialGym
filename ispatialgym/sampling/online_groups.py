"""Persistent, acceptance-driven grouped sampling for eval and training.

The sampler owns goal discovery. No discovery job or published plan queue is
consumed. A screened first draw supplies the goal and remains a candidate member;
its plan is sealed before the collector-visible summary is published. The same
sampler/world/cache objects serve the next group. Plans and state are durable
provenance and crash recovery, never a second production pass.
"""
from __future__ import annotations

import copy
import json
import math
import os
import re
import time
import uuid
from pathlib import Path

from scripts.campaign_20261110.contracts import digest, file_digest, require, write_json_atomic
from ispatialgym.sampling import goal_identity as identity

ENV = 'ISPATIALGYM_ONLINE_GROUPS'
PLAN_ENV = 'ISPATIALGYM_CAMPAIGN_GOAL_PLAN'
CONFIG_ENV = 'ISPATIALGYM_CAMPAIGN_CONFIG'
FIXED_CLOSING_CLASSES = frozenset(
    'configuration_matching-articulation_close_large_scale_' + suffix
    for suffix in ('01', '02', '03'))


def read(path):
    return json.loads(Path(path).read_text())


def ref(path):
    return {'path': str(Path(path).resolve()), 'sha256': file_digest(path)}


def parse_plan(raw, activated_at):
    """Use the native family validators and their native group summaries."""
    from ispatialgym.sampling import campaign_goal_queue as q
    kw = dict(template_id=raw['template_id'], scene=raw['scene'], rooms=[raw['room']],
              activated_at=activated_at, now=time.time())
    schema = q.schema_for_task(raw['task'])
    if schema == q.SCHEMA:
        from ispatialgym.sampling.campaign_held_goals import validate_goal_plan
        return validate_goal_plan(raw, task=raw['task'], **kw)
    if schema == q.PAIR_SCHEMA:
        from ispatialgym.sampling.campaign_pair_goals import validate_pair_plan
        relation, scale, _ = raw['task'].split('-', 1)[1].split('_')
        return validate_pair_plan(raw, relation=relation, scale=scale, **kw)
    if schema == q.SURFACE_SCHEMA:
        from ispatialgym.sampling.campaign_surface_goals import validate_surface_plan
        return validate_surface_plan(raw, **kw)
    if schema == q.TRAY_SCHEMA:
        from ispatialgym.sampling.campaign_tray_goals import validate_tray_plan
        return validate_tray_plan(raw, **kw)
    if schema == q.ATTACHMENT_SCHEMA:
        from ispatialgym.sampling.campaign_attachment_goals import validate_attachment_plan
        return validate_attachment_plan(raw, **kw)
    if schema == q.BLOCK_SCHEMA:
        from ispatialgym.sampling.campaign_block_goals import validate_block_plan
        return validate_block_plan(raw, **kw)
    from ispatialgym.sampling.campaign_joint_goals import validate_joint_plan
    return validate_joint_plan(raw, target=raw['target'], joint=raw['joint'], **kw)


class GoalRegistry:
    """One atomic directory lock per class on the shared coordination filesystem.

    No rounded-hash-only deduplication: compare all native goal fields under the
    lock, including quaternion sign and numerical-noise tolerances. A crashed
    lock is never stolen; its recorded owner must recover it. Remote workers use
    the request/reply broker, which invokes this same transaction.
    """
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def claim(self, record, owner, deadline):
        cid = record['class_id']
        require(re.fullmatch(r'[A-Za-z0-9_.-]+', cid), 'unsafe class ID')
        identity.physical_goal(record)  # fail closed before acquiring the lock
        exception = None
        policy_path = self.root / 'quota_policy.json'
        if policy_path.exists():
            policy = read(policy_path)
            require(policy.get('schema') == 'isg-goal-quota-policy-v1'
                    and policy.get('authorized_by') == 'user'
                    and policy.get('native_goal_definition_unchanged') is True,
                    'invalid goal quota authorization')
            exceptions = set(policy.get('fixed_goal_exceptions', []))
            require(exceptions <= FIXED_CLOSING_CLASSES, 'unauthorized goal quota exception')
            if cid in exceptions: exception = ref(policy_path)
        directory = self.root / cid
        directory.mkdir(exist_ok=True)
        lock = directory / '.lock'
        while True:
            try:
                lock.mkdir()
                break
            except FileExistsError:
                if time.time() >= deadline:
                    raise TimeoutError('goal registry class lock: ' + cid)
                time.sleep(.1)
        try:
            write_json_atomic(lock / 'owner.json', {'owner': owner, 'pid': os.getpid(), 'at': time.time()})
            for path in directory.glob('*.json'):
                old = read(path)
                if old['owner'] == owner:
                    require(identity.equivalent_goal(old['record'], record), 'resume owner changed physical goal')
                    return {'claimed': True, 'receipt': ref(path), 'resumed': True}
                if identity.equivalent_goal(old['record'], record) and exception is None:
                    return {'claimed': False, 'duplicate_of': ref(path)}
            path = directory / (digest(owner) + '.json')
            write_json_atomic(path, {'owner': owner, 'record': record, 'claimed_at': time.time(),
                                    'fixed_goal_quota_exception': exception})
            return {'claimed': True, 'receipt': ref(path), 'resumed': False,
                    'fixed_goal_quota_exception': exception}
        finally:
            (lock / 'owner.json').unlink(missing_ok=True)
            lock.rmdir()


def saved_member(tro_path):
    t = read(tro_path); m = t['ispatialgym']; c = m['campaign']; runtime = c['runtime']; g = runtime['goal_group']
    side = read(Path(tro_path).with_name(m['instance_id'] + '.oracle.json'))
    art = side.get('articulation')
    return {'class_id': c['class_id'], 'instance_id': m['instance_id'], 'requested_goal': g['requested_goal'],
            'goal_group': g, 'requested_start': g.get('requested_start'),
            'object_states': {k: {'root_link': v.get('root_link'), 'joint_pos': v.get('joint_pos')}
                              for k, v in t.items() if k in m['object_bindings'] and isinstance(v, dict)},
            'robot_poses': t.get('robot_poses'), 'robot_state': m.get('robot_state'),
            'actual_grasps': [{k: x.get(k) for k in ('object', 'arm', 'eef_in_object', 'finger_q')}
                              for x in side.get('grasps', [])],
            'actual_articulation_grasp': ({'object': art.get('object'), 'arm': art.get('base_follow_path', {}).get('arm'),
                                          'initial_waypoint': art.get('waypoints', {}).get('0.0')} if art else None)}


def verify_result(candidate, result, accepted):
    """Online admission; downstream immutable dataset allocation stays separate."""
    require(result.get('success') is True and result.get('status') == 'success' and result.get('rc') == 0,
            'oracle did not pass')
    require(result.get('provisional') is False and result.get('hdf5_ok') is True, 'demo provisional or HDF5 invalid')
    require(result.get('finished_at_unix', math.inf) <= candidate['deadline'], 'member deadline exceeded')
    require(Path(result['tro']).resolve() == Path(candidate['tro']['path']).resolve(), 'result TRO differs')
    require(file_digest(candidate['tro']['path']) == candidate['tro']['sha256'], 'published TRO changed')
    require(file_digest(candidate['sidecar']['path']) == candidate['sidecar']['sha256'], 'published sidecar changed')
    side = read(candidate['sidecar']['path'])
    require(side['tro_sha256'] == candidate['tro']['sha256'], 'sidecar/TRO mismatch')
    meta_path = Path(result['out']) / 'meta.json'
    meta = read(meta_path)
    require(meta.get('instance_id') == candidate['instance_id'] and meta.get('mode') == 'oracle'
            and meta.get('success') is True and meta.get('status') == 'success', 'oracle report mismatch')
    qa = meta.get('qa') or {}
    require(qa.get('enabled') is True and not qa.get('hard_triggered'), 'hard QA failed or disabled')
    require(not any(x.get('triggered') for x in (qa.get('metrics') or {}).values() if x.get('mode') == 'hard'),
            'hard QA metric failed')
    demo = Path(result['out']) / 'demo.hdf5'
    require(demo.is_file() and demo.stat().st_size > 0, 'missing demo')
    state = saved_member(candidate['tro']['path'])
    for previous in accepted:
        comparison = identity.start_comparison(state, saved_member(previous['tro']['path']))
        require(comparison['different'], 'no meaningful saved-state difference from accepted member')
    return {'demo': ref(demo), 'report': ref(meta_path), 'result': result}


class OnlineGroups:
    def __init__(self, sampler, settings_path, start_index):
        from ispatialgym.sampling.v2.core import publish_after_screens
        from ispatialgym.sampling.campaign_runtime import configuration
        require(publish_after_screens(), 'online groups require atomic publication after screens')
        require(not os.environ.get('ISPATIALGYM_CAMPAIGN_GOAL_QUEUE') and not os.environ.get(PLAN_ENV),
                'online sampling cannot consume an external goal plan/queue')
        self.sampler = sampler
        self.settings_path = Path(settings_path).resolve()
        self.options = read(self.settings_path)
        self.config_path = os.environ[CONFIG_ENV]
        self.config, self.template, _, _ = configuration()
        if self.template.split != self.config['split']:
            from dataclasses import replace
            self.template = replace(self.template, split=self.config['split'])
        if self.options.get('wallclock_policy'):
            from scripts.campaign_20261110.class_wallclock import resolve
            self.options.update(resolve(self.options['wallclock_policy'],self.template.class_id,self.config['split']))
        self.target = int(self.options.get('target_members', 3))
        self.minimum = int(self.options.get('minimum_members', 2 if self.config['split'] == 'eval_standard' else 1))
        require(1 <= self.minimum <= self.target <= 3, 'invalid accepted-member quota')
        require(self.config['split'] != 'eval_standard' or self.minimum >= 2, 'standard eval needs siblings')
        for field in ('group_limit_s', 'member_limit_s'):
            require(math.isfinite(self.options[field]) and self.options[field] > 0, 'invalid ' + field)
        require(self.options['groups'] > 0 and self.options['index_count'] > 0, 'empty online allocation')
        self.root = Path(self.options['state_dir']); self.root.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / 'state.json'
        self.results_path = Path(self.options['results'])
        self.registry = GoalRegistry(self.options['registry']) if self.options.get('registry') else None
        require(self.registry is not None or self.options.get('registry_mailbox'), 'shared uniqueness registry required')
        self.original_cells = copy.deepcopy(sampler.place['cells']) if isinstance(getattr(sampler, 'place', None), dict) and 'cells' in sampler.place else None
        self.original_pairs = copy.deepcopy(getattr(sampler, '_pairs', None))
        self.state = read(self.state_path) if self.state_path.exists() else {
            'schema': 'isg-online-groups-v1', 'settings_sha256': file_digest(self.settings_path),
            'config_sha256': file_digest(self.config_path), 'owner': self.options['worker_id'],
            'next_index': int(start_index), 'index_end': int(start_index) + self.options['index_count'],
            'groups': [], 'current': None}
        require(self.state['settings_sha256'] == file_digest(self.settings_path), 'resume settings changed')
        require(self.state['config_sha256'] == file_digest(self.config_path), 'resume template/config changed')
        # Parked groups: target reachable by accepted + in-flight members; the
        # sampler has moved on and they finalize when their members resolve.
        self.state.setdefault('open', [])
        self.member_deadline = None
        # A crash between sealing and publication leaves a known screened
        # pending summary. Republish only those exact immutable candidates.
        resumed = ([self.state['current']] if self.state['current'] else []) + self.state['open']
        if resumed:
            from ispatialgym.sampling.v2.core import publish_instance
            for member in (m for g in resumed for m in g['members']):
                if member['status'] == 'pending' and time.time() < member['deadline']:
                    require(file_digest(member['tro']['path']) == member['tro']['sha256'], 'resume TRO changed')
                    require(file_digest(member['sidecar']['path']) == member['sidecar']['sha256'], 'resume sidecar changed')
                    publish_instance(member['tro']['path'])
        self.event('worker_started', world_id=id(sampler.world), kit_id=id(sampler.kit))
        self.save()

    def event(self, event, **data):
        row = {'event': event, 'at': time.time(), 'pid': os.getpid(), 'world_id': id(self.sampler.world),
               'worker': self.state['owner'], **data}
        with (self.root / 'events.jsonl').open('a') as f:
            f.write(json.dumps(row, sort_keys=True) + '\n'); f.flush(); os.fsync(f.fileno())

    def save(self):
        write_json_atomic(self.state_path, self.state)

    def reset_goal(self):
        from ispatialgym.sampling.campaign_goal_queue import RESET_FIELDS
        for key in RESET_FIELDS:
            if hasattr(self.sampler, key): delattr(self.sampler, key)
        os.environ.pop(PLAN_ENV, None)
        os.environ[CONFIG_ENV] = self.config_path
        if self.original_cells is not None: self.sampler.place['cells'] = copy.deepcopy(self.original_cells)
        if self.original_pairs is not None:
            self.sampler._pairs = copy.deepcopy(self.original_pairs)
            self.sampler._pair_drawn = False

    def activate(self):
        group = self.state['current']
        self.reset_goal()
        if not group.get('plan'): return
        require(file_digest(group['plan']['path']) == group['plan']['sha256'], 'resume plan changed')
        require(file_digest(group['config']['path']) == group['config']['sha256'], 'resume group config changed')
        os.environ[PLAN_ENV] = group['plan']['path']; os.environ[CONFIG_ENV] = group['config']['path']
        # Native loaders must use the original wallclock, including discovery,
        # collector waits and downtime. Resume never grants a fresh budget.
        self.sampler._campaign_goal_started_at = group['started_at']
        self.sampler._campaign_group_started_at = group['started_at']

    def claim_goal(self, record, group):
        owner = self.state['owner'] + '/' + group['id']
        if self.registry is not None:
            return self.registry.claim(record, owner, min(group['deadline'], self.member_deadline))
        from ispatialgym.sampling.online_registry import mailbox_claim
        return mailbox_claim(self.options['registry_mailbox'], record, owner,
                             min(group['deadline'], self.member_deadline))

    def before_publish(self, draft, path, seed):
        """Executed inside the sampler, after screens but before queue publication."""
        from ispatialgym.sampling.v2.core import instance_summary_path, Reject
        from ispatialgym.sampling.campaign_runtime import annotate
        from scripts.campaign_20261110.goal_groups import prepare_group
        group = self.state['current']
        if time.time() >= self.member_deadline:
            raise Reject('online_member_deadline', 'member budget spent before publication')
        path = Path(path); summary_path = Path(instance_summary_path(str(path), True))
        tro = read(path); summary = read(summary_path); iid = tro['ispatialgym']['instance_id']
        side_path = path.with_name(iid + '.oracle.json')
        if not group.get('plan'):
            # Copy private proposal evidence before binding metadata; these
            # immutable sources remain valid after the final TRO is sealed.
            proposal = self.root / 'proposals' / (str(self.state['next_index'] - 1) + '-' + uuid.uuid4().hex)
            proposal.mkdir(parents=True, exist_ok=False)
            for name, source in [('tro', path), ('summary', summary_path), ('sidecar', side_path)]:
                (proposal / (name + '.json')).write_bytes(source.read_bytes())
            runtime = tro['ispatialgym']['campaign']['runtime']
            self.sampler.write_template_meta()
            anchor = {'instance_id': iid, 'tro': tro, 'summary': summary,
                      'template_meta': self.sampler.template_meta, 'template_dir': str(path.parent.parent),
                      'sources': {name: ref(proposal / (name + '.json')) for name in ('tro', 'summary', 'sidecar')},
                      'allocation': {'worker': self.state['owner'], 'instance_index': self.state['next_index'] - 1},
                      'goal_proposal_source': 'online_native_candidate_before_publication',
                      'grasp_library_root': os.environ.get('ISPATIALGYM_GRASP_LIBRARY_V2'),
                      **{k: runtime.get(k, {}) for k in ('planned_goal_poses', 'planned_start_poses', 'bound_object_poses', 'planned_goal_joints', 'joint_names')}}
            output = proposal / 'group'
            key = self.config['validation_key']
            prepare_group(anchor=anchor, template=self.template, campaign_id=self.config['campaign_id'],
                          group_id=group['id'], output=output, code_sha256=key['code_sha256'],
                          physics_sha256=key['physics_sha256'], data_sha256=key['data_sha256'],
                          magnitude_range=self.options['magnitude_range'], budget_seconds=self.options['group_limit_s'],
                          max_members=3, reuse_anchor_goal=True)
            plan = read(output / 'goal_plan.json')
            # The discovery time is part of this group's deadline. Provenance
            # accurately identifies a jointly proposed first start and goal.
            plan['created_at'] = group['started_at']
            plan['source_instance_is_member'] = True
            plan['online_worker'] = self.state['owner']
            write_json_atomic(output / 'goal_plan.json', plan)
            parsed = parse_plan(plan, group['started_at'])
            native = parsed.summary()
            record = {'class_id': self.template.class_id, 'requested_goal': native['requested_goal'], 'goal_group': native}
            claim = self.claim_goal(record, group)
            if not claim['claimed']:
                self.event('duplicate_goal_rejected', instance_id=iid, claim=claim)
                raise Reject('online_duplicate_goal', 'native goal already claimed by another group')
            group.update(plan=ref(output / 'goal_plan.json'), config=ref(output / 'campaign_config.json'),
                         goal=record, claim=claim, first_candidate=iid)
            self.save()  # crash before publication resumes this exact claim
            self.activate()
            from ispatialgym.sampling.campaign_class_routes import resolve
            if resolve(read(group['config']['path'])) is not None:
                # The fresh native draw supplied a goal, but did not execute
                # the required source-bound recipe. Keep its evidence privately
                # and generate the first eligible member using that recipe.
                group['first_candidate_retained'] = False
                self.save()
                self.event('goal_selected_proposal_only', group_id=group['id'], goal=record,
                           reason='class_recipe_requires_goal_before_start')
                raise Reject('online_goal_requires_routed_member', 'goal claimed; next draw must execute the class recipe')
            group['first_candidate_retained'] = True
            source = read(output / 'source.json')
            source.update(anchor_is_final_member=True, status='online_candidate_awaiting_oracle')
            write_json_atomic(output / 'source.json', source)
            native['phase'] = 'online_first_candidate_before_publication'
            native['requested_start'] = {'native_first_candidate': True, 'planned_start_poses': runtime['planned_start_poses'],
                                         'planned_start_joints': runtime['planned_start_joints']}
            draft.summary['campaign_goal_group'] = native
            self.event('goal_selected', group_id=group['id'], goal=record, first_candidate=iid, claim=claim)
        else:
            native = draft.summary.get('campaign_goal_group') or {}
            record = {'class_id': self.template.class_id, 'requested_goal': native.get('requested_goal'), 'goal_group': native}
            require(identity.equivalent_goal(group['goal'], record), 'member changed its group native goal')
        draft.summary['online_group'] = {'worker': self.state['owner'], 'group_id': group['id'],
                                         'member_deadline': self.member_deadline, 'group_deadline': group['deadline'],
                                         'sampler_pid': os.getpid(), 'world_id': id(self.sampler.world)}
        tro = annotate(self.sampler, tro, draft, seed)
        tro['ispatialgym']['campaign']['runtime']['online_group'] = draft.summary['online_group']
        write_json_atomic(path, tro)
        # Hash-bound sidecar is resealed before the atomic collector publication.
        side = read(side_path); side['tro_sha256'] = file_digest(path); write_json_atomic(side_path, side)
        summary.update(campaign_goal_group=draft.summary['campaign_goal_group'], online_group=draft.summary['online_group'])
        write_json_atomic(summary_path, summary)
        candidate = {'instance_id': iid, 'tro': ref(path), 'sidecar': ref(side_path), 'group_id': group['id'],
                     'deadline': self.member_deadline, 'status': 'pending', 'published_at': time.time()}
        group['members'].append(candidate)
        self.save()

    def results(self):
        rows = {}
        if self.results_path.exists():
            for line in self.results_path.read_text().splitlines():
                try:
                    row = json.loads(line); rows[row['instance']] = row
                except (ValueError, KeyError): continue  # partial O_APPEND line
        return rows

    def collect(self):
        return self.collect_group(self.state['current'], self.results())

    def collect_group(self, group, rows):
        accepted = [m for m in group['members'] if m['status'] == 'accepted']
        for member in group['members']:
            if member['status'] != 'pending': continue
            require(member.get('group_id', group['id']) == group['id'], 'member attributed to another group')
            result = rows.get(member['instance_id'])
            if result is not None:
                try:
                    member['evidence'] = verify_result(member, result, accepted)
                    member['status'] = 'accepted'; accepted.append(member)
                except (ValueError, OSError, KeyError) as exc:
                    member.update(status='rejected', reason=str(exc))
            elif time.time() >= member['deadline']:
                member.update(status='expired', reason='member_deadline_without_success')
            else:
                continue
            self.event('member_' + member['status'], group_id=group['id'], instance_id=member['instance_id'],
                       reason=member.get('reason'), accepted=len(accepted))
            self.save()
        return len(accepted), sum(m['status'] == 'pending' for m in group['members'])

    @staticmethod
    def counts(group):
        return (sum(m['status'] == 'accepted' for m in group['members']),
                sum(m['status'] == 'pending' for m in group['members']))

    def finalize(self, group, accepted):
        for member in group['members']:
            if member['status'] == 'pending': member.update(status='expired', reason='group_deadline')
        group.update(finished_at=time.time(), accepted_members=accepted,
                     qualifies=accepted >= self.minimum, reason='target_reached' if accepted >= self.target else 'budget_or_indices_exhausted')
        times = sorted(m['evidence']['result']['finished_at_unix']-group['started_at'] for m in group['members'] if m['status']=='accepted')
        group['timing'] = {'class_id':self.template.class_id,'group_seconds':group['finished_at']-group['started_at'],
            'seconds_to_accepted_members':times,'policy':self.options.get('wallclock_policy'),
            'group_limit_s':self.options['group_limit_s'],'member_limit_s':self.options['member_limit_s'],
            'censored':not group['qualifies']}
        self.event('group_finalized', group_id=group['id'], accepted=accepted, qualifies=group['qualifies'], reason=group['reason'],timing=group['timing'])
        self.state['groups'].append(group)

    def poll_open(self):
        """Resolve parked groups from collector results; finalize resolved ones."""
        if not self.state['open']: return
        rows = self.results()
        for group in list(self.state['open']):
            accepted, pending = self.collect_group(group, rows)
            if pending and time.time() < group['deadline']: continue
            group['parked_resolution'] = 'members_resolved' if not pending else 'group_deadline'
            self.state['open'].remove(group)
            self.finalize(group, accepted)
            self.save()

    def run(self, max_attempts):
        sampler = self.sampler
        original_deadline = getattr(sampler, 'deadline', None)
        poll_s = getattr(self, 'poll_s', 1.)
        self.state.setdefault('open', [])
        paths = []
        try:
            require_qualifying = self.options.get('require_qualifying_groups')
            def completed():
                return sum(g['qualifies']for g in self.state['groups'])if require_qualifying else len(self.state['groups'])
            def committed():
                # Upper bound on the final total: a parked group can only lose
                # in-flight members, so one that cannot reach the minimum never will.
                return completed() + sum(1 for g in self.state['open']
                                         if not require_qualifying or sum(self.counts(g)) >= self.minimum)
            while True:
                self.poll_open()
                if self.state['current'] is None:
                    if (completed() >= self.options['groups'] or self.state['next_index'] >= self.state['index_end']
                            or (original_deadline and time.time() >= original_deadline)):
                        break
                    if committed() >= self.options['groups']:
                        time.sleep(poll_s); continue  # a parked group may still finalize short
                    now = time.time()
                    self.state['current'] = {'id': 'online-' + uuid.uuid4().hex, 'started_at': now,
                                             'deadline': min(now + self.options['group_limit_s'], original_deadline or math.inf),
                                             'members': []}
                    self.reset_goal(); self.save(); self.event('group_started', group_id=self.state['current']['id'])
                elif original_deadline and time.time() >= original_deadline: break
                group = self.state['current']
                if group.get('plan') and time.time() < group['deadline']: self.activate()
                parked = False
                while time.time() < group['deadline']:
                    self.poll_open()
                    accepted, pending = self.collect()
                    if accepted >= self.target: break
                    if accepted + pending >= self.target:
                        parked = True; break  # move on; collectors finish this group's members
                    if self.state['next_index'] >= self.state['index_end']: break
                    index = self.state['next_index']; self.state['next_index'] += 1; self.save()
                    self.member_deadline = min(group['deadline'], time.time() + self.options['member_limit_s'])
                    sampler.deadline = self.member_deadline
                    # Legacy family counters count writes. For online operation
                    # only accepted and in-flight members occupy the quota.
                    sampler._campaign_group_written = accepted + pending
                    from scripts.campaign_20261110.online_deadline_watch import process
                    self.state['sampling_operation'] = {**process(os.getpid()), 'active': True,
                                                        'member_deadline': self.member_deadline, 'index': index}
                    self.save()
                    try:
                        paths.extend(sampler.run_instances(1, max_attempts=max_attempts, indices=[index]))
                    finally:
                        self.state['sampling_operation']['active'] = False
                        self.save()
                    self.collect()
                if parked:
                    group['parked_at'] = time.time()
                    self.state['open'].append(group); self.state['current'] = None
                    self.reset_goal(); self.save()
                    self.event('group_parked', group_id=group['id'], accepted=accepted, pending=pending)
                    continue
                accepted, pending = self.collect()
                self.finalize(group, accepted)
                self.state['current'] = None; self.save()
            # Parked members resolve in the collectors; each group is bounded by
            # its own deadline, so this wait is bounded.
            while self.state['open']:
                self.poll_open()
                if self.state['open']: time.sleep(poll_s)
            return paths
        finally:
            sampler.deadline = original_deadline
            self.reset_goal()
            self.save()
            self.event('worker_stopped', finalized=len(self.state['groups']))


def run(sampler, n, max_attempts, start_index, indices):
    settings = read(os.environ[ENV])
    root = Path(settings['state_dir']); root.mkdir(parents=True, exist_ok=True)
    lock = root / '.worker_active'
    try:
        lock.mkdir()
    except FileExistsError:
        raise RuntimeError('worker already active or requires crash-owner recovery: ' + str(lock))
    write_json_atomic(lock / 'owner.json', {'pid': os.getpid(), 'host': os.uname().nodename, 'at': time.time()})
    try:
        start = (list(indices)[0] if indices is not None else sampler.existing_instances() if start_index is None else start_index)
        worker = OnlineGroups(sampler, os.environ[ENV], start)
        sampler._online_groups = worker
        try:
            return worker.run(max_attempts)
        finally:
            del sampler._online_groups
    finally:
        (lock / 'owner.json').unlink(missing_ok=True)
        lock.rmdir()
