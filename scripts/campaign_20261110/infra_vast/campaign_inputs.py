"""Stage immutable held queues inside new Vast job directories, including journals."""
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import sys


QUEUE_ENV = "ISPATIALGYM_CAMPAIGN_GOAL_QUEUE"
QUEUE_SCHEMAS = ('isg-held-goal-queue-v1','isg-pair-goal-queue-v1','isg-surface-goal-queue-v1',
                 'isg-tray-goal-queue-v1','isg-joint-goal-queue-v1','isg-attachment-goal-queue-v1',
                 'isg-block-goal-queue-v1')


def compatible_gpus(requested, config):
    proven = {'RTX 4090', 'RTX 5090'}
    if (config.get('allow_proven_gpu_substitution') and requested in proven
            and float(config['vram_per_job_gb']) <= 20 and int(config['max_jobs_per_instance']) == 1):
        return sorted(proven.intersection(config['gpus']))
    return [requested]


def gpu_matches(job, gpu):
    return not job.get('gpu') or gpu in job.get('compatible_gpus', [job['gpu']])


def worker_can_run(job, instance, worker, creation, timestamp):
    """Require the complete requested runtime plus transfer headroom before expiry."""
    lifetime = float(creation.get('max_lifetime_s', 21600)) - (timestamp - float(creation.get('time', timestamp)))
    return (gpu_matches(job, worker.get('gpu_name')) and job.get('avoid') != instance
            and lifetime >= float(job.get('time_s', 0)) + 900)


def drained_workers(live, config):
    accounts = set(config.get('drain_accounts', ()))
    return {iid for iid, worker in live.items() if worker.get('_key_name') in accounts}


def rental_accounts(accounts, config, live_by_account):
    allowed = set(config.get('rental_accounts', accounts)) - set(config.get('drain_accounts', ()))
    return sorted((name for name in accounts if name in allowed),
                  key=lambda name: (live_by_account.get(name, 0), list(accounts).index(name)))


def drained_worker_finished(instance, draining, reserved, heartbeat):
    return instance in draining and not reserved.get(instance) and not (heartbeat.get('running') or {})


def uncovered_jobs(queued, live, heartbeats, created, assigned_to, timestamp):
    """Booting and ready compatible workers reserve future capacity once."""
    remaining = list(queued)
    for iid, worker in live.items():
        hb, event = heartbeats.get(iid) or {}, created.get(iid) or {}
        if hb.get('phase') in ('destroyed','unusable','stopped','idle_exit'): continue
        slots = max(0,int(hb.get('slots') or event.get('slots') or 1)-len(assigned_to.get(iid,[])))
        for job in list(remaining):
            if not slots: break
            if worker_can_run(job, iid, worker, event, timestamp):
                remaining.remove(job);slots-=1
    return remaining


def staging_record(state_root, job_id):
    """Staging children read one coordinator-owned record and never ingest work."""
    if not str(job_id).isdigit(): raise ValueError('invalid staging job ID')
    value = json.loads((Path(state_root) / (str(job_id) + '.json')).read_text())
    if str(value['id']) != str(job_id): raise ValueError('staging state identity differs')
    return value


def assignment_plan(queued, live, heartbeats, created, assigned_to, timestamp):
    """Reserve each queued job once against compatible, ready worker slots."""
    remaining = list(queued)
    planned = []
    for instance, worker in live.items():
        heartbeat = heartbeats.get(instance) or {}
        if heartbeat.get('phase') not in ('ready', 'run'):
            continue
        free = int(heartbeat.get('slots') or 0) - len(assigned_to.get(instance, []))
        creation = created.get(instance) or {}
        for job in list(remaining):
            if free <= 0:
                break
            if not worker_can_run(job, instance, worker, creation, timestamp):
                continue
            planned.append((instance, worker, job))
            remaining.remove(job)
            free -= 1
    return planned, remaining


def dispatch_order(states, config):
    allowed = config.get("dispatch_job_ids")
    allowed = None if allowed is None else {str(jid) for jid in allowed}
    priority = {str(jid): index for index, jid in enumerate(config.get("dispatch_priority_job_ids", []))}
    rows = [row for row in states if row["status"] == "queued" and
            (allowed is None or str(row["id"]) in allowed)]
    rows = [row for row in rows if production_decision(row, config)['allowed']]
    standard = bool(config.get("prioritize_standard_eval", False))
    recovery_classes = set(config.get("standard_recovery_class_ids", []))
    explicit = set(map(str, config.get("standard_recovery_priority_job_ids", [])))
    if standard:
        try:
            from .cost_provenance import scope
        except ImportError:
            from cost_provenance import scope

    def rank(row):
        tier = 0
        if standard:
            evidence = scope(row)
            split = evidence.get('split')
            if str(row['id']) in explicit:
                tier = 0
            elif evidence.get('status') == 'observed' and split == 'eval_standard':
                tier = 0 if evidence.get('class_id') in recovery_classes else 1
            elif split == 'train':
                tier = 3
            else:
                tier = 2  # unknown/held-out scope is never relabeled standard
        return tier, priority.get(str(row["id"]), len(priority)), int(row["id"])
    return sorted(rows, key=rank)


def production_decision(job, config):
    """Re-read the persistent gate at each dispatch boundary, including retries."""
    # vastctl also runs as a file, without the repository on sys.path.
    source_root = str(Path(__file__).resolve().parents[3])
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    from scripts.campaign_20261110.production_gate import check_job
    root = config.get('campaign_root', '/vision/u/minyeong/projects/isg_home/campaign_20261110')
    return check_job(root, job.get('job_dir', ''), kind=job.get('job_kind', 'collection'))


def prepare_environment(job_dir, root):
    """Let the image's Isaac import select its existing apps directory.

    Cluster account overlays are not transferred to Vast. They must not override
    the image's working default, which the architecture canaries exercised.
    Keep the exact pre-staging environment as evidence inside the new job.
    """
    job, root = Path(job_dir).resolve(), Path(root).resolve()
    require(job.is_relative_to(root / "out"), "Vast environment outside campaign output")
    path = job / "job.env"
    removed = [line for line in path.read_text().splitlines()
               if line.startswith(("ISGF_EXP_PATH=", "EXP_PATH="))]
    if not removed:
        return None
    for line in removed:
        values = shlex.split(line.split("=", 1)[1])
        require(len(values) == 1 and Path(values[0]).resolve().is_relative_to(root / "isaacsim_overlay"),
                "unexpected Isaac apps override")
    before = job / "job.env.before_vast_apps"
    require(not before.exists(), "Isaac apps normalization was already recorded")
    shutil.copyfile(path, before)
    lines = [line for line in path.read_text().splitlines() if line not in removed]
    temporary = path.with_suffix(".env.apps_new")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(path)
    record = {"schema": "isg-vast-apps-environment-v1", "before_sha256": sha(before),
              "after_sha256": sha(path), "removed_overrides": removed,
              "resolution": "native_image_isaac_import_default"}
    (job / "vast_apps_environment.json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def require(value, message):
    if not value:
        raise ValueError(message)


def register_exact_template_customs_inputs(env, root):
    """Extend an existing frozen-source receipt for an exact ready standard source.

    This staging preflight never authorizes a new external file or changes a
    source. It joins a newly authored config to already approved immutable
    customs bytes; the ordinary reader still verifies both hashes afterwards.
    """
    if not env.get('ISPATIALGYM_CAMPAIGN_CONFIG') or not env.get('TEMPLATE'):
        return None
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.campaign_20261110.contracts import ObjectRef, Template, write_json_atomic
    from scripts.campaign_20261110.storage import atomic_directory_lock
    root = Path(root).resolve()
    config_path = Path(env['ISPATIALGYM_CAMPAIGN_CONFIG']).resolve()
    require(config_path.is_relative_to(root), 'customs preflight config outside campaign')
    cfg = json.loads(config_path.read_text())
    if cfg.get('purpose') != 'validation' or cfg.get('split') != 'eval_standard':
        return None
    require(cfg.get('campaign_id') == 'campaign_20261110', 'customs preflight campaign differs')
    template = Path(env['TEMPLATE']).resolve()
    metadata_path = template/'template_meta.json'
    metadata = json.loads(metadata_path.read_text())
    candidates = [metadata.get('customs'), str(template.parent.parent/'task_custom_lists.json')]
    external = [Path(v).resolve() for v in candidates if v and Path(v).is_file() and not Path(v).resolve().is_relative_to(root)]
    if not external:
        return None
    receipt = root/'manifests/external_template_customs_inputs.json'
    require(receipt.is_file(), 'template customs dependency outside owned campaign')
    existing = json.loads(receipt.read_text())
    if all(existing.get('sources', {}).get(str(p), {}).get('templates', {}).get(str(template)) == sha(metadata_path) for p in external):
        return []
    raw = cfg['template']
    template_contract = Template(**{**raw, 'objects': tuple(ObjectRef(**x) for x in raw['objects'])})
    setup = template_contract.scene_setup
    require(Path(setup['source_template']).resolve() == template, 'customs preflight source template differs')
    require(sha(template/'template.json') == setup['source_template_sha256'], 'customs preflight template bytes changed')
    require(sha((template/metadata['activity_definition']).resolve()) == setup['source_bddl_sha256'] == metadata['bddl_sha256'],
            'customs preflight source BDDL changed')
    with atomic_directory_lock(root/'manifests/.external_template_customs_inputs.lock'):
        approved = json.loads(receipt.read_text())
        require(approved.get('schema') == 'isg-exact-external-template-customs-v1' and
                approved.get('campaign_id') == 'campaign_20261110', 'external customs receipt identity differs')
        added = []
        for path in external:
            source = approved.get('sources', {}).get(str(path))
            require(isinstance(source, dict) and source.get('sha256') == sha(path),
                    'customs preflight requires previously frozen external source bytes')
            selected = json.loads(path.read_text())
            require(isinstance(selected, dict) and metadata.get('variant') in selected,
                    'customs preflight selected variant absent')
            old = source.setdefault('templates', {}).get(str(template))
            require(old in (None, sha(metadata_path)), 'customs preflight recorded template metadata changed')
            if old is None:
                source['templates'][str(template)] = sha(metadata_path)
                added.append({'source': str(path), 'source_sha256': source['sha256'],
                              'template': str(template), 'template_metadata_sha256': sha(metadata_path),
                              'config': str(config_path), 'config_sha256': sha(config_path),
                              'template_sha256': sha(template/'template.json'),
                              'source_bddl_sha256': setup['source_bddl_sha256'],
                              'validation_key': cfg['validation_key']})
        if added:
            approved.setdefault('automatic_exact_source_joins', []).extend(added)
            write_json_atomic(receipt, approved)
        return added


def template_customs_inputs(env, root):
    """Retain the template's original customs file across runtime revisions.

    The native sampler consults template metadata before its adjacent customs
    copy. Those dependencies are not necessarily named in job.env or contained
    in the selected native code snapshot. Never rewrite the recorded path.
    """
    root = Path(root).resolve()
    template = Path(env['TEMPLATE']).resolve()
    metadata = json.loads((template / 'template_meta.json').read_text())
    candidates = [metadata.get('customs'), str(template.parent.parent / 'task_custom_lists.json')]
    result = []
    for value in candidates:
        if not value: continue
        path = Path(value).resolve()
        if not path.exists(): continue
        if not path.is_relative_to(root):
            receipt=root/'manifests/external_template_customs_inputs.json'
            require(receipt.is_file(), 'template customs dependency outside owned campaign')
            approved=json.loads(receipt.read_text())
            require(approved.get('schema')=='isg-exact-external-template-customs-v1' and
                    approved.get('campaign_id')=='campaign_20261110','external customs receipt identity differs')
            source=approved.get('sources',{}).get(str(path),{})
            require(source.get('sha256')==sha(path) and
                    source.get('templates',{}).get(str(template))==sha(template/'template_meta.json'),
                    'external template customs bytes or template identity differ')
        require(path.is_file() and isinstance(json.loads(path.read_text()), dict), 'invalid template customs dependency')
        result.append({'path': str(path), 'sha256': sha(path)})
        # multiply_isg selects metadata.variant, which can differ from the
        # counted ALIAS in job.env. Absolute native handle records are indirect
        # inputs of that selected customs entry; preserve their original paths.
        selected = json.loads(path.read_text()).get(metadata.get('variant'), {})
        def handle_paths(value):
            if isinstance(value, dict):
                for child in value.values(): yield from handle_paths(child)
            elif isinstance(value, list):
                for child in value: yield from handle_paths(child)
            elif isinstance(value, str) and Path(value).is_absolute():
                yield Path(value).resolve()
        for role in selected.get('roles', {}).values() if isinstance(selected, dict) else []:
            if not isinstance(role, dict): continue
            for dependency in handle_paths(role.get('handle_file')):
                require(dependency.is_relative_to(root), 'native handle dependency outside owned campaign')
                require(dependency.is_file(), 'native handle dependency missing')
                row = {'path': str(dependency), 'sha256': sha(dependency)}
                if row not in result: result.append(row)
    return result


def camera_context_inputs(env, root):
    """Stage the exact hash-checked private camera proof closure.

    Encrypted asset references are checked by the native contract and remain in
    the existing asset bundle; source scenes, proofs and reviewed images travel
    privately at their unchanged campaign paths.
    """
    if not env.get('ISPATIALGYM_CAMPAIGN_CONFIG'): return []
    root = Path(root).resolve()
    config = json.loads(Path(env['ISPATIALGYM_CAMPAIGN_CONFIG']).read_text())
    template = config['template']
    context = template.get('scene_setup', {}).get('camera_context')
    if not context: return []
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from scripts.campaign_20261110.contracts import ObjectRef, Template
    from scripts.campaign_20261110.camera_context import load_camera_context
    native = dict(template); native['objects'] = tuple(ObjectRef(**row) for row in native['objects'])
    load_camera_context(Template(**native), root=root)
    assets = {(row['path'], row['sha256']) for row in context['asset_files'].values()}
    result = {}

    def visit(value):
        if isinstance(value, list):
            for row in value: visit(row)
        elif isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                reference(value['path'], value['sha256'])
            if 'source_scene' in value and 'source_scene_sha256' in value:
                reference(value['source_scene'], value['source_scene_sha256'])
            for row in value.values(): visit(row)

    def reference(name, expected):
        if (name, expected) in assets: return
        path = Path(name).resolve()
        require(path.is_relative_to(root), 'camera proof dependency outside owned campaign')
        require(path.is_file() and sha(path) == expected, 'camera proof dependency bytes changed')
        if str(path) in result: return
        require(len(result) < 1000, 'camera proof dependency closure exceeds bound')
        result[str(path)] = {'path': str(path), 'sha256': expected}
        if path.suffix == '.json': visit(json.loads(path.read_text()))

    visit(context)
    return list(result.values())


def prepare_cache_environment(job_dir, root):
    """Route new Vast jobs to the exact-key templates restored by boot.sh.

    Native kit_appdata.sh still selects GPU/driver/Behavior identity, requires
    .complete, and clones into private per-process app data. A missing exact key
    remains a logged cold start. No mutable process cache is shared.
    """
    job, root = Path(job_dir).resolve(), Path(root).resolve()
    require(job.is_relative_to(root / 'out'), 'Vast cache job outside campaign output')
    path = job / 'job.env'
    routes = {'KIT_TPL_NFS_ROOT': ('kit_templates', '/vision/u/minyeong/projects/ispatialgym-lab-vec-eval-cache/kit_templates'),
              'KIT_TEX_STORE': ('kit_textures', '/vision/u/minyeong/projects/ispatialgym-lab-vec-eval-cache/kit_textures')}
    lines = path.read_text().splitlines(); changes = []
    for index, line in enumerate(lines):
        key, _, value = line.partition('=')
        if key not in routes: continue
        values = shlex.split(value)
        require(len(values) == 1, 'invalid cache environment path')
        suffix, target = routes[key]
        if values[0] == target: continue
        require(Path(values[0]).resolve() == root / suffix, 'unexpected cluster cache override')
        changes.append({'key': key, 'before': values[0], 'after': target})
        lines[index] = key + '=' + shlex.quote(target)
    if not changes: return None
    before = job / 'job.env.before_vast_caches'
    require(not before.exists(), 'cache normalization was already recorded')
    shutil.copyfile(path, before)
    temporary = path.with_suffix('.env.caches_new')
    temporary.write_text('\n'.join(lines) + '\n'); temporary.replace(path)
    record = {'schema': 'isg-vast-cache-environment-v1', 'before_sha256': sha(before), 'after_sha256': sha(path),
              'changes': changes, 'selection': 'native_exact_gpu_driver_behavior_complete_template',
              'runtime_appdata': 'private_clone_per_process', 'texture_store': 'local_preconverted_links'}
    (job / 'vast_cache_environment.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def runtime_scenes(env, root):
    """Use the actual annotated template, never the counted alias's old scene."""
    root = Path(root).resolve()
    config_path = env.get('ISPATIALGYM_CAMPAIGN_CONFIG')
    if env.get(QUEUE_ENV):
        queue = Path(env[QUEUE_ENV]).resolve()
        require(queue.is_relative_to(root), 'scene queue outside owned root')
        raw = json.loads(queue.read_text())
        config_path = str(checked(raw['config_path'], root, raw['config_sha256']))
    template = Path(env['TEMPLATE']).resolve()
    source_scene = template.name.split('__',1)[0]
    metadata = template/'template_meta.json'
    if metadata.is_file():
        observed_scene = json.loads(metadata.read_text()).get('scene')
        require(isinstance(observed_scene,str), 'native source metadata lacks scene')
        if '__' in template.name:
            require(observed_scene == source_scene, 'native source metadata and conventional scene directory differ')
        source_scene = observed_scene
    if config_path:
        config_path = Path(config_path).resolve()
        require(config_path.is_relative_to(root), 'scene config outside owned root')
        scene = json.loads(config_path.read_text())['template']['scene']
        require(scene == source_scene, 'annotated scene differs from actual source template')
    else:
        scene = source_scene
    require(isinstance(scene, str) and re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', scene), 'invalid texture scene')
    return [scene]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(path, root, expected):
    path = Path(path).resolve()
    require(path.is_relative_to(root.resolve()) and path.is_file(), "queue input outside owned campaign")
    require(sha(path) == expected, "queue input digest differs")
    return path


def prepare_pools(job_dir, env, root):
    """Return exact pool files to ship at their contract-pinned absolute paths."""
    root, job = Path(root).resolve(), Path(job_dir).resolve()
    require(job.is_relative_to(root / "out"), "Vast pool job outside campaign output")
    configs = []
    if env.get("ISPATIALGYM_CAMPAIGN_CONFIG"):
        configs.append(Path(env["ISPATIALGYM_CAMPAIGN_CONFIG"]))
    if env.get(QUEUE_ENV):
        queue = Path(env[QUEUE_ENV]).resolve()
        require(queue.is_relative_to(root) and queue.is_file(), "pool queue outside owned campaign")
        raw = json.loads(queue.read_text())
        configs.append(checked(raw["config_path"], root, raw["config_sha256"]))
    records = {}
    for config in configs:
        config = config.resolve()
        require(config.is_relative_to(root) and config.is_file(), "pool config outside owned campaign")
        raw = json.loads(config.read_text())
        ref = raw.get("template", {}).get("scene_setup", {}).get("binding_pool")
        if ref is None:
            continue
        require(ref.get("schema") == "isg-active-binding-pool-v1", "unsupported binding pool reference")
        path = checked(ref["manifest"], root, ref["manifest_sha256"])
        require(json.loads(path.read_text()).get("schema") == "isg-binding-pool-v1", "unsupported binding pool manifest")
        record = {"manifest": str(path), "manifest_sha256": sha(path)}
        require(str(path) not in records or records[str(path)] == record, "pool references disagree")
        records[str(path)] = record
    if records:
        evidence = {"schema": "isg-vast-pool-inputs-v1", "files": list(records.values()),
                    "resolution": "original_absolute_paths_contract_bytes_unchanged"}
        target = job / "campaign_pool_inputs.json"
        if target.exists():
            require(json.loads(target.read_text()) == evidence, "previous pool input evidence differs")
        else:
            target.write_text(json.dumps(evidence, sort_keys=True, indent=2) + "\n")
    return list(records)


def prepare_queue(job_dir, env, root):
    """Copy only reviewed inputs; runtime events then travel with normal job output."""
    if not env.get(QUEUE_ENV):
        return None
    root, job = Path(root).resolve(), Path(job_dir).resolve()
    require(job.is_relative_to(root / "out"), "Vast queue job outside campaign output")
    source = Path(env[QUEUE_ENV]).resolve()
    require(source.is_relative_to(root / "queues") or source.is_relative_to(job / "campaign_inputs"),
            "queue path outside authored or staged queue roots")
    raw = json.loads(source.read_text())
    require(raw.get('schema') in QUEUE_SCHEMAS and raw.get('entries'), 'invalid campaign goal queue')
    inputs = [(checked(raw["config_path"], root, raw["config_sha256"]), "campaign_config.json")]
    ids = set()
    for entry in raw["entries"]:
        path = checked(entry["path"], root, entry["sha256"])
        plan = json.loads(path.read_text())
        gid = plan["group_id"]
        require(isinstance(gid, str) and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", gid) and gid not in ids,
                "unsafe or duplicate queue group")
        require(plan["campaign_id"] == raw["campaign_id"] and 1 <= plan["max_members"] <= 3,
                "queue campaign/member contract differs")
        ids.add(gid)
        inputs.append((path, "plans/" + gid + ".json"))
    if source.is_relative_to(job / "campaign_inputs"):
        record = json.loads((job / "campaign_queue_inputs.json").read_text())
        require(record["staged_queue"] == str(source) and record["staged_queue_sha256"] == sha(source), "staged queue changed")
        return record
    destination = job / "campaign_inputs"
    for path, relative in inputs:
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            require(sha(target) == sha(path), "previous partial input copy differs")
        else:
            shutil.copyfile(path, target)
    staged = {**raw, "config_path": str(destination / inputs[0][1]),
              "entries": [{**entry, "path": str(destination / relative)}
                          for entry, (_, relative) in zip(raw["entries"], inputs[1:])]}
    queue = destination / "queue.json"
    queue.write_text(json.dumps(staged, sort_keys=True, indent=2) + "\n")
    record = {"schema": "isg-vast-queue-inputs-v1", "original_queue": str(source), "original_queue_sha256": sha(source),
              "staged_queue": str(queue), "staged_queue_sha256": sha(queue),
              "journal": str(queue.with_suffix(".events.jsonl")),
              "input_sha256": {relative: sha(path) for path, relative in inputs}}
    (job / "campaign_queue_inputs.json").write_text(json.dumps(record, indent=2) + "\n")
    env_path = job / "job.env"
    lines = env_path.read_text().splitlines()
    require(sum(line.startswith(QUEUE_ENV + "=") for line in lines) == 1, "missing/duplicate queue environment binding")
    updated = [QUEUE_ENV + "=" + shlex.quote(str(queue)) if line.startswith(QUEUE_ENV + "=") else line for line in lines]
    temporary = env_path.with_suffix(".env.queue_new")
    temporary.write_text("\n".join(updated) + "\n")
    temporary.replace(env_path)
    return record
