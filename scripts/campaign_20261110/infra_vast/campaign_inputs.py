"""Stage immutable held queues inside new Vast job directories, including journals."""
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil


QUEUE_ENV = "ISPATIALGYM_CAMPAIGN_GOAL_QUEUE"


def dispatch_order(states, config):
    allowed = config.get("dispatch_job_ids")
    allowed = None if allowed is None else {str(jid) for jid in allowed}
    priority = {str(jid): index for index, jid in enumerate(config.get("dispatch_priority_job_ids", []))}
    return sorted((row for row in states if row["status"] == "queued" and
                   (allowed is None or str(row["id"]) in allowed)),
                  key=lambda row: (priority.get(str(row["id"]), len(priority)), int(row["id"])))


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
    require(raw.get("schema") in ("isg-held-goal-queue-v1", "isg-pair-goal-queue-v1") and raw.get("entries"), "invalid campaign goal queue")
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
