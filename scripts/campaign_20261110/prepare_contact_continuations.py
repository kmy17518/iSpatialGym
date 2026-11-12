"""Prepare immutable contact validation tickets; dispatch remains coordinator-owned."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reference(path):
    return {"path": str(path), "sha256": digest(path)}


def write(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n")


RUNNER = '''from pathlib import Path
import collections, hashlib, json, os, subprocess, sys, time
K = Path(__file__).parent
p = json.loads((K / 'prepared.json').read_text())
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
for path, expected in {**p['source_files'], **p['private_model_files'], p['native_library_input']['path']: p['native_library_input']['sha256']}.items():
    assert sha(path) == expected, path
argv = [sys.executable, '-u', '-m', 'ispatialgym.grasping.validate_contacts', '--root', str(K), '--kind', p['kind'], '--arm', p['arm'], '--glib', str(Path(p['private_model']).parent.parent), '--only', p['entry_id'], '--per-group-valid', '3', '--per-group-max', '8']
rec = {'started_at': time.time(), 'cpu_affinity': sorted(os.sched_getaffinity(0)), 'argv': argv, 'runtime': p['native_runtime'], 'job_id': os.environ.get('SLURM_JOB_ID'), 'prepared_sha256': sha(K / 'prepared.json'), 'physical_dedup_key': p['physical_dedup_key']}
(K / 'started.json').write_text(json.dumps(rec, indent=2) + '\\n')
with (K / 'native.log').open('w') as log:
    try:
        rc = subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT, timeout=p['runtime_limit_seconds']).returncode
    except subprocess.TimeoutExpired:
        rc = 124
lib = json.loads(Path(p['native_library_input']['path']).read_text())
verdicts = collections.Counter(e.get('validation', {}).get('verdict', 'unvalidated') for e in lib['entries'])
rec.update(finished_at=time.time(), returncode=rc, done_marker='GB_CONTACT_VALIDATE_DONE' in (K / 'native.log').read_text(errors='replace'), verdicts=dict(verdicts), output_library_sha256=sha(p['native_library_input']['path']), strict_native_reader_required=True, benchmark_credit=0)
(K / 'terminal.json').write_text(json.dumps(rec, indent=2) + '\\n')
sys.exit(0 if rc == 0 and rec['done_marker'] else 2)
'''


def prepare(source_root, account_root, account, inventory, output_relative, relocations=None,
            route_all_account=None):
    source_root, account_root = Path(source_root), Path(account_root)
    inventory = Path(inventory)
    data = json.loads(inventory.read_text())
    owned = json.loads((inventory.parent / "contact_owned_both_accounts.json").read_text())
    owned_keys = {(x["entry_id"], x["kind"]) for x in owned if x.get("submission")}
    jobs = sorted((x for x in data["rows"] if x["status"] == "ready_for_physics"), key=lambda x: (x["entry_id"], x["kind"]))
    root = account_root / output_relative
    if root.exists():
        assert not any(root.iterdir()), "Refuse to replace existing prepared work"
    else:
        root.mkdir(parents=True)
    relocations = json.loads(Path(relocations).read_text()) if relocations else {}
    access = lambda path: Path(relocations.get(str(path), str(path)))
    hashes = {}

    def check(path, expected):
        if path not in hashes:
            hashes[path] = digest(access(path))
        assert hashes[path] == expected, path

    tickets = []
    for index, row in enumerate(jobs):
        expected_account = route_all_account or ("minyeong" if index % 2 == 0 else "shiyuc")
        if account != expected_account:
            continue
        assert (row["entry_id"], row["kind"]) not in owned_keys
        check(row["checkpoint"]["path"], row["checkpoint"]["sha256"])
        checkpoint = json.loads(access(row["checkpoint"]["path"]).read_text())
        runtime = Path(checkpoint["native_runtime"])
        assert (runtime / ".snapshot_commit").read_text().strip() == "3908643f36d0dda4a8427d6aebdc0e81fedd7ce7"
        code = runtime / "ispatialgym/grasping/validate_contacts.py"
        payload = row["payload"]
        check(payload["library"]["path"], payload["library"]["sha256"])
        identity = {"entry_id": row["entry_id"], "kind": row["kind"], "arm": row["validation_arm"], "candidate_sha256": payload["library"]["sha256"], "validator_sha256": digest(code), "native_runtime": str(runtime), "per_group_valid": 3, "per_group_max": 8}
        key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        job_id = "contact_" + key[:20]
        K = root / job_id
        K.mkdir()
        original_model = Path(checkpoint["job"]["model_dir"])
        private_model = K / "glib/models" / row["entry_id"]
        private_model.mkdir(parents=True)
        source_files = {str(access(row["checkpoint"]["path"])): row["checkpoint"]["sha256"], str(access(payload["library"]["path"])): payload["library"]["sha256"], str(code): digest(code), str(runtime / ".snapshot_commit"): digest(runtime / ".snapshot_commit")}
        for path, expected in checkpoint["input_files"].items():
            check(path, expected)
            source_files[str(access(path))] = expected
            if Path(path).parent == original_model:
                target = private_model / Path(path).name
                target.write_bytes(access(path).read_bytes())
                assert digest(target) == expected
        assert all((private_model / n).is_file() for n in ("geometry.json", "geometry.obj", "rests.json", "library.json"))
        original = json.loads(access(payload["library"]["path"]).read_text())
        assert len(original["entries"]) == payload["entries"] > 0
        library = json.loads(json.dumps(original))
        before_source = library["object"].get("source")
        library["object"]["source"] = str(private_model)
        compare = json.loads(json.dumps(library))
        if "source" in original["object"]:
            compare["object"]["source"] = before_source
        else:
            compare["object"].pop("source")
        assert compare == original
        libpath = K / row["kind"] / "models" / row["entry_id"] / "library.json"
        libpath.parent.mkdir(parents=True)
        write(libpath, library)
        prepared = {"account": account, "entry_id": row["entry_id"], "kind": row["kind"], "arm": row["validation_arm"], "checkpoint": row["checkpoint"], "source_payload": payload["library"], "source_files": source_files, "private_model": str(private_model), "private_model_files": {str(p): digest(p) for p in private_model.iterdir()}, "native_library_input": reference(libpath), "native_runtime": str(runtime), "physical_dedup_key": key, "physical_identity": identity, "transport_only_change": {"object.source": {"before": before_source, "after": str(private_model)}}, "entries": payload["entries"], "screen_pass_entries": payload["screen_pass_entries"], "runtime_limit_seconds": 6900, "physics_verified": False, "benchmark_credit": 0}
        write(K / "prepared.json", prepared)
        (K / "run.py").write_text(RUNNER)
        compile(RUNNER, str(K / "run.py"), "exec")
        env = {"ISG_GB_ROOT": str(runtime), "GBH": str(account_root), "KIT_TPL_NFS_ROOT": str(account_root / "kit_templates"), "KIT_TEX_STORE": str(account_root / "kit_textures"), "EXP_PATH": str(account_root / "isaacsim_overlay/apps")}
        inner = "#!/bin/bash\nset -euo pipefail\n" + "\n".join("export " + k + "=" + shlex.quote(v) for k, v in env.items()) + '\nset +u\nsource "$ISG_GB_ROOT/scripts/grasp_bands/env_cluster.sh"\nset -u\nexport PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR="$JB/numba_cache"\nKIT_RC=1\ntrap \'kit_appdata_teardown\' EXIT\npython -u ' + shlex.quote(str(K / "run.py")) + "\nKIT_RC=0\n"
        (K / "job.sh").write_text(inner)
        (K / "pinned.sh").write_text("#!/bin/bash\nset -euo pipefail\nexec srun --cpu-bind=cores --exact -n1 -c8 bash " + shlex.quote(str(K / "job.sh")) + "\n")
        for name in ("job.sh", "pinned.sh"):
            subprocess.run(["bash", "-n", str(K / name)], check=True)
        argv = ["sbatch", "--parsable", "--nodes=1", "--ntasks=1", "-A", "vision", "-p", "svl,viscam", "--exclude=svl9", "--gres=gpu:titanrtx:1", "-c", "8", "--mem=48G", "--time=02:00:00", "--nice=1000", "-J", "isg261110_contact_" + row["kind"], "-o", str(K / "slurm_%j.log"), str(K / "pinned.sh")]
        global_root = source_root / output_relative
        ticket = {"id": job_id, "expected_account": account, "physical_dedup_key": key, "dedup_key": key, "claim_path": str(global_root / "claims" / (key + ".json")), "submission_path": str(global_root / "receipts" / (key + ".json")), "sbatch_argv": argv, "prepared": reference(K / "prepared.json"), "request": reference(inventory), "runtime_ref": reference(runtime / ".snapshot_commit"), "input_refs": [{"path": p, "sha256": h} for p, h in source_files.items()], "scripts": [reference(K / n) for n in ("run.py", "job.sh", "pinned.sh")], "output": str(K), "dispatch_workdir": str(runtime), "hard_time_seconds": 7200, "native_time_seconds": 6900, "status": "prepared_unsubmitted", "workstream": "training preparation", "stage": "contact physics validation", "owner": "/root/templates_assets", "entry_id": row["entry_id"], "kind": row["kind"], "arm": row["validation_arm"], "selected_dependency_ids": row["selected_dependency_ids"], "benchmark_credit": 0, "production_authorized": False, "followup": {"verification_owner": "/root/templates_assets", "required": "strict native contact-library reader, exact kind/arm/source identity and positive entries", "positive_next_stage": "feed exact measured height and verified library into existing contact map intake", "unchanged_zero_retry": False}}
        assert not Path(ticket["claim_path"]).exists() and not Path(ticket["submission_path"]).exists()
        tickets.append(ticket)
    manifest = root / "tickets.json"
    write(manifest, {"schema": "isg-preparation-dispatch-tickets-v1", "campaign_id": "campaign_20261110", "account": account, "source": reference(inventory), "owner": "/root/templates_assets", "dispatch_owner": "/root", "jobs": tickets, "benchmark_credit": 0, "production_authorized": False})
    return {"manifest": reference(manifest), "jobs": len(tickets), "account": account}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--account-root", required=True)
    parser.add_argument("--account", choices=("minyeong", "shiyuc"), required=True)
    parser.add_argument("--inventory", required=True)
    parser.add_argument("--output-relative", required=True)
    parser.add_argument("--relocations")
    parser.add_argument("--route-all-account", choices=("minyeong", "shiyuc"))
    print(json.dumps(prepare(**vars(parser.parse_args()))))
