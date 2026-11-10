"""Measure collection GPU-hours separately from setup and predict required capacity."""
import argparse
import csv
import datetime
import json
from pathlib import Path
import subprocess
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", required=True, type=Path)
    p.add_argument("--job", action="append", required=True)
    p.add_argument("--deadline", default="2026-11-12T19:00:00+00:00")
    p.add_argument("--target-train", default=100000, type=int)
    a = p.parse_args()
    root = a.root
    allocations = []
    for path in (root / "manifests").glob("*allocations.tsv"):
        with path.open() as stream:
            allocations.extend(csv.DictReader(stream, delimiter="\t"))
    accepted = []
    for path in (root / "dataset/groups").glob("*.json"):
        accepted.extend(json.loads(path.read_text())["members"])
    camera_count = 0
    for name in ("camera_quota.json", "camera_obstructed_quota.json"):
        path = root / "dataset" / name
        if path.exists():
            rows = list(json.loads(path.read_text())["accepted"].values())
            camera_count += sum(row["split"] == "train" for row in rows)
    group_counts = json.loads((root / "dataset/group_counts.json").read_text())["counts"]
    train_count = group_counts.get("train", 0) + camera_count
    now = time.time()
    remaining_hours = (datetime.datetime.fromisoformat(a.deadline).timestamp() - now) / 3600
    required = max(0, a.target_train - train_count) / max(remaining_hours, .001)
    observed = []
    for jid in a.job:
        rows = [row for row in allocations if row["jobid"] == jid]
        if len(rows) != 1:
            continue
        row = rows[0]
        job = Path(row["job_dir"])
        account = subprocess.check_output(["sacct", "-j", jid, "-n", "-P", "--format=JobIDRaw,ElapsedRaw,State"], text=True)
        accounting = next((line.split("|") for line in account.splitlines() if line.split("|")[0] == jid), None)
        if accounting is None:
            continue
        elapsed = max(1, int(accounting[1]))
        reports = [json.loads(path.read_text()) for path in (job / "demo").glob("*/meta.json")]
        successes = [report for report in reports if report.get("success") is True and report.get("status") == "success"]
        completed_times = sorted(datetime.datetime.fromisoformat(report["started_utc"].replace("Z", "+00:00")).timestamp()
                                 + float(report.get("wall_time_s", 0)) for report in successes)
        steady = ((len(completed_times) - 1) * 3600 / (completed_times[-1] - completed_times[0])) if len(completed_times) > 1 and completed_times[-1] > completed_times[0] else None
        rate = len(successes) * 3600 / elapsed
        record = {"job_id": jid, "job_dir": str(job), "key": row["key"], "state": accounting[2], "gpu_count": 1,
                  "elapsed_gpu_seconds": elapsed, "attempted_demos": len(reports), "successes": len(successes),
                  "success_rate_completed_demos": len(successes) / max(1, len(reports)),
                  "successes_per_gpu_hour_with_startup": rate,
                  "successes_per_gpu_hour_between_completed_demos": steady,
                  "gpus_if_this_lane_rate_generalized": required / rate if rate else None,
                  "warning": "Collection-only rate; excludes prior template validation, dependency prebuild and observation rendering. Goal groups also reserve train siblings and evaluation quotas."}
        stats = job / "job_stats.json"
        if stats.exists():
            data = json.loads(stats.read_text())
            record["sampler"] = {key: data.get(key) for key in ("world_load_s", "attempts", "written", "sample_wall_s", "rejections")}
        observed.append(record)
    result = {"observed_at": datetime.datetime.now(datetime.timezone.utc).isoformat(), "deadline": a.deadline,
              "target_training_instances": a.target_train, "accepted_training_instances": train_count,
              "remaining_hours": remaining_hours, "required_instances_per_hour": required,
              "collection_jobs": observed, "forecast_limit": "No all-family throughput claim: only named native lanes are measured; pipeline and acceptance overhead reduce rates."}
    (root / "infra/throughput.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
