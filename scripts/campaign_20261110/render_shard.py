"""Prepare and reconcile one accepted-manifest kinematic rendering shard."""
import argparse
import json
from pathlib import Path
import sys

from ispatialgym.multienv.campaign_rendering import selected_records
from scripts.campaign_20261110.acceptance import read
from scripts.campaign_20261110.contracts import file_digest, require, write_json_atomic


def prepare(root, release, shard_id, snapshot, output, limit=None):
    root, release, snapshot, output = [Path(p).resolve() for p in (root, release, snapshot, output)]
    require(release.is_relative_to(root / "dataset/releases"), "release outside campaign dataset")
    require(snapshot.is_relative_to(root / "code") and (snapshot / ".snapshot_commit").is_file(), "unowned rendering snapshot")
    require(output.is_relative_to(root / "rendering") and output != root / "rendering", "rendering output outside campaign")
    summary = read(release / "release.json")
    for name, expected in summary["files"].items():
        require(Path(name).name == name and file_digest(release / name) == expected, "release manifest changed")
    matches = [s for s in read(release / "shards.json")["shards"] if s["id"] == shard_id]
    require(len(matches) == 1, "unknown or duplicate render shard")
    shard = matches[0]
    listing = Path(shard["list"]).resolve()
    require(listing.is_relative_to(release) and file_digest(listing) == shard["list_sha256"], "render shard input changed")
    demos = listing.read_text().splitlines()
    if limit is not None:
        require(limit > 0, "invalid canary episode limit")
        demos = demos[:limit]
    records = selected_records(release / "instances.jsonl", demos)
    require(all(row["variant"] == shard["variant"] for row in records.values()), "shard mixes variants")
    source_job = Path(demos[0]).parent.parent
    require(source_job.is_relative_to(root / "out") and (source_job / "job.env").is_file(), "missing owned rendering environment")
    output.mkdir(parents=True, exist_ok=False)
    (output / "demos.txt").write_text("".join(path + "\n" for path in demos))
    plan = {"schema": "isg-render-launch-v1", "purpose": "kinematic_observation_rendering", "release": str(release),
            "release_sha256": file_digest(release / "release.json"), "snapshot": str(snapshot),
            "snapshot_sha256": file_digest(snapshot / ".snapshot_commit"), "shard": shard,
            "instances": [row["instance_id"] for row in records.values()], "source_job": str(source_job),
            "output": str(output), "canary_limit": limit, "scene_mode": "recorded"}
    write_json_atomic(output / "launch.json", plan)
    return plan


def reconcile(output, returncode):
    output = Path(output)
    launch = read(output / "launch.json")
    results = [json.loads(line) for line in (output / "results.jsonl").read_text().splitlines()] if (output / "results.jsonl").exists() else []
    episodes = [r for r in results if r.get("tag") == "REPLAY_DEMO"]
    expected = set(launch["instances"])
    seen = [Path(row["demo"]).name for row in episodes]
    passed = (returncode == 0 and len(seen) == len(set(seen)) and set(seen) == expected
              and any(r.get("tag") == "REPLAY_SUMMARY" for r in results)
              and all(r.get("kept") is True and r.get("episode_index") is not None
                      and (r.get("campaign_restoration") or {}).get("scale_verified") is True for r in episodes))
    result = {"schema": "isg-render-result-v1", "purpose": "kinematic_observation_rendering", "success": passed,
              "returncode": returncode, "expected_instances": sorted(expected), "rendered_instances": seen,
              "kept_instances": [Path(r["demo"]).name for r in episodes if r.get("kept") is True],
              "episodes": episodes, "launch_sha256": file_digest(output / "launch.json"),
              "physics_action_validation": "separate per-template gate in accepted manifest"}
    write_json_atomic(output / "result.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("root", "release", "shard", "snapshot"):
        p.add_argument("--" + name)
    p.add_argument("--output", required=True)
    p.add_argument("--limit", type=int)
    p.add_argument("--returncode", type=int)
    a = p.parse_args()
    if a.returncode is not None:
        result = reconcile(a.output, a.returncode)
        print(json.dumps({"success": result["success"], "rendered": len(result["rendered_instances"])}))
        raise SystemExit(0 if result["success"] else 1)
    plan = prepare(a.root, a.release, a.shard, a.snapshot, a.output, a.limit)
    for value in (plan["source_job"], plan["shard"]["variant"]):
        sys.stdout.write(value + "\0")


if __name__ == "__main__":
    main()
