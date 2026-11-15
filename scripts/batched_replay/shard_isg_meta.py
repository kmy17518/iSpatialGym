#!/usr/bin/env python3
"""Move a merged dataset's isg_meta/goal_images/*.png and isg_meta/tro/*.json into per-family-chunk subfolders (chunk-XXX, the episode's
LeRobot chunk) and rewrite isg_meta/episodes.jsonl + README.md (the Hugging Face Hub takes at most 10,000 files per folder; merges since
2026-11-15 v2 write this layout directly).   shard_isg_meta.py <dataset dir> [...]"""
import json
import os
import sys
from pathlib import Path

for root in map(Path, sys.argv[1:]):
    eps = [json.loads(x) for x in open(root / "isg_meta/episodes.jsonl")]
    moved = 0
    for e in eps:
        c = f"chunk-{int(e['chunk_index']):03d}"
        for k, sub in (("goal_image", "goal_images"), ("tro", "tro")):
            p = e.get(k)
            if not p or f"/{sub}/chunk-" in p:
                continue
            new = f"isg_meta/{sub}/{c}/{os.path.basename(p)}"
            (root / new).parent.mkdir(parents=True, exist_ok=True)
            os.rename(root / p, root / new)
            e[k] = new
            moved += 1
    tmp = root / "isg_meta/episodes.jsonl.tmp"
    tmp.write_text("".join(json.dumps(e) + "\n" for e in eps))
    os.replace(tmp, root / "isg_meta/episodes.jsonl")
    rd = (root / "README.md").read_text()
    rd = rd.replace("- Goal image: isg_meta/goal_images/episode_<i>.png,",
                    "- Goal image: isg_meta/goal_images/chunk-<family chunk>/episode_<i>.png (the path is in isg_meta/episodes.jsonl `goal_image`),")
    rd = rd.replace("isg_meta/tro/, isg_meta/templates/.", "isg_meta/tro/chunk-<family chunk>/,\n  isg_meta/templates/.")
    (root / "README.md").write_text(rd)
    left = [p for p in (root / "isg_meta/goal_images").iterdir() if p.is_file()] + [p for p in (root / "isg_meta/tro").iterdir() if p.is_file()]
    big = {str(d.relative_to(root)): n for d in root.rglob("*") if d.is_dir() for n in [sum(1 for _ in d.iterdir())] if n > 10000}
    print(f"{root.name}: moved {moved} files; flat files left {len(left)}; folders over 10,000 entries {big}")
