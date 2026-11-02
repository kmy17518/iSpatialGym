#!/bin/bash
# The eval bundle for the fleet (absolute cluster paths, unpacked at / on the instances): the vec worktree's code, the two eval
# datasets, the val instances' template dirs + TRO files, the instance lists, goal images, the ACT checkouts, the msgpack wheel.
# Then units.json and the worker tarball to the run prefix. Run on sc (CPU only).
#   bash scripts/vec_eval/fleet/build_bundle.sh [--no-upload]
# Another campaign (never while the first campaign's fleet runs: the upload reaches its running workers):
#   ISG_RUN_PREFIX=s3://isg-demos/isg-vec-eval-c2 ISG_UNITS=$V/campaigns/c2/units.json ISG_BUNDLE_DIR=$V/campaigns/c2/fleet/bundle bash build_bundle.sh
set -euo pipefail
source /vision/u/minyeong/projects/isg_home/train85/work/vast_replay/bin/creds.sh
V=/vision/u/minyeong/projects/isg_home/vec_eval85
E=${ISG_EVAL_ROOT:-/vision/u/minyeong/projects/ispatialgym-eval-vec}   # the worktree shipped; another worktree (vec_speed) is renamed to this path inside the archives
E0=/vision/u/minyeong/projects/ispatialgym-eval-vec
XF=(); [ "$E" != "$E0" ] && XF=(--transform "s#^${E#/}#${E0#/}#")
P=${ISG_RUN_PREFIX:-s3://$S3_BUCKET/isg-vec-eval}        # the campaign's run prefix (c2: s3://isg-demos/isg-vec-eval-c2)
UNITS=${ISG_UNITS:-$V/fleet/units.json}                    # the units.json shipped in the bundle and uploaded to the prefix
B=${ISG_BUNDLE_DIR:-$V/fleet/bundle}                       # where the archives are built
mkdir -p $B
cd /
python3 - "$V" > $B/filelist.txt <<'EOF'
import glob, json, os, sys
V = sys.argv[1]
out = set()
def add(p):
    out.add(os.path.relpath(p, "/"))
for f in glob.glob(f"{V}/instances/val/*.json"):
    for s in json.load(open(f))["instances"]:
        td = os.path.dirname(os.path.dirname(s["tro"]))
        for name in ("template.json", "template_meta.json", "template-partial_rooms.json"):
            if os.path.exists(os.path.join(td, name)):
                add(os.path.join(td, name))
        add(os.path.join(td, "activity_definitions"))
        add(s["tro"])
print("\n".join(sorted(out)))
EOF
wc -l $B/filelist.txt
EXCL="--exclude=.git --exclude=__pycache__ --exclude=.pytest_cache --exclude=.venv --exclude=outputs --exclude=*.pyc"
tar -cf - $EXCL "${XF[@]}" \
  --exclude=${E#/}/Behavior-1K --exclude=${E#/}/runs \
  ${E#/} \
  vision/u/minyeong/projects/ispatialgym-eval/Behavior-1K/datasets/ispatialgym-eval-instances \
  vision/u/minyeong/projects/ispatialgym-eval/Behavior-1K/datasets/ispatialgym-eval-demos \
  vision/u/minyeong/projects/isg_home/vec_eval85/instances vision/u/minyeong/projects/isg_home/vec_eval85/goals \
  ${UNITS#/} vision/u/minyeong/projects/isg_home/vec_eval85/fleet/wheels \
  vision/u/minyeong/projects/act-todo1 vision/u/minyeong/projects/act-side vision/u/minyeong/projects/act-piaug vision/u/minyeong/projects/openpi-todo1 \
  -T $B/filelist.txt | zstd -q -T8 -3 -o $B/eval.tar.zst -f
ls -la $B/eval.tar.zst
cd $E/scripts/vec_eval/fleet/worker
python3 -c "import ast; ast.parse(open('eval_worker.py').read())"
bash -n boot.sh
tar -czf $B/worker.tar.gz boot.sh eval_worker.py pi_serve.sh
# the code patch running workers fetch before each unit (eval_worker.refresh_code): the worktree's python code only
cd /
tar -cf - $EXCL "${XF[@]}" ${E#/}/ispatialgym ${E#/}/scripts \
  | zstd -q -T4 -3 -o $B/code.tar.zst -f
sha256sum < $B/code.tar.zst | cut -c1-16 > $B/code.sha
cd $E/scripts/vec_eval/fleet/worker
if [ "${1:-}" != "--no-upload" ]; then
  S5="s5cmd --endpoint-url $S3_ENDPOINT_URL"
  $S5 cp $B/eval.tar.zst $P/bundle/eval.tar.zst > /dev/null
  $S5 cp $B/worker.tar.gz $P/bundle/worker.tar.gz > /dev/null
  $S5 cp $B/code.tar.zst $P/bundle/code.tar.zst > /dev/null
  $S5 cp $B/code.sha $P/bundle/code.sha > /dev/null
  $S5 cp $UNITS $P/units.json > /dev/null
  echo "uploaded to $P: code $(cat $B/code.sha), eval.tar.zst ($(md5sum < $B/eval.tar.zst | cut -c1-8)), worker.tar.gz ($(md5sum < $B/worker.tar.gz | cut -c1-8)), units.json"
fi
