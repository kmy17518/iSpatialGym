#!/bin/bash
# Re-balance the rb fleet: drop every queued (not started) job on the live pods and re-queue what is still missing
# (no band file on any pod / local root, not running), in priority order: representative slot (wave 1) first, then
# variation_geometry (queue wave 4), then heldout (queue wave 5); brace jobs only on L40S pods (A40 hosts need > 50 min
# per brace job), at the end of their queues (wave 9). The pod command's timeout is raised to 5400 s for the A40 hosts.
#   scripts/reach_bands/redistribute.sh [--no-sync]
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
source ~/miniconda3/etc/profile.d/conda.sh && conda activate ispatialgym
R=runlogs/reach_bands_20260915
PODS=runlogs/runpod/rb/pods.txt
sync_flag=${1:---no-sync}
python scripts/reach_bands/fleet_remaining.py --manifest $R/jobs.json --local runlogs/runpod/rb_local --pulled runlogs/runpod/rb --out $R/jobs_remaining_w1.json 2>&1 | grep -v -i warn
python scripts/reach_bands/fleet_remaining.py --manifest $R/jobs_w2.json --pulled runlogs/runpod/rb --out $R/jobs_remaining_w2.json 2>&1 | grep -v -i warn | head -1
python scripts/reach_bands/fleet_remaining.py --manifest $R/jobs_w3_heldout.json --pulled runlogs/runpod/rb --out $R/jobs_remaining_w3.json 2>&1 | grep -v -i warn | head -1
python3 - "$R" <<'EOF'
import json, sys, os
R = sys.argv[1]
def fix(path, out, brace_out=None):
    jobs = json.load(open(path))
    for j in jobs:
        j["cmd"] = j["cmd"].replace("timeout 3000 ", "timeout 5400 ")
    brace = [j for j in jobs if j["kind"] == "brace"]
    rest = [j for j in jobs if j["kind"] != "brace"]
    json.dump(rest, open(out, "w"), indent=1)
    if brace_out is not None:
        json.dump(brace, open(brace_out, "w"), indent=1)
    print(os.path.basename(out), len(rest), "jobs;", len(brace), "brace jobs")
fix(f"{R}/jobs_remaining_w1.json", f"{R}/jobs_rq_w1.json", f"{R}/jobs_rq_brace.json")
fix(f"{R}/jobs_remaining_w2.json", f"{R}/jobs_rq_w2.json")
fix(f"{R}/jobs_remaining_w3.json", f"{R}/jobs_rq_w3.json")
EOF
python scripts/runpod/dispatch_samp.py --manifest $R/jobs_rq_w1.json --pods-file $PODS --wave 1 --ref survey --reset-queue $sync_flag 2>&1 | grep -v -i warn | tail -1
python scripts/runpod/dispatch_samp.py --manifest $R/jobs_rq_w2.json --pods-file $PODS --wave 4 --ref survey --no-sync 2>&1 | grep -v -i warn | tail -1
python scripts/runpod/dispatch_samp.py --manifest $R/jobs_rq_w3.json --pods-file $PODS --wave 5 --ref survey --no-sync 2>&1 | grep -v -i warn | tail -1
# brace jobs: L40S pods only (names contain l40s), lowest priority
l40s=$(python3 -c "
import json; f=json.load(open('runlogs/runpod/fleet.json'))['pods']
print(' '.join(p for p in open('$PODS').read().split() if 'l40s' in f.get(p,{}).get('name','') and not f.get(p,{}).get('terminated_at')))")
if [ -s $R/jobs_rq_brace.json ] && [ "$(python3 -c "import json; print(len(json.load(open('$R/jobs_rq_brace.json'))))")" != "0" ]; then
  python scripts/runpod/dispatch_samp.py --manifest $R/jobs_rq_brace.json --pods-file $PODS --wave 9 --ref survey --no-sync --only $l40s 2>&1 | grep -v -i warn | tail -1
fi
rm -f $R/assignment_w0*.json
