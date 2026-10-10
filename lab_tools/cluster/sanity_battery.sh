#!/usr/bin/env bash
# Sanity battery of the lab evaluator for the cluster worktree (lab_tools/cluster/MERGE_SANITY_PROMPT.md). Run it once per
# code version on the same GPU type - label base on the unmerged lab-vec-eval, label merged on the merge - then compare:
#   python lab_tools/cluster/compare_replays.py --a sanity/base/records --b sanity/merged/records --strict
#
#   lab_tools/cluster/sanity_battery.sh <label> [static] [replay] [vec] [closedloop]      (no part named: all four)
#
# Parts:
#   static      compileall, check_eval_surface.py (--base $BASE_REF), the no-simulator tests (test_vec_stream: the vectorized
#               executor against the single-env Executor on fake environments)
#   replay      single-env open-loop replays of every task's recorded demos (replay_episodes.py), in the episode order of
#               lab_tools/cluster/expected_replays.json; the train split only when datasets/isg-init is there
#   vec         vectorized replays (vec_eval.py --mode replay): N=1 on each TRO kind, N=4 on relocalization; gate 3b against
#               this label's single-env replays (N=1 bit-identical, N=4 counters + trajectories within 5 mm / 5 mrad) and
#               gate 3d (every slot's first frame vs the dataset's)
#   closedloop  one checkpoint on 5 relocalization episodes through the single-env path and the N=4 path; this script starts
#               its own ACT server on $ACT_PORT and stops it (plumbing check: outcomes are chaotic, see compare_replays.py)
# Outputs: sanity/<label>/{records,raw,closedloop,logs}/, sanity/<label>/steps.jsonl (per step: exit code, minutes,
# command), gate summaries sanity/<label>/gate*.json. One Isaac Sim process at a time (~10-15 GB of GPU memory, ~10 GB RAM)
# plus the ACT server (~1 GB) during closedloop; about 2-3 h for all four parts on an RTX 4090-class GPU. Run from anywhere
# on a GPU node with the worktree's conda env active. DRY_RUN=1 prints the commands without running them.
#
# Variables: BASE_REF (default origin/lab-vec-eval), ACT_PORT (default 8010), CKPT_RUN / CKPT_STEP / CKPT_REPO / CKPT_FILE /
# CKPT_SHA256 (default r1-early-copy05-st0-s0 @100k, expected under act-isg/ckpts/<run>/<file>), CLOSEDLOOP_SPLIT
# (train when datasets/isg-init is there, else eval50).
set -uo pipefail

L="${1:?usage: sanity_battery.sh <label> [static] [replay] [vec] [closedloop]}"
shift
PARTS=" ${*:-static replay vec closedloop} "
WT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$WT" || exit 2
S="sanity/$L"
REC="$S/records"
RAW="$S/raw"
LOG="$S/logs"
mkdir -p "$REC" "$RAW" "$LOG" "$S/closedloop"
BASE_REF="${BASE_REF:-origin/lab-vec-eval}"
ACT_PORT="${ACT_PORT:-8010}"
CKPT_RUN="${CKPT_RUN:-r1-early-copy05-st0-s0}"
CKPT_STEP="${CKPT_STEP:-100000}"
CKPT_REPO="${CKPT_REPO:-kmy17518/act-isg-r1-early-copy05-st0-s0}"
CKPT_FILE="${CKPT_FILE:-eval/step-00100000.pt}"
CKPT_SHA256="${CKPT_SHA256:-629210205d12fbe984a43415c2b705de8d87a402be711a82628ca74bf9be950f}"
RELOC=camera_relocalization-standard
ALIGN=alignment-axial-board_game-breakfast_table
DISH=configuration_matching-articulation_open_large_scale-dishwasher
BLEND=configuration_matching-articulation_open_small_scale-blender_eyedvd-breakfast_table
BANANA=mental_rotation-banana
CAN=object_scaling-canned_food
HAVE_TRAIN=0
[ -f datasets/isg-init/meta/info.json ] && HAVE_TRAIN=1

run() {  # run <step name> <command...>: output to $LOG/<step>.log, one json line per step in $S/steps.jsonl
    local name="$1" cmd t0 rc
    shift
    cmd=$(printf '%q ' "$@")
    echo "[$(date -Is)] $name: $cmd"
    [ "${DRY_RUN:-0}" = 1 ] && return 0
    t0=$(date +%s)
    "$@" > "$LOG/$name.log" 2>&1
    rc=$?
    python - "$name" "$rc" "$t0" "$cmd" >> "$S/steps.jsonl" <<'EOF'
import json, sys, time
print(json.dumps({"step": sys.argv[1], "exit": int(sys.argv[2]), "minutes": round((time.time() - int(sys.argv[3])) / 60, 1), "cmd": sys.argv[4]}))
EOF
    echo "    -> exit $rc ($LOG/$name.log)"
}

replay() {  # replay <split> <task> <episodes...>
    local split="$1" task="$2" ds=datasets/isg-init-eval inst=lab/instances.json
    shift 2
    if [ "$split" = train ]; then ds=datasets/isg-init; inst=lab/instances-train.json; fi
    run "replay-$split-$task" python lab_tools/replay_episodes.py --split "$split" --dataset "$ds" --instances "$inst" --task "$task" \
        --episodes "$@" --horizon 1500 --no-planner --tag "sanity-$L" --lab-out "$REC/replay-$split/$task" --raw-out "$RAW/replay-$split/$task"
}

vec() {  # vec <run name> <task> <num_envs> <episodes...>
    local name="$1" task="$2" n="$3"
    shift 3
    run "vec-$name" python lab_tools/vec_eval.py --mode replay --split eval50 --dataset datasets/isg-init-eval --instances lab/instances.json \
        --task "$task" --episodes "$@" --num-envs "$n" --tag "sanity-$L-vec-$name" --lab-out "$REC/vec/$name" --raw-out "$RAW/vec/$name"
}

{
    echo "label=$L host=$(hostname -s) gpu=\"$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null | head -1)\""
    echo "commit=$(git rev-parse HEAD) branch=$(git branch --show-current) dirty=$(git status --porcelain --untracked-files=no | wc -l)"
    echo "b1k=$(git -C Behavior-1K rev-parse HEAD 2>/dev/null) act=$(git -C act-isg rev-parse HEAD 2>/dev/null) python=$(command -v python)"
    echo "started=$(date -Is) parts=$PARTS"
} | tee "$S/battery_env.txt"

if [[ "$PARTS" == *" static "* ]]; then
    run compileall python -m compileall -q ispatialgym lab_tools
    run eval-surface python lab_tools/cluster/check_eval_surface.py --base "$BASE_REF" --out "$S/eval_surface.json"
    run unit-tests python -m pytest -q -p no:cacheprovider ispatialgym/tests/test_vec_stream.py ispatialgym/tests/test_problem_parsing.py \
        ispatialgym/tests/test_geometry_and_targets.py
fi

if [[ "$PARTS" == *" replay "* ]]; then
    replay eval50 "$RELOC" 2 3 33 34 45
    replay eval50 "$ALIGN" 150 151 152 153
    replay eval50 "$DISH" 50 51 52
    replay eval50 "$BLEND" 100 101 102 103
    replay eval50 "$BANANA" 200 201 202
    replay eval50 "$CAN" 250 251 252
    if [ "$HAVE_TRAIN" = 1 ]; then
        replay train "$RELOC" 39 59 74 82 98 99 107 113 116
    else
        echo "datasets/isg-init missing: the train-split replays are skipped"
    fi
fi

if [[ "$PARTS" == *" vec "* ]]; then
    vec reloc-n1 "$RELOC" 1 2
    vec reloc-n4 "$RELOC" 4 2 3 33 34
    vec align-n1 "$ALIGN" 1 150
    vec dish-n1 "$DISH" 1 50
    vec blend-n1 "$BLEND" 1 100
    vec banana-n1 "$BANANA" 1 200
    vec can-n1 "$CAN" 1 250
    run gate3b-reloc python lab_tools/vec_gate_3b.py --root "$REC/vec" --runs reloc-n1 reloc-n4 --ref "$REC/replay-eval50/$RELOC" --out "$S/gate3b_reloc.json"
    for t in align:"$ALIGN" dish:"$DISH" blend:"$BLEND" banana:"$BANANA" can:"$CAN"; do
        run "gate3b-${t%%:*}" python lab_tools/vec_gate_3b.py --root "$REC/vec" --runs "${t%%:*}-n1" --ref "$REC/replay-eval50/${t#*:}" --out "$S/gate3b_${t%%:*}.json"
    done
    run gate3d-reloc python lab_tools/vec_gate_3d.py --root "$REC/vec" --runs reloc-n4 --ref "$REC/replay-eval50/$RELOC" --dataset datasets/isg-init-eval \
        --out "$S/gate3d_reloc.json"
fi

if [[ "$PARTS" == *" closedloop "* ]]; then
    CKPT="act-isg/ckpts/$CKPT_RUN/$CKPT_FILE"
    if [ ! -f "$CKPT" ] && [ "${DRY_RUN:-0}" != 1 ]; then
        echo "checkpoint $CKPT missing (SETUP_PROMPT.md, checkpoint step): closedloop skipped"
    elif [ "${DRY_RUN:-0}" != 1 ] && [ "$(sha256sum "$CKPT" | cut -d' ' -f1)" != "$CKPT_SHA256" ]; then
        echo "checkpoint $CKPT: sha256 differs from $CKPT_SHA256: closedloop skipped"
    else
        SPLIT=eval50 DS=datasets/isg-init-eval INST=lab/instances.json EPS="2 3 33 34 45"
        if [ "${CLOSEDLOOP_SPLIT:-$([ "$HAVE_TRAIN" = 1 ] && echo train || echo eval50)}" = train ]; then
            SPLIT=train DS=datasets/isg-init INST=lab/instances-train.json EPS="101 105 111 113 121"
        fi
        SARGS="serve_b1k.py --model-path ckpts/$CKPT_RUN/$CKPT_FILE --port $ACT_PORT --device cuda"
        SERVER_PID=""
        if [ "${DRY_RUN:-0}" != 1 ]; then
            (cd act-isg && exec .venv/bin/python scripts/b1k/serve_b1k.py --model-path "ckpts/$CKPT_RUN/$CKPT_FILE" --port "$ACT_PORT" --device cuda) \
                > "$LOG/act_server.log" 2>&1 &
            SERVER_PID=$!
            trap '[ -n "$SERVER_PID" ] && kill "$SERVER_PID" 2>/dev/null' EXIT
            for _ in $(seq 1 150); do curl -sf "http://127.0.0.1:$ACT_PORT/healthz" > /dev/null && break; sleep 2; done
            curl -sf "http://127.0.0.1:$ACT_PORT/healthz" > /dev/null || echo "ACT server on :$ACT_PORT not healthy after 300 s (see $LOG/act_server.log)"
        fi
        CK=(--ckpt-repo "$CKPT_REPO" --ckpt-file "$CKPT_FILE" --ckpt-sha256 "$CKPT_SHA256" --ckpt-path "$CKPT" --server "ws://127.0.0.1:$ACT_PORT"
            --server-args "$SARGS" --action-horizon 16)
        # shellcheck disable=SC2086
        run closedloop-single python lab_tools/eval_episodes.py --run "$CKPT_RUN" --step "$CKPT_STEP" "${CK[@]}" --task "$RELOC" --split "$SPLIT" \
            --dataset "$DS" --instances "$INST" --episodes $EPS --horizon 1500 --no-planner --cleanup --tag "sanity-$L" \
            --lab-out "$S/closedloop/single" --raw-out "$RAW/closedloop/single"
        # shellcheck disable=SC2086
        run closedloop-vec-n4 python lab_tools/vec_eval.py --mode policy --run "$CKPT_RUN" --step "$CKPT_STEP" "${CK[@]}" --task "$RELOC" \
            --split "$SPLIT" --dataset "$DS" --instances "$INST" --episodes $EPS --num-envs 4 --horizon 1500 --tag "sanity-$L-vec" \
            --lab-out "$S/closedloop/vec-n4" --raw-out "$RAW/closedloop/vec-n4"
        if [ -n "$SERVER_PID" ]; then
            kill "$SERVER_PID" 2>/dev/null
            wait "$SERVER_PID" 2>/dev/null
            SERVER_PID=""
        fi
        run closedloop-summary python lab_tools/cluster/compare_replays.py --summarize "$S/closedloop" --out "$S/closedloop_summary.json"
    fi
fi

echo "finished=$(date -Is)" | tee -a "$S/battery_env.txt"
[ "${DRY_RUN:-0}" = 1 ] || python - "$S/steps.jsonl" <<'EOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1])]
for r in rows:
    print(f"{'ok  ' if r['exit'] == 0 else 'FAIL'} {r['step']:<40} exit {r['exit']:>3}  {r['minutes']:6.1f} min")
EOF
