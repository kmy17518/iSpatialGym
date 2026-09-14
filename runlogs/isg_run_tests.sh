#!/bin/bash
# Run OmniGibson non-primitives tests one file per process (matches CI isolation).
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate ispatialgym
cd /home/minyeongk/Desktop/projects/ispatialgym/Behavior-1K/OmniGibson || exit 1
export OMNIGIBSON_HEADLESS=1

LOGDIR=/tmp/isg_testlogs
mkdir -p "$LOGDIR"
SUMMARY="$LOGDIR/summary.txt"
: > "$SUMMARY"

# Tier 1 = CI non-primitives set; Tier 2 = extra non-primitives not in CI matrix.
FILES=(
  test_systems
  test_object_removal
  test_scene_graph
  test_sensors
  test_dump_load_states
  test_robot_teleoperation
  test_controllers
  test_robot_states_flatcache
  test_symbolic_primitives
  test_multiple_envs
  test_object_states
  test_transition_rules
  test_snapshots
  test_ag_states
  test_behavior_task
  test_data_collection
)

for f in "${FILES[@]}"; do
  echo "=== RUN $f ($(date +%T)) ===" | tee -a "$SUMMARY"
  timeout 2700 pytest -q "tests/$f.py" > "$LOGDIR/$f.log" 2>&1
  code=$?
  res=$(grep -E "[0-9]+ (passed|failed|error|skipped)" "$LOGDIR/$f.log" | tail -1)
  if [ "$code" -eq 124 ]; then res="TIMEOUT (>2700s); $res"; fi
  echo "RESULT $f : exit=$code : $res" | tee -a "$SUMMARY"
done
echo "ALL_DONE ($(date +%T))" | tee -a "$SUMMARY"
