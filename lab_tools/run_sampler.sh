#!/bin/bash
# usage: run_sampler.sh <alias> [<alias> ...]  -- regenerates the chosen eval_1003 instances of each alias in turn (one Isaac Sim process each)
cd /home/minyeongk/Desktop/projects/ispatialgym-v2.0-vectorized || exit 2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ispatialgym-v2.0-vectorized
export HF_HUB_OFFLINE=1 OMNIGIBSON_HEADLESS=1
for alias in "$@"; do
  out=datasets/eval_1003_regen/$alias
  mkdir -p "$out"
  echo "=== $(date -Is) start $alias" >> "$out/sample.log"
  python lab_tools/sample_chosen.py --alias "$alias" --manifest datasets/isg-init-eval/isg_meta/isg-init-eval.manifest.json --out "$out" --seed 1003 --max-attempts 8 >> "$out/sample.log" 2>&1
  echo "=== $(date -Is) end $alias rc=$?" >> "$out/sample.log"
done
