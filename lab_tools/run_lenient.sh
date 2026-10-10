#!/bin/bash
# usage: run_lenient.sh <alias> -- waits for the alias's strict pass to end, then writes lenient candidates for the indices it could not reproduce
cd /home/minyeongk/Desktop/projects/ispatialgym-v2.0-vectorized || exit 2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ispatialgym-v2.0-vectorized
export HF_HUB_OFFLINE=1 OMNIGIBSON_HEADLESS=1
alias=$1; out=datasets/eval_1003_regen/$alias
until grep -q "end $alias rc=" "$out/sample.log" 2>/dev/null; do sleep 30; done
echo "=== $(date -Is) start lenient $alias" >> "$out/sample_lenient.log"
python lab_tools/sample_chosen.py --alias "$alias" --manifest datasets/isg-init-eval/isg_meta/isg-init-eval.manifest.json --out "$out" --seed 1003 --max-attempts 8 --lenient-candidates >> "$out/sample_lenient.log" 2>&1
echo "=== $(date -Is) end lenient $alias rc=$?" >> "$out/sample_lenient.log"
