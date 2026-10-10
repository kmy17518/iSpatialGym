#!/bin/bash
cd /home/minyeongk/Desktop/projects/ispatialgym-v2.0-vectorized || exit 2
until grep -q "end configuration_matching-articulation_open_small_scale-blender_eyedvd-breakfast_table rc=" datasets/eval_1003_regen/configuration_matching-articulation_open_small_scale-blender_eyedvd-breakfast_table/sample.log 2>/dev/null; do sleep 30; done
exec lab_tools/run_sampler.sh configuration_matching-articulation_open_large_scale-dishwasher
