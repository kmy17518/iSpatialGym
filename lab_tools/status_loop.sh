#!/bin/bash
# every 15 minutes: pull m1/ + docs/, rewrite lab/status.json, upload lab/ (one commit). Log: lab_tools/status_loop.log
cd /home/minyeongk/Desktop/projects/ispatialgym-v2.0-vectorized || exit 2
while true; do
  echo "=== $(date -u +%Y-%m-%dT%H:%M:%SZ) sync" >> lab_tools/status_loop.log
  act-isg/.venv/bin/python lab_tools/lab_sync.py sync >> lab_tools/status_loop.log 2>&1
  sleep 900
done
