#!/bin/bash
# Per-process GPU memory (nvidia-smi --query-compute-apps) and host RSS of the lab's evaluator / replay / server processes,
# sampled every INTERVAL seconds into lab/diagnostics/eval-fixes/memlog.jsonl (one JSON line per sample). Started
# 2026-10-08 before the chunk-jerk pause as the production baseline for Part 2 (memory growth per episode).
#   nohup lab_tools/mem_monitor.sh [interval_s] > /dev/null 2>&1 &     pid -> lab_tools/mem_monitor.pid
cd /home/minyeongk/Desktop/projects/ispatialgym-v2.0-vectorized || exit 2
INTERVAL=${1:-20}
OUT=lab/diagnostics/eval-fixes/memlog.jsonl
mkdir -p "$(dirname "$OUT")"
echo $$ > lab_tools/mem_monitor.pid
while true; do
  ts=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  total=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  procs=""
  while IFS=, read -r pid mem; do
    pid=$(echo "$pid" | tr -d ' '); mem=$(echo "$mem" | tr -d ' ')
    [ -z "$pid" ] && continue
    cmd=$(tr '\0' ' ' < /proc/$pid/cmdline 2>/dev/null | cut -c1-400)
    kind=other
    case "$cmd" in
      *eval_episodes.py*) kind=eval ;;
      *replay_episodes.py*) kind=replay ;;
      *chunk_jerk*) kind=diag ;;
      *eval_worker.py*) kind=worker ;;
      *serve_b1k.py*) kind=server ;;
      *first_frames.py*|*reloc_recover.py*) kind=instances ;;
    esac
    rss=$(awk '/VmRSS/{print int($2/1024)}' /proc/$pid/status 2>/dev/null)
    run=$(echo "$cmd" | grep -o -- '--run [^ ]*' | cut -d' ' -f2)
    procs="$procs{\"pid\":$pid,\"kind\":\"$kind\",\"gpu_mib\":$mem,\"rss_mib\":${rss:-null},\"run\":\"${run}\"},"
  done < <(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits)
  echo "{\"ts\":\"$ts\",\"gpu_used_mib\":${total:-null},\"procs\":[${procs%,}]}" >> "$OUT"
  sleep "$INTERVAL"
done
