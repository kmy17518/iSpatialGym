#!/bin/bash
# Copies the vec-eval overlay files over a Behavior-1K checkout (6a6194600): scripts/vec_eval/apply_b1k_overlay.sh <Behavior-1K dir>
set -eu
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
B=${1:?Behavior-1K dir}
cp -r "$HERE/b1k_overlay/OmniGibson/." "$B/OmniGibson/"
echo "overlay applied to $B: $(cd "$HERE/b1k_overlay" && find . -type f | tr '\n' ' ')"
# fleet workers: the post-refresh hook (fixes for running workers; no-op elsewhere)
HOOK="$HERE/fleet/worker/post_refresh.sh"
if [ -f "$HOOK" ] && [ -n "${ISG_S3:-}" ]; then bash "$HOOK" || true; fi
