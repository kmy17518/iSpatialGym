#!/bin/bash
# Frozen code snapshots for slurm_sweep produce jobs (their job.sh resolves <prod root>/current when the job starts).
#
#   prod_snapshot.sh make <isg commit> <b1k commit>    worktrees <prod root>/code_<isg>/{ispatialgym,Behavior-1K}; the band
#                                                      files get this checkout's mtimes (the band loader breaks ties by mtime)
#   prod_snapshot.sh flip <snapshot dir> "<reason>"     current -> snapshot, one line in <prod root>/flips.log
#                                                      (time, from, to, reason)
#
# A snapshot is made and validated first; current moves once per validated batch of changes.
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
W="$(cd "$HERE/../.." && pwd)"
P="${PROD_ROOT:-$(dirname "$W")/isg_prod}"
case "$1" in
  make)
    C=$(git -C "$W" rev-parse --short=8 "$2")
    B=$(git -C "$W/Behavior-1K" rev-parse --short=9 "$3")
    S="$P/code_$C"
    [ -d "$S/ispatialgym" ] || git -C "$W" worktree add --detach "$S/ispatialgym" "$C" > /dev/null 2>&1
    [ -d "$S/Behavior-1K" ] || git -C "$W/Behavior-1K" worktree add --detach "$S/Behavior-1K" "$B" > /dev/null 2>&1
    for f in "$W"/runlogs/reach_bands_20260915/bands/*; do
      t="$S/ispatialgym/runlogs/reach_bands_20260915/bands/$(basename "$f")"
      [ -e "$t" ] && touch -r "$f" "$t"
    done
    echo "snapshot $S (isg $C, b1k $B); current unchanged -> $(readlink "$P/current")"
    ;;
  flip)
    S="$(readlink -f "$2")"
    [ -d "$S/ispatialgym" ] && [ -d "$S/Behavior-1K" ] || { echo "not a snapshot: $2" >&2; exit 2; }
    [ -n "$3" ] || { echo "flip needs a reason" >&2; exit 2; }
    FROM="$(readlink "$P/current" || echo none)"
    ln -sfn "$S" "$P/current.new" && mv -T "$P/current.new" "$P/current"
    printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "$(basename "$FROM")" "$(basename "$S")" "$3" >> "$P/flips.log"
    echo "current: $(basename "$FROM") -> $(basename "$S")"
    ;;
  *)
    sed -n '2,10p' "$0"; exit 2 ;;
esac
