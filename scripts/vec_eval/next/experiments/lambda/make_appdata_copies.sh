#!/bin/bash
# Per-process Kit app-data copies for the multi-process measurement (2026-10-31): global/data, local, warp_cache copied (rsync -a),
# the 46 GB texture cache linked file by file (read-only store; a texture a process converts itself is written next to the links).
set -eu
SRC=/home/minyeongk/Desktop/projects/Behavior-1K-vec-speed-6a6194600/OmniGibson/appdata
DST=$HOME/vec_eval85/next/appdata_mp
for k in $(seq 0 $((${1:-8} - 1))); do
  d=$DST/p$k; [ -f $d/.ready ] && continue
  mkdir -p $d/global/cache/texturecache
  rsync -a $SRC/local $d/; rsync -a $SRC/global/data $SRC/global/warp_cache $d/global/
  for x in $SRC/global/cache/*; do [ "$(basename $x)" = texturecache ] || rsync -a $x $d/global/cache/; done
  ( cd $SRC/global/cache/texturecache && find . -type f ) | while read f; do mkdir -p "$d/global/cache/texturecache/$(dirname "$f")"; ln -sf "$SRC/global/cache/texturecache/$f" "$d/global/cache/texturecache/$f"; done
  touch $d/.ready; echo "p$k ready $(du -sh $d | cut -f1)"
done
