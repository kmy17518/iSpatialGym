#!/bin/bash
# Runs on every fleet worker right after a code refresh (apply_b1k_overlay.sh calls it when ISG_S3 is set): fixes a running
# worker needs before its next unit without a reboot. Idempotent; never fails the caller.
W=${ISG_WORK:-/work}
# the PaliGemma tokenizer for pi05 servers (openpi would fetch it from storage.googleapis.com, where the venv's TLS check fails)
TOK=$W/pi/openpi_data/big_vision/paligemma_tokenizer.model
if [ ! -s "$TOK" ] && [ -n "${ISG_S3:-}" ] && command -v s5cmd > /dev/null; then
  mkdir -p "$(dirname "$TOK")"
  s5cmd --endpoint-url "$S3_ENDPOINT_URL" cp "$ISG_S3/bundle/openpi_data/big_vision/paligemma_tokenizer.model" "$TOK" > /dev/null 2>&1 && echo "post_refresh: tokenizer seeded"
fi
# campaign rooms (2026-10-28): the val demos (536 x demo.hdf5, 0.8 GB, cluster-absolute paths) the segmentation pass replays;
# the eval bundle ships only the eval demos. Fetched once per instance (marker), extracted at /.
if [ -n "${ISG_S3:-}" ] && [ ! -f "$W/.val_demos_ok" ] && command -v s5cmd > /dev/null; then
  if s5cmd --endpoint-url "$S3_ENDPOINT_URL" cat "$ISG_S3/bundle/val_demos.tar.zst" 2>/dev/null | zstd -d -q | tar -x --no-same-owner -C / 2>/dev/null; then
    touch "$W/.val_demos_ok" && echo "post_refresh: val demos unpacked"
  else
    echo "post_refresh: val demos archive not fetched (not on this prefix?)"
  fi
fi
exit 0
