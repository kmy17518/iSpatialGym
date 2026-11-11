#!/bin/bash
# The image has no Python before the normal environment bundle is initialized.
set -euo pipefail
policy=${1:?}; instance=${2:?}
[[ $instance =~ ^[0-9]+$ ]]
read -r schema campaign < "$policy"
[[ $schema == isg-private-worker-boot-policy-v1 && $campaign == campaign_20261110 ]]
mode=normal
while read -r iid action extra; do
  [[ -z ${iid:-} ]] && continue
  [[ $iid =~ ^[0-9]+$ && $action == recover && -z ${extra:-} ]]
  if [[ $iid == "$instance" ]]; then
    [[ $mode == normal ]]
    mode=recover
  fi
done < <(tail -n +2 "$policy")
printf '%s\n' "$mode"
