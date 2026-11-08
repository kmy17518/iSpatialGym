#!/bin/bash
# i7-vast: freeze the committed vast_fast tooling on the cluster (run on lambda in the i7-vast worktree):
#   scripts/vast_fast/deploy.sh [commit, default HEAD]  -> $OFH/vast/tool_<sha> (read-only git archive of scripts/vast_fast) and
#   $OFH/vast/tool -> tool_<sha> (the copy the coordinator job and the agents' fake sbatch use). Never an rsync of a working tree.
set -euo pipefail
c=$(git rev-parse --short=8 "${1:-HEAD}")
V=/vision/u/minyeong/projects/isg_home/oracle_fix/vast
H=minyeong@svl11
if ! ssh $H "test -d $V/tool_$c"; then
  ssh $H "mkdir $V/tool_$c.tmp"
  git archive --format=tar "$c" scripts/vast_fast | ssh $H "tar -x -C $V/tool_$c.tmp --strip-components=2 && echo $c > $V/tool_$c.tmp/.commit && chmod -R a-w $V/tool_$c.tmp && mv $V/tool_$c.tmp $V/tool_$c"
fi
ssh $H "ln -sfn tool_$c $V/tool.new && mv -T $V/tool.new $V/tool && readlink $V/tool"
