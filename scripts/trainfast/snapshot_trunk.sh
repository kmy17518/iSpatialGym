#!/bin/bash
# one trunk (owner 2026-11-05): a FROZEN cluster snapshot of the trunk for every run (validation, collection, evaluation) - never the copy
# being edited. Run on lambda in the trunk worktree:  scripts/trainfast/snapshot_trunk.sh [commit, default HEAD]   -> prints the path.
#   $OFH/code/trunk_<isg commit>/            git archive of the commit (tracked files only: the worktree must hold nothing uncommitted that
#                                            the jobs need), read-only; .snapshot_isg / .snapshot_b1k name the commits
#   $OFH/code/trunk_<c>/Behavior-1K/         one link per entry of a frozen Behavior-1K copy ($OFH/code/b1k_<commit>_p<patch md5>: the
#                                            trunk's cluster Behavior-1K without datasets/, with its uncommitted overlay patches, made once
#                                            and shared by the snapshots of that state; .git links the worktree pointer, so kit_appdata.sh
#                                            reads the commit) + datasets/ of its own: the shared datasets and this snapshot's
#                                            ispatialgym-assets
# An existing snapshot of the commit is reused.
set -euo pipefail
c=$(git rev-parse --short=8 "${1:-HEAD}")
B1K_SRC=/vision/u/minyeong/projects/Behavior-1K-grasp-bands-6a6194600
SHARED=/vision/u/minyeong/projects/ispatialgym/Behavior-1K/datasets
O=/vision/u/minyeong/projects/isg_home/oracle_fix
S=$O/code/trunk_$c
H=minyeong@scdt.stanford.edu
if ssh $H "test -d $S"; then echo $S; exit 0; fi
ssh $H "mkdir -p $O/code && mkdir $S.tmp"   # fails when a half-made $S.tmp exists: look at it, mv it aside
git archive --format=tar "$c" | ssh $H "tar -x -C $S.tmp"
ssh $H bash -s -- "$S" "$B1K_SRC" "$SHARED" "$O" "$c" <<'REMOTE'
set -euo pipefail
S=$1; B1K_SRC=$2; SHARED=$3; O=$4; c=$5
b1k=$(git -C $B1K_SRC rev-parse --short=9 HEAD)
patch=$(cd $B1K_SRC && git status --porcelain --untracked-files=all OmniGibson bddl3 | awk '{print $2}' | sort | xargs md5sum | md5sum | cut -c1-8)
FB=$O/code/b1k_${b1k}_p$patch
if [ ! -d $FB ]; then
  rsync -a --exclude datasets --exclude .git --exclude '__pycache__' $B1K_SRC/ $FB.tmp/
  ln -s $B1K_SRC/.git $FB.tmp/.git
  (cd $B1K_SRC && git status --porcelain --untracked-files=all OmniGibson bddl3) > $FB.tmp/.snapshot_patches
  mv $FB.tmp $FB
  chmod -R a-w $FB
fi
mkdir $S.tmp/Behavior-1K
for e in $(ls -A $FB); do [ "$e" = datasets ] || ln -s $FB/$e $S.tmp/Behavior-1K/$e; done
mkdir $S.tmp/Behavior-1K/datasets
for d in 2026-challenge-task-instances behavior-1k-assets omnigibson-robot-assets omnigibson.key; do ln -s $SHARED/$d $S.tmp/Behavior-1K/datasets/$d; done
ln -s $S/ispatialgym/assets/datasets/ispatialgym-assets $S.tmp/Behavior-1K/datasets/ispatialgym-assets
echo $c > $S.tmp/.snapshot_isg
echo "${b1k}+p$patch" > $S.tmp/.snapshot_b1k
echo "isg=$c b1k=${b1k}+p$patch (frozen $FB) created=$(date -Is)" > $S.tmp/.snapshot.txt
mv $S.tmp $S
chmod -R a-w $S
echo $S
REMOTE
