#!/bin/bash
# i7-vast: the worker bundle (boot.sh at the archive root, worker.py, the git stand-in) of this tool copy -> $JOB_S3/bundle/worker.tar.gz
set -euo pipefail
VTOOL=$(readlink -f ${VTOOL:-/vision/u/minyeong/projects/isg_home/oracle_fix/vast/tool})
source $VTOOL/creds.sh
bash -n $VTOOL/worker/boot.sh
$VPY -c "import ast,sys; ast.parse(open(sys.argv[1]).read())" $VTOOL/worker/worker.py
grep -q 'worker.py' $VTOOL/worker/boot.sh
T=$(mktemp -d); trap 'rm -rf $T' EXIT
tar -C $VTOOL/worker -czf $T/worker.tar.gz boot.sh worker.py git
s5cmd --endpoint-url $S3_ENDPOINT_URL cp $T/worker.tar.gz $JOB_S3/bundle/worker.tar.gz > /dev/null
echo "worker.tar.gz ($(md5sum < $T/worker.tar.gz | cut -c1-8)) from $VTOOL ($(cat $VTOOL/.commit)) -> $JOB_S3/bundle/"
