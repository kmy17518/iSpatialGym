PREBUILD_ARG=""
if [ "${ISPATIALGYM_FAST:-1}" != 0 ] && [ "${ISPATIALGYM_FAST_COLLECTOR_PREBUILD:-1}" != 0 ]; then
  # source: a template copy an EARLIER train job of this alias wrote (the sampler's output - scene variant and settle as this job's
  # sampler writes them), newest first, this campaign then train85; the counted eval template (the sampler's input) only when none
  # exists - it differs where the train sampler draws another scene variant (disembedding_03)
  SRC=""
  for d in $(ls -td $F/out/*/$ALIAS_KEY/g*/inst/*/$TEMPLATE_ID /vision/u/minyeong/projects/isg_home/train85/out/$ALIAS_KEY/g*/inst/*/$TEMPLATE_ID 2>/dev/null); do
    case "$d" in $JOBDIR/*) continue ;; esac
    if [ -f "$d/template.json" ] && ls $d/instances/*-tro_state.json >/dev/null 2>&1; then SRC=$d; break; fi
  done
  [ -z "$SRC" ] && [ -n "${COUNTED_TEMPLATE:-}" ] && SRC=$COUNTED_TEMPLATE
  CT=""
  [ -n "$SRC" ] && CT=$(ls $SRC/instances/*-tro_state.json 2>/dev/null | head -1)
  if [ -n "$CT" ]; then
    PD=$JOBDIR/prebuild/$(basename $(dirname $SRC))/$(basename $SRC)
    mkdir -p $PD/instances
    rsync -a --exclude instances --exclude runs $SRC/ $PD/
    cp $CT $PD/instances/
    PREBUILD_ARG="--prebuild $PD/instances/$(basename $CT)"
    echo "PREBUILD source=$SRC $PREBUILD_ARG" | tee -a $JOBDIR/job.txt
  fi
fi
