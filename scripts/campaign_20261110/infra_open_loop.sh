#!/bin/bash
# Campaign-owned true action replay gate. Source demo and TRO are read-only.
set -euo pipefail
JOBDIR=${1:?source job directory}
SNAPSHOT=${2:?frozen snapshot}
GATE_OUT=${3:?gate output directory}
REQUESTED_REPLAY_MEMBER=${4:-}
source "$JOBDIR/job.env"
export ISGF_ROOT
export ISGE_CODE="$SNAPSHOT"
# Replay is submitted by the primary campaign account, including demos collected
# by shiyuc. Its writable caches must belong to the replay account; inheriting
# the collector's .kit overlay caused PermissionError before simulation began.
export KIT_TPL_NFS_ROOT="$ISGF_ROOT/kit_templates"
export KIT_TEX_STORE="$ISGF_ROOT/kit_textures"
export ISGF_JB_NFS="$ISGF_ROOT/jobs/replay"
export ISGF_EXP_PATH="$ISGF_ROOT/isaacsim_overlay/apps"
export EXP_PATH="$ISGF_EXP_PATH"
if [ "${ISGF_REPLAY_BACKEND:-}" = vast ]; then
    # Same image-native app selection and exact-key private warm caches used by
    # validated Vast collection. Cluster account overlays are not transported.
    export KIT_TPL_NFS_ROOT=/vision/u/minyeong/projects/ispatialgym-lab-vec-eval-cache/kit_templates
    export KIT_TEX_STORE=/vision/u/minyeong/projects/ispatialgym-lab-vec-eval-cache/kit_textures
    unset ISGF_EXP_PATH EXP_PATH
fi
source "$SNAPSHOT/scripts/trainfast/env_fast.sh"
set -a
source <(grep -E '^(ISPATIALGYM|ISG)_[A-Z0-9_]*=' "$JOBDIR/job.env")
set +a
unset ISPATIALGYM_HEAD_CAMERA_ONLY ISPATIALGYM_PHYSICS_ONLY
export PYTHONDONTWRITEBYTECODE=1 NUMBA_CACHE_DIR="$JB/numba_cache" ISPATIALGYM_ORACLE_STILL_EXIT=1
export ISPATIALGYM_SAMPLING_CACHE="$JB/isg_cache/sampling"
mkdir -p "$GATE_OUT"
KIT_RC=1
trap 'kit_appdata_teardown' EXIT
mapfile -d '' -t INPUTS < <(python3 - "$JOBDIR" "$REQUESTED_REPLAY_MEMBER" <<'PY'
import json,pathlib,re,sys
root=pathlib.Path(sys.argv[1])
member=sys.argv[2]
if member and not re.fullmatch(r'[A-Za-z0-9_-]+',member):
    raise ValueError('unsafe exact replay member')
for meta in sorted((root/'demo').glob('*/meta.json')):
    if member and meta.parent.name!=member:continue
    if not json.loads(meta.read_text()).get('success'):continue
    h5=meta.parent/'demo.hdf5'
    tros=list((root/'inst').rglob(meta.parent.name+'-tro_state.json'))
    if h5.is_file() and len(tros)==1:
        for p in (tros[0],h5):sys.stdout.write(str(p)+'\0')
        break
PY
)
if [ "${#INPUTS[@]}" -ne 2 ]; then echo 'No unambiguous successful demo + TRO to replay' >&2; exit 2; fi
SCENE_HELPER="${ISGF_REPLAY_SCENE_HELPER:-$ISGF_ROOT/tools/infra_replay_scene.py}"
export ISPATIALGYM_FAST_PARTIAL_SCENE
ISPATIALGYM_FAST_PARTIAL_SCENE=$(python "$SCENE_HELPER" --demo "${INPUTS[1]}" --job "$JOBDIR" --identity "$GATE_OUT/scene_identity.json")
set +e
timeout -s INT -k 120 1800 python -m ispatialgym.oracle.run_isg --instance "${INPUTS[0]}" --mode replay --replay "${INPUTS[1]}" --record hdf5 --no-goal-override --out "$GATE_OUT/replay" > "$GATE_OUT/replay.log" 2>&1
REPLAY_RC=$?
set -e
REPLAY_META="$GATE_OUT/replay/$(basename "$(dirname "${INPUTS[1]}")")/meta.json"
SCENE_RC=1
if [ -f "$REPLAY_META" ]; then
    set +e
    python "$SCENE_HELPER" --identity "$GATE_OUT/scene_identity.json" --replay-report "$REPLAY_META" > "$GATE_OUT/scene_verify.log" 2>&1
    SCENE_RC=$?
    set -e
fi
if [ "$SCENE_RC" -ne 0 ]; then REPLAY_RC=5; fi
GATE_HELPER="${ISGF_REPLAY_GATE_HELPER:-$ISGF_ROOT/tools/replay_gate_quality.py}"
python3 - "$GATE_OUT" "$JOBDIR" "$SNAPSHOT" "${INPUTS[0]}" "${INPUTS[1]}" "$REPLAY_RC" "$GATE_HELPER" <<'PY'
import datetime,hashlib,json,pathlib,sys
import importlib.util
out,job,snapshot,tro,demo,rc,helper=sys.argv[1:]
def sha256(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
results=[]
for line in (pathlib.Path(out)/'replay.log').read_text(errors='replace').splitlines():
    if line.startswith('RUN_ISG_RESULT '):
        try:results.append(json.loads(line.split(' ',1)[1]))
        except ValueError:pass
result=results[-1] if results else {}
passed=int(rc)==0 and result.get('success') is True and result.get('status')=='success'
gate={'schema':1,'validated_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'validation_only':True,'source_job':job,'snapshot':snapshot,'instance':tro,'demo':demo,'demo_sha256':sha256(demo),'tro_sha256':sha256(tro),'replay_meta_paths':[str(p) for p in pathlib.Path(out,'replay').rglob('meta.json')],'replay_rc':int(rc),'replay_mode':'physics_action_open_loop','success':passed,'result':result}
gate['scene_identity']=json.loads((pathlib.Path(out)/'scene_identity.json').read_text())
spec=importlib.util.spec_from_file_location('campaign_replay_quality',helper)
quality=importlib.util.module_from_spec(spec);spec.loader.exec_module(quality)
gate['native_quality']=quality.assess(gate,pathlib.Path(out)/'gate.json')
passed=gate['success']=bool(passed and gate['native_quality']['passed'])
(pathlib.Path(out)/'gate.json').write_text(json.dumps(gate,indent=2)+'\n')
print(json.dumps(gate))
sys.exit(0 if passed else 2)
PY
KIT_RC=0
