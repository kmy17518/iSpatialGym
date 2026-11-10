import json
from pathlib import Path
import subprocess


def test_replay_uses_own_writable_paths_and_preserves_collected_semantics(tmp_path):
    job=tmp_path/'job';job.mkdir()
    root=tmp_path/'campaign';root.mkdir()
    snapshot=tmp_path/'snapshot';env=snapshot/'scripts/trainfast/env_fast.sh';env.parent.mkdir(parents=True)
    env.write_text('JB='+str(root/'private_replay')+'\n')
    (job/'job.env').write_text('\n'.join([
        'ISGF_ROOT='+str(root),'ISGF_EXP_PATH=/foreign/collector/apps',
        'ISGF_JB_NFS=/foreign/collector/jobs','KIT_TPL_NFS_ROOT=/foreign/collector/templates',
        'KIT_TEX_STORE=/foreign/collector/textures','ISGF_SAMPLING_CACHE=/foreign/collector/cache',
        'ISPATIALGYM_TEST_PROPERTY=unchanged'])+'\n')
    source=Path(__file__).parents[1]/'infra_open_loop.sh'
    prefix=source.read_text().split('mkdir -p "$GATE_OUT"')[0]
    script=tmp_path/'probe.sh'
    script.write_text(prefix+'''python3 - <<'PY'
import json,os
print(json.dumps({k:os.environ[k] for k in ('EXP_PATH','ISGF_JB_NFS','KIT_TPL_NFS_ROOT','KIT_TEX_STORE','ISPATIALGYM_SAMPLING_CACHE','ISPATIALGYM_TEST_PROPERTY')}))
PY
''')
    result=subprocess.run(['bash',str(script),str(job),str(snapshot),str(tmp_path/'out')],
        capture_output=True,text=True,check=True)
    values=json.loads(result.stdout)
    assert values['EXP_PATH']==str(root/'isaacsim_overlay/apps')
    assert values['ISGF_JB_NFS']==str(root/'jobs/replay')
    assert values['KIT_TPL_NFS_ROOT']==str(root/'kit_templates')
    assert values['KIT_TEX_STORE']==str(root/'kit_textures')
    assert values['ISPATIALGYM_SAMPLING_CACHE']==str(root/'private_replay/isg_cache/sampling')
    assert values['ISPATIALGYM_TEST_PROPERTY']=='unchanged'
