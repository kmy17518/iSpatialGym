import copy
from types import SimpleNamespace as NS
import pytest
from scripts.campaign_20261110.training_pristine_probe import install_override
from scripts.campaign_20261110.contracts import ContractError,digest


def test_override_keeps_source_bytes_key_and_physical_state_unchanged(tmp_path):
    path=tmp_path/'input.json';path.write_bytes(b'original immutable source')
    campaign={'campaign_id':'campaign_20261110','split':'train','task':'disembedding','class_id':'disembedding_07',
              'runtime':{'purpose':'training_pipeline_canary'},'validation_key':{'code_sha256':'old'},
              'scene_setup':{'restoration_options':{'serialized_root_motion_v1':True}}}
    original=copy.deepcopy(campaign)
    class Loader:
        def __init__(self):
            self.tro_path=str(path);self.report={}
            self.tro=NS(meta=NS(raw={'campaign':copy.deepcopy(campaign)}),object_states={'tray':{'position':[1,2,3]}})
    initialize=Loader.__init__;install_override(NS(InstanceLoader=Loader));a=Loader()
    assert path.read_bytes()==b'original immutable source' and campaign==original
    assert a.tro.meta.raw['campaign']['validation_key']==original['validation_key']
    assert a.tro.object_states=={'tray':{'position':[1,2,3]}}
    assert a.report['physical_diagnostic_override']['source_campaign_sha256']==digest(original)
    assert a.tro.meta.raw['campaign']['scene_setup']['restoration_options']['pristine_mass_before_reset_v1']
    campaign['scene_setup']['restoration_options']['serialized_mass_before_reset_v1']=True
    with pytest.raises(ContractError,match='original fde'):Loader()
    Loader.__init__=initialize
