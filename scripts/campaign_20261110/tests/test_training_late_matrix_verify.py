import pytest
from scripts.campaign_20261110.contracts import ContractError
from scripts.campaign_20261110.training_sequential_augmentation import (
    effective_probe_identity, additional_reconstruction_reference, PHYSICS)


def test_only_control_labels_and_descriptive_scope_are_excluded_from_effective_identity():
    first={'scope':'First independent cold reference','diagnostic_control':'FRESH_A','runtime_compatibility':{'sha256':'same'},'canaries':['a','b'],'late_incoming_physics_rebuild_v1':True}
    second={**first,'scope':'Warm recovery after independently verified cold reference','diagnostic_control':'A_FAILURE_A'}
    assert effective_probe_identity(first)==effective_probe_identity(second)
    for key,value in [('canaries',['other','b']),('runtime_compatibility',{'sha256':'other'}),('late_incoming_physics_rebuild_v1',False)]:
        assert effective_probe_identity(first)!=effective_probe_identity({**second,key:value})
    with pytest.raises(ContractError):effective_probe_identity({**first,'scope':{'command':'unreviewed'}})


def test_late_reconstruction_requires_exact_complete_audit_matrix_and_declared_request():
    refs={name:{'path':name,'sha256':'exact'}for name in PHYSICS};doc={'additional_reconstructions':refs};request={'fixed_mass_baseline':True,'late_incoming_physics_rebuild_v1':True}
    assert additional_reconstruction_reference(doc,request,'ABA',PHYSICS)==refs['ABA']
    with pytest.raises(ContractError):additional_reconstruction_reference({'additional_reconstructions':{'ABA':refs['ABA']}},request,'ABA',PHYSICS)
    with pytest.raises(ContractError):additional_reconstruction_reference(doc,{'fixed_mass_baseline':True},'ABA',PHYSICS)
    with pytest.raises(ContractError):additional_reconstruction_reference(doc,{'late_incoming_physics_rebuild_v1':True},'ABA',PHYSICS)
    assert additional_reconstruction_reference({}, {'fixed_mass_baseline':True}, 'ABA',PHYSICS)is None
