from types import SimpleNamespace
import json

import numpy as np
import pytest

from scripts.campaign_20261110.training_attachment_release import positive_lattice
from scripts.campaign_20261110.materialize_training_attachment import materialize


def test_native_lattice_rejects_empty_unreachable_unknown_and_nonfinite_maps():
    states={'unreachable':0,'feasible':1}
    native=SimpleNamespace(cells=np.zeros((2,2)),yaws=np.zeros(2),arrays={'s0':np.array([0,1])})
    assert positive_lattice(native,states)==1
    for bad in (np.array([0,0]),np.array([1,3])):
        native.arrays={'s0':bad}
        with pytest.raises(ValueError):positive_lattice(native,states)
    native.arrays={'s0':np.array([1,1])};native.cells[0,0]=np.nan
    with pytest.raises(ValueError,match='nonfinite'):positive_lattice(native,states)


def test_tabletop_only_release_cannot_materialize_attachment(tmp_path):
    (tmp_path/'training_height_reader_audit.json').write_text(json.dumps({
        'schema':'isg-training-height-release-v1','native_reader_errors':0}))
    with pytest.raises(ValueError,match='dedicated attachment'):
        materialize(tmp_path,tmp_path,tmp_path,tmp_path/'out',{'configuration_matching-attachment_04'},tmp_path/'jobs.json')
    assert not (tmp_path/'out').exists()
