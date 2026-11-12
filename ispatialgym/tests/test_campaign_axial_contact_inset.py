from types import SimpleNamespace as NS

import pytest
import torch as th

from ispatialgym.sampling import campaign_runtime as CR
from ispatialgym.sampling.v2 import contact_poses as CPZ
from scripts.campaign_20261110.contracts import ContractError


def test_exact_r19_native_contact_counterfactual_preserves_height_orientation_and_approach():
    # Source50306300, r19 native full-program proof, then r20 oracle640steps.
    positions=[[-1.3264256715774536,-2.742126941680908,.8002086877822876],
               [-1.3014256954193115,-2.7421302795410156,.7569074034690857],
               [-1.3583754301071167,-2.7571194171905518,.7569074034690857],
               [-1.404318928718567,-2.7939586639404297,.7569074034690857]]
    expected=[[-1.3259447241358069,-2.7521153694630156,.8002086877822876],
              [-1.3009447479776648,-2.752118707323123,.7569074034690857],
              [-1.3538768011945237,-2.766050393503743,.7569074034690857],
              [-1.3965790846520567,-2.8002907105152826,.7569074034690857]]
    chain=[(th.tensor(p),th.tensor([0.,0.,0.,1.]))for p in positions]
    pivot=th.tensor([-1.2946083731904015,-2.883714005441849,0.])
    assert CPZ.inset_axial_push_chain(chain,pivot,0) is chain
    changed=CPZ.inset_axial_push_chain(chain,pivot,.01)
    for (p,q),(old,oldq),want in zip(changed,chain,expected):
        assert th.allclose(p,th.tensor(want),atol=3e-7,rtol=0)
        assert q is oldq and p[2]==old[2]
    assert th.allclose(changed[0][0]-changed[1][0],chain[0][0]-chain[1][0],atol=2e-7,rtol=0)
    assert th.equal(chain[1][0],th.tensor(positions[1]))
    for distance in [-.01,float('nan'),float('inf'),.2]:
        with pytest.raises(ValueError):CPZ.inset_axial_push_chain(chain,pivot,distance)


def test_inset_only_enabled_for_exact_canonical_magazine(monkeypatch):
    option='axial_magazine_contact_inward_1cm'
    monkeypatch.setattr(CR,'active',lambda:False)
    assert not CR.contact_recovery(option)
    t=NS(class_id='alignment-axial_04',task='alignment-axial',split='eval_standard',
         scene_setup={'recovery_options':{option:True}},
         objects=[NS(role='target',category='magazine',model='tfovwy',default_scale=(1.,)*3)])
    monkeypatch.setattr(CR,'active',lambda:True);monkeypatch.setattr(CR,'configuration',lambda:(None,t,None,None))
    assert CR.contact_recovery(option)
    for obj,key,bad in [(t,'class_id','alignment-axial_05'),(t,'split','eval_model'),
                        (t.objects[0],'model','other'),(t.objects[0],'default_scale',(.9,)*3)]:
        old=getattr(obj,key);setattr(obj,key,bad)
        with pytest.raises(ContractError):CR.contact_recovery(option)
        setattr(obj,key,old)
