import copy
import unittest
from scripts.campaign_20261110.carton_lip_probe import ASSET_SHA256,contact_variants
from scripts.campaign_20261110.contracts import ContractError


class ContactProbeTests(unittest.TestCase):
    def audit(self):
        return {'schema':'isg-carton-lip-contact-audit-v1','asset_sha256':ASSET_SHA256,
                'scale':[1,1,1],'joint_initial_rad':0.,'results':[
                    {'arm':'right','yaw_deg':0,'pitch_deg':p,'side_y_m':-.07,
                     'geometry_contact_candidate':True,'conservative_body_plane_clearance_m':.001,
                     'first_lid_contact':{'role':'finger','lift_m':.04},
                     'origin_closed_fingers':[.12,-.07,.03],
                     'rotation_eef_to_object':[[0,0,-1],[0,-1,0],[-1,0,0]]}for p in [0,10]]}

    def test_contact_preserves_closed_geometry_and_exact_two_trial_bound(self):
        d=self.audit();before=copy.deepcopy(d);v=contact_variants(d,'right')
        self.assertEqual(d,before);self.assertEqual(len(v),2)
        self.assertAlmostEqual(v[0]['handle']['contact'][2],.07)
        self.assertAlmostEqual(v[0]['handle']['contact'][0],.108)
        self.assertEqual(v[0]['preshape_q'],0.)

    def test_rejects_ajar_wrong_asset_and_nonfinger_contacts(self):
        for change in ['ajar','asset','palm','clearance','duplicate']:
            with self.subTest(change=change):
                d=self.audit()
                if change=='ajar':d['joint_initial_rad']=.1
                if change=='asset':d['asset_sha256']='0'*64
                if change=='palm':d['results'][0]['first_lid_contact']['role']='palm'
                if change=='clearance':d['results'][0]['conservative_body_plane_clearance_m']=0.
                if change=='duplicate':d['results'].append(copy.deepcopy(d['results'][0]))
                with self.assertRaises(ContractError):contact_variants(d,'right')

    def exact_audit(self):
        d=self.audit();d['schema']='isg-carton-lip-exact-clearance-audit-v1'
        for r,(pitch,y,inset) in zip(d['results'],[(10,0.,.004),(0,.07,.006)]):
            r.update(pitch_deg=pitch,side_y_m=y,inset_from_body_plane_m=inset,
                     body_clearance_m=.001,first_inflated_body_contact=None)
        return d

    def test_exact_piece_retry_uses_only_changed_clearance_checked_contacts(self):
        d=self.exact_audit();before=copy.deepcopy(d);v=contact_variants(d,'right')
        self.assertEqual(d,before);self.assertEqual(len(v),2)
        self.assertEqual([r['geometry_proposal']['inset_from_body_plane_m']for r in v],[.004,.006])
        for change in ['intersection','clearance','old_pose']:
            with self.subTest(change=change):
                d=self.exact_audit()
                if change=='intersection':d['results'][0]['first_inflated_body_contact']={'lift_m':.01}
                if change=='clearance':d['results'][0]['body_clearance_m']=0.
                if change=='old_pose':d['results'][0]['inset_from_body_plane_m']=0.
                with self.assertRaises(ContractError):contact_variants(d,'right')


if __name__=='__main__':unittest.main()
