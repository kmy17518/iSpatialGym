import unittest

from scripts.campaign_20261110.infra_vast.funding import remaining_credit


class FundingBudgetTests(unittest.TestCase):
    def test_existing_lease_and_spend_since_credit_observation_are_both_reserved(self):
        state={'accounts':{'main':{'credit_usd':100, 'observed_at_unix':1000}}}
        events=[{'ev':'create','instance':1,'account':'main','time':900,'dph_total':36,'max_lifetime_s':1000}]
        # $3 charged since the observation, $6 remains committed at t=1300.
        self.assertEqual(remaining_credit(state,'main',events,1300,reserve=10),81)
        # Future committed cost shrinks as incurred cost grows; no fictitious new credit.
        self.assertEqual(remaining_credit(state,'main',events,1400,reserve=10),81)

    def test_destroyed_lease_cost_and_other_accounts_do_not_pollute_commitment(self):
        state={'accounts':{'main':{'credit_usd':100, 'observed_at_unix':1000}}}
        events=[{'ev':'create','instance':1,'account':'main','time':900,'dph_total':36,'max_lifetime_s':1000},
                {'ev':'destroy','instance':1,'time':1200,'ok':True},
                {'ev':'create','instance':2,'account':'other','time':1000,'dph_total':3600,'max_lifetime_s':1000}]
        self.assertEqual(remaining_credit(state,'main',events,1300,reserve=10),88)

    def test_failed_destroy_retains_commitment_and_stale_or_missing_credit_fails_closed(self):
        state={'accounts':{'main':{'credit_usd':100, 'observed_at_unix':1000}}}
        events=[{'ev':'create','instance':1,'account':'main','time':900,'dph_total':36,'max_lifetime_s':1000},
                {'ev':'destroy','instance':1,'time':1200,'ok':False}]
        self.assertEqual(remaining_credit(state,'main',events,1300,reserve=10),81)
        self.assertEqual(remaining_credit(state,'main',events,1601),0)
        self.assertEqual(remaining_credit(state,'unknown',events,1300),0)


if __name__ == '__main__':
    unittest.main()
