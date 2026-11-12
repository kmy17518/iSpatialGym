"""Owner policy for campaign_20261110; buckets describe results, never eligibility.

Physical task bounds, native collision/visibility checks, meaningful changes,
heldouts and group distinctness remain independent requirements.
"""
POLICY = {"schema": "isg-bucket-policy-v2", "version": 2,
          "authority": "NEW_SESSION_PROMPT_20261112.md#2", "campaign_id": "campaign_20261110",
          "bucket_ranges_enforced": False, "bucket_quotas_enforced": False,
          "bucket_labels": "descriptive", "physical_constraints_unchanged": True}


def active():
    from ispatialgym.sampling.campaign_runtime import active as configured, configuration
    return configured() and configuration()[0].get("campaign_id") == POLICY["campaign_id"]
