"""Owner policy for campaign_20261110; buckets describe results, never eligibility.

Physical task bounds, native collision/visibility checks, meaningful changes,
heldouts and group distinctness remain independent requirements.
"""
POLICY = {"schema": "isg-bucket-policy-v2", "version": 2,
          "authority": "NEW_SESSION_PROMPT_20261112.md#2", "campaign_id": "campaign_20261110",
          "bucket_ranges_enforced": False, "bucket_quotas_enforced": False,
          "bucket_labels": "descriptive", "physical_constraints_unchanged": True}

# Preparation queries have no collection/acceptance contract. Their versioned
# request environment must opt into the same owner policy explicitly; a code
# snapshot alone does not identify which campaign owns an ordinary native query.
PREPARATION_ENV = "ISPATIALGYM_CAMPAIGN_BUCKET_POLICY"
PREPARATION_SELECTOR = "campaign_20261110:descriptive:v2"
PREPARATION_ACTIVATION = {
    "schema": "isg-bucket-policy-preparation-activation-v1",
    "environment": {PREPARATION_ENV: PREPARATION_SELECTOR},
    "policy": POLICY,
    "acceptance_authorization": False,
}


def active():
    import os
    from ispatialgym.sampling.campaign_runtime import active as configured, configuration
    selected = os.environ.get(PREPARATION_ENV)
    if selected is not None and selected != PREPARATION_SELECTOR:
        raise ValueError("Unknown campaign bucket-policy preparation selector")
    if configured():
        matches = configuration()[0].get("campaign_id") == POLICY["campaign_id"]
        if selected is not None and not matches:
            raise ValueError("Preparation bucket policy conflicts with campaign configuration")
        return matches
    return selected == PREPARATION_SELECTOR
