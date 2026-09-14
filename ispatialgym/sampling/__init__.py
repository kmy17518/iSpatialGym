"""Instance generation for iSpatialGym (see ``docs/instance_sampling_spec.md``).

Entry points:

* ``python -m ispatialgym.sampling.sample_instances`` -- generate templates + TROs for a problem.
* ``python -m ispatialgym.sampling.validate_instances`` -- reload generated instances in a fresh process
  and re-check stability / initial conditions / goal-not-satisfied.

Modules that do not import OmniGibson (``knowledge``, ``spec``, ``bddl_writer``, ``geometry``,
``grasp_library``) can be used and tested without a simulator.
"""

SAMPLER_VERSION = "1.5.0"
