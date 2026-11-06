"""Grasp library v2 and its shared contact policy (2026-11-02, branch grasp-bands).

* :mod:`hand_geometry`  - the R1Pro hand's PhysX collision hulls (fingers, palm, wrist camera, wrist) in the eef frame
* :mod:`contact_policy` - the versioned finger/support allowance and the non-finger rejection shared by candidate generation,
  physics validation and reach-map IK
* :mod:`schema`         - versioned records: rest configurations, symmetry families, validated grasps, physics profiles
* :mod:`symmetry`       - rotational / translational grasp families and their deduplication
* :mod:`candidates`     - tabletop / attachment candidate generation (top-down and side grasps)
* :mod:`validate_tabletop` - supported physics validation in a persistent minimal world
* :mod:`artic_profile` / :mod:`validate_artic` - articulation physics profiles and validation

See ``docs/GRASP_BANDS.md``.
"""
