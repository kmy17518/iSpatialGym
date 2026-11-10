# Campaign owner decisions — November 10, 2026

## Goal reuse policy v2

The owner's answer to the large-articulation closing viewpoint question was:

> Don't care about repeats between train and eval when generating for the whole benchmark.

This applies throughout the benchmark. Independently generated groups may share canonical goal poses, including between training and evaluation. Changing a robot viewpoint, pair translation, or relation gap solely to avoid a previously used goal is no longer required.

Fresh generation and unique group/instance ownership remain mandatory. A group still has at most three members with distinct starts or grasps, and a standard-evaluation member still needs a successful unperturbed training sibling. Every exact physical validation key retains its demonstration and genuine open-loop replay gate. Global model, category, room, asset-byte, and scale checks are unchanged.

The allocator migration removes only the unique canonical-goal constraint. It hashes accepted table contents before and after, preserves group-ID uniqueness and existing artifact/export bytes, and records the exact owner statement in database metadata and `manifests/policy_migrations/goal_reuse_v2.json`. Existing accepted data remains valid.

For narrow supports or constrained floor goals, pair queues may reuse the exact successful fresh anchor's goal, partner, and release poses while sampling new starts and grasps under the existing geometry, contact, visibility, IK, and physics checks. Region translation and gap proposals remain available as optional modes.
