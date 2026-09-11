"""Task samplers. ``get_sampler_class(problem_name)`` maps ``<activity>-<def_id>`` to a sampler class."""

from __future__ import annotations


def get_sampler_class(problem_name: str):
    activity, did = problem_name.rsplit("-", 1)
    did = int(did)
    if activity == "camera_relocalization":
        from ispatialgym.sampling.tasks.camera_relocalization import CameraRelocalizationSampler

        return CameraRelocalizationSampler
    if activity == "object_scaling":
        from ispatialgym.sampling.tasks.held_object import ObjectScalingSampler

        return ObjectScalingSampler
    if activity == "mental_rotation":
        from ispatialgym.sampling.tasks.held_object import MentalRotationSampler

        return MentalRotationSampler
    if activity == "disembedding":
        from ispatialgym.sampling.tasks.disembedding import DisembeddingSampler

        return DisembeddingSampler
    if activity == "configuration_matching":
        if did == 0:
            from ispatialgym.sampling.tasks.block_design import BlockDesignSampler

            return BlockDesignSampler
        if did in (1, 2):
            from ispatialgym.sampling.tasks.articulation import ArticulationSampler

            return ArticulationSampler
        if did == 3:
            from ispatialgym.sampling.tasks.attachment import AttachmentSampler

            return AttachmentSampler
    if activity == "world_alignment":
        from ispatialgym.sampling.tasks.world_alignment import TableSurfaceSampler, WorldVerticalSampler

        return TableSurfaceSampler if did == 0 else WorldVerticalSampler
    if activity == "inter_object_relation_satisfaction":
        from ispatialgym.sampling.tasks.iors import IORSSampler

        return IORSSampler
    raise KeyError(f"no sampler for problem {problem_name!r}")
