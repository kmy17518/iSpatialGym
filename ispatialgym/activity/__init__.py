"""iSpatialGym activity definitions: the ``ispatialgym`` BDDL domain, ``:alias`` / ``:groups`` parsing, group
expansion, and compilation into the stock ``bddl`` expression trees consumed by the evaluator.

Activity names: retired names (``LEGACY_ACTIVITIES``, ``world_alignment`` -> ``alignment``) are accepted everywhere
and equal their canonical name (``canonical_activity``). The optional per-activity ``problems.json`` registry maps
aliases to definition ids (``load_problem_registry`` / ``alias_of`` / ``id_of_alias`` / ``validate_problem_registry``).
"""

from ispatialgym.activity.problem import (
    ACTIVITY_DEFINITIONS_ROOT,
    AGGREGATE_PREDICATES,
    ALIAS_RE,
    DOMAIN_NAME,
    GROUP_SYMBOLS,
    LEGACY_ACTIVITIES,
    LEGACY_GROUP_SYMBOLS,
    MEASURE_PREDICATES,
    PROBLEM_REGISTRY_FILE,
    ISGCompiledTask,
    ISGProblem,
    ISGProblemError,
    activity_dir,
    activity_names,
    alias_of,
    canonical_activity,
    collect_predicate_args,
    id_of_alias,
    list_activities,
    list_problem_ids,
    load_problem,
    load_problem_registry,
    problem_aliases,
    problem_path,
    problem_registry_path,
    same_activity,
    validate_problem_registry,
)

__all__ = [
    "ACTIVITY_DEFINITIONS_ROOT",
    "AGGREGATE_PREDICATES",
    "ALIAS_RE",
    "DOMAIN_NAME",
    "GROUP_SYMBOLS",
    "LEGACY_ACTIVITIES",
    "LEGACY_GROUP_SYMBOLS",
    "MEASURE_PREDICATES",
    "PROBLEM_REGISTRY_FILE",
    "ISGCompiledTask",
    "ISGProblem",
    "ISGProblemError",
    "activity_dir",
    "activity_names",
    "alias_of",
    "canonical_activity",
    "collect_predicate_args",
    "id_of_alias",
    "list_activities",
    "list_problem_ids",
    "load_problem",
    "load_problem_registry",
    "problem_aliases",
    "problem_path",
    "problem_registry_path",
    "same_activity",
    "validate_problem_registry",
]
