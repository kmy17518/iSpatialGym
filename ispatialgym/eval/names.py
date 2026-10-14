"""Eval names: task family -> task -> subtask, from a subtask's variant (the sampler's grounded setup of it).

The variant (``alignment-axial-board_game-breakfast_table``: a BDDL problem's ``(:variant ...)``, a TRO's ``task_variant``,
a sidecar's ``variant``) fills the name template of its task's iddl file (``alignment-axial-{target}-{support}``) with one
``-``-separated token per slot. The task is the template's fixed tokens (``alignment_axial``); the subtask is the task and
the slot tokens except the support (``alignment_axial_board_game``), a token that names one object model
(``bar_soap_utgixp``) or one scene object (``drop_in_sink_lkklqs_0``) reduced to its category. ``isg_customs.json`` is keyed
by subtask.
"""
from __future__ import annotations

import functools
import os
import re

from ispatialgym.activity.problem import canonical_variant
from ispatialgym.eval.instances import data_root
from ispatialgym.oracle.sidecar import family_of_subtask, subtask_of_variant, variant_templates

#: task names that are not the template's fixed tokens joined by ``_``
TASK_RENAMES = {"alignment-world_vertical": "alignment_vertical"}
SUPPORT_SLOT = "{support}"
#: ``<category>_<model>`` or a scene object ``<category>_<model>_<k>``; a category alone may end in a six-letter word
#: (``pepper_shaker``), so the model must exist in the asset datasets
_OBJECT_TOKEN = re.compile(r"^(?P<category>[a-z0-9_]+?)_(?P<model>[a-z]{6})(?:_\d+)?$")
_ASSET_DATASETS = ("behavior-1k-assets", "ispatialgym-assets")


@functools.lru_cache(maxsize=None)
def _is_model(category: str, model: str) -> bool:
    return any(os.path.isdir(os.path.join(data_root(), d, "objects", category, model)) for d in _ASSET_DATASETS)


def object_category(token: str) -> str:
    """``bar_soap_utgixp`` / ``drop_in_sink_lkklqs_0`` -> ``bar_soap`` / ``drop_in_sink``; any other token unchanged."""
    m = _OBJECT_TOKEN.match(token)
    return m.group("category") if m and _is_model(m.group("category"), m.group("model")) else token


def _template(variant: str):
    iddl_id = subtask_of_variant(variant)
    return iddl_id, variant_templates()[iddl_id][1]


def _is_slot(token: str) -> bool:
    return token.startswith("{") and token.endswith("}")


def task_family(variant: str) -> str:
    return family_of_subtask(_template(variant)[0])


def task_name(variant: str) -> str:
    """``alignment-axial-board_game-breakfast_table`` -> ``alignment_axial``."""
    fixed = []
    for token in _template(variant)[1].split("-"):
        if _is_slot(token):
            break
        fixed.append(token)
    prefix = "-".join(fixed)
    return TASK_RENAMES.get(prefix, prefix.replace("-", "_"))


def subtask_name(variant: str) -> str:
    """``alignment-axial-board_game-breakfast_table`` -> ``alignment_axial_board_game`` (the support left out, object ids
    reduced to their category)."""
    variant = canonical_variant(variant)
    template = _template(variant)[1]
    kept = [object_category(tok) for tok, slot in zip(variant.split("-"), template.split("-")) if _is_slot(slot) and slot != SUPPORT_SLOT]
    return "_".join([task_name(variant)] + kept)
