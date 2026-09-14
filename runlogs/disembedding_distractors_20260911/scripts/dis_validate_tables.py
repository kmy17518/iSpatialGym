#!/usr/bin/env python3
"""Launcher for validate_distractors.py whose file name ends in ``validate_tables.py`` on purpose.

The table-selection queue running on the same GPU (``runlogs/table_selection_20260911/scripts/queue_scenes.sh``)
keeps at most ``ISG_TABLE_MAXP`` (2) OmniGibson processes alive by counting processes whose command line matches
``python .*validate_tables.py``. Three Isaac Sim processes do not fit into the 24 GB of the RTX 4090 (its 20:53 launch
of a fourth process segfaulted), so this distractor validation registers itself as one of those two slots: while it
runs, the table queue keeps one process instead of two, and nobody runs out of GPU memory. ``run_phase.sh`` only
launches it when at most one table-selection process is alive.
"""
import os
import runpy
import sys

sys.argv[0] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validate_distractors.py")
runpy.run_path(sys.argv[0], run_name="__main__")
