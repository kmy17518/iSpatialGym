"""The success check, the text of the first user turn. Runs are image-only: the goal image is the task and no instruction
states it."""

from __future__ import annotations

SUCCESS_CHECK = (
    "Success check:\n"
    "Replicate the goal image exactly. The robot itself does not have to match it.\n"
    "Avoid any contact with the environment other than through the grippers."
)
