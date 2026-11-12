"""Finite correction budget for opt-in articulated arc pushes."""
import math


def next_correction(residual, previous, tolerance, minimum, used, cap, rounds, direction):
    """Return a live residual only while it improves and fits the original cap."""
    values = (residual, previous, tolerance, minimum, used, cap, direction)
    if any(v is None or not math.isfinite(float(v)) for v in values):
        return None
    if not 1 <= rounds < 3 or direction not in (-1.0, 1.0):
        return None
    remaining = float(residual) * direction
    if tolerance <= 0 or minimum <= 0 or used < 0 or cap <= 0:
        return None
    if remaining <= tolerance or remaining < minimum:
        return None
    if abs(previous) - abs(residual) < 0.001:
        return None
    if used + abs(residual) > cap:
        return None
    return float(residual)
