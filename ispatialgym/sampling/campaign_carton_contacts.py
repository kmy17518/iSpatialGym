"""Strict native checks for the carton press-to-slide transition."""
import math


def press_slide_extensions(*, track, set_flap, q_press, press_waypoints, direction,
                           slide_pose, q_other, angle, lower, tolerance):
    """Check the slide after two bounded residual presses used by the oracle.

    The nominal program remains unchanged. Every extra leg uses the native
    branch-continuous tracking callback; any failure rejects this candidate.
    """
    from ispatialgym.oracle.execute import ExecuteConfig, extend_push, push_geometry

    geometry = push_geometry(press_waypoints, direction)
    if geometry.kind != 'arc':
        return 'carton residual screen requires an arc press', None
    cap = max(ExecuteConfig().push_residual_max_fraction * geometry.total, 2 * float(tolerance))
    sign = math.copysign(1., geometry.angle)
    try:
        for fraction in (.5, 1.):
            extra = fraction * cap
            set_flap(max(float(lower), float(q_other) - float(angle) - extra))
            q, previous = q_press.clone(), press_waypoints[-1]
            for pose in extend_push(geometry, previous, sign * extra):
                q, error = track(q, previous, pose, f'press residual {extra:.4f} rad')
                if error:
                    return error, cap
                previous = pose
            _, error = track(q, previous, slide_pose, f'slide after residual {extra:.4f} rad')
            if error:
                return error, cap
    finally:
        set_flap(float(q_other) - float(angle))
    return None, cap
