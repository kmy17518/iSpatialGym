"""Fail-closed physical clearance for an optional support-place retreat."""


def target_clear(robot_path, target_path, arm, held, pairs):
    if held or not robot_path or not target_path or arm not in ('left', 'right') or pairs is None:
        return False
    finger_prefix = robot_path.rstrip('/') + '/' + arm + '_gripper_finger'
    target_prefix = target_path.rstrip('/') + '/'
    for a, b in pairs:
        a, b = str(a), str(b)
        if (a.startswith(finger_prefix) and b.startswith(target_prefix)) or (b.startswith(finger_prefix) and a.startswith(target_prefix)):
            return False
    return True
