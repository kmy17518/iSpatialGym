"""Keep a campaign touching push on the line to its measured settled goal."""
import math


def settled_endpoint(start, settled, direction, distance, base_end, *, fixed_goal=False):
    """Return an exact collinear endpoint adjustment; never invent push-on slack."""
    values = [*start[0], *start[1], *settled[0], *settled[1], *direction, distance, *base_end]
    if not all(math.isfinite(float(v)) for v in values):
        raise ValueError('nonfinite touching endpoint')
    if len(direction) != 3 or abs(direction[2]) > 1e-6 or abs(math.hypot(*direction[:2])-1) > 1e-5:
        raise ValueError('touching endpoint needs the original horizontal unit line')
    movement = [float(b)-float(a) for a,b in zip(start[0],settled[0])]
    actual = sum(movement[i]*direction[i] for i in range(2))
    side = movement[0]*direction[1]-movement[1]*direction[0]
    norm = math.sqrt(sum(float(v)**2 for v in start[1])*sum(float(v)**2 for v in settled[1]))
    if norm <= 0:
        raise ValueError('invalid touching endpoint orientation')
    turn = 2*math.acos(min(1.,abs(sum(float(a)*float(b) for a,b in zip(start[1],settled[1]))/norm)))
    delta = actual-float(distance)
    if abs(side) > .001 or turn > .01 or delta < -.001 or delta > .02:
        raise ValueError('settled touching goal left the bounded straight push line')
    if fixed_goal and abs(delta) > .001:
        raise ValueError('fixed group goal would require a different push endpoint')
    if delta <= 1e-5:
        return None
    return {'distance_m':actual,'advance_m':delta,
            'base_end':[base_end[0]+delta*direction[0],base_end[1]+delta*direction[1],base_end[2]],
            'delta_xyz':[delta*v for v in direction]}
