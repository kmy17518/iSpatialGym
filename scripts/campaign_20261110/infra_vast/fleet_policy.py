"""Fleet bounds after compatible ready/booting capacity has been accounted for."""
import math


def activated_queue_reserve(config, states):
    """Planning-only upper cost for queued work, excluding assigned jobs.

    This is not an additional lease commitment or an admission limit. Rental
    admission reserves each actual offer's maximum lease cost exactly once.
    Existing leases may execute these jobs, so this estimate must not be added
    to their commitments as though the two were disjoint expenditures.
    """
    active = set(map(str, config.get('dispatch_job_ids', ())))
    rate = float(config['max_dph_gpu'])
    if not math.isfinite(rate) or rate < 0:
        raise ValueError('negative queue reservation rate')
    durations = [float(job.get('time_s', 10800)) for key, job in states.items()
                 if str(key) in active and job.get('status') == 'queued']
    if any(not math.isfinite(value) or value <= 0 for value in durations):
        raise ValueError('invalid activated job duration')
    return sum(durations) / 3600 * rate


def rental_ceiling(config, live_instances, uncovered_worker_demand):
    """Ready-demand mode has no fixed fleet ceiling; funding is checked per offer.

    ``uncovered_worker_demand`` must come from rental_needs, which subtracts
    already rented compatible idle and booting workers. This bound authorizes
    no rental by itself: exact job eligibility, offer compatibility, campaign
    commitments and actual prepaid funds remain separate mandatory checks.
    """
    if type(live_instances) is not int or live_instances < 0:
        raise ValueError('invalid live instance count')
    if type(uncovered_worker_demand) is not int or uncovered_worker_demand < 0:
        raise ValueError('invalid uncovered demand')
    mode = config.get('fleet_scaling_mode', 'fixed')
    if mode == 'ready_budget':
        return live_instances + uncovered_worker_demand
    if mode != 'fixed':
        raise ValueError('unknown fleet scaling mode')
    maximum = config['max_instances']
    if type(maximum) is not int or maximum < 0:
        raise ValueError('invalid fixed maximum')
    return maximum
