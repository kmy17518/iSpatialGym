"""Fleet bounds after compatible ready/booting capacity has been accounted for."""


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
