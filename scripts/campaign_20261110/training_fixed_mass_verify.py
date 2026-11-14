"""Read-only persistence proof for the bounded fixed-tray physics experiment."""
from pathlib import Path
import math
import numpy as np
from .contracts import require, digest
from .training_restoration import checked
from .training_fixed_mass_probe import DELTA, matches


def audit_persistence(root, references, probe_ref, request, records, episodes, *, additional_reconstruction_reference=None,
                      separate_reconstruction_lifecycle=False):
    probe = checked(root, probe_ref)
    directory = Path(probe_ref['path']).resolve().parent
    require(set(references) == {'audit', 'baseline'}, 'exact mass audit and baseline required')
    for key, name in [('audit', 'fixed_mass_audit.json'), ('baseline', 'fixed_mass_baseline_config.json')]:
        require(Path(references[key]['path']).resolve() == directory / name,
                'mass evidence is not from this native process output')
    audit = checked(root, references['audit']); baseline = checked(root, references['baseline'])
    require(request.get('fixed_mass_baseline') is True and
            audit.get('schema') == 'isg-fixed-tray-mass-inertia-audit-v1' and
            audit.get('physical_delta') == baseline.get('physical_delta') == DELTA and
            audit.get('runtime_compatibility') == request['runtime_compatibility'] and
            probe.get('runtime_compatibility') == request['runtime_compatibility'] and
            audit.get('physics_steps_added') == 0 and audit.get('production_released') is False,
            'mass audit scope or runtime differs')
    require(baseline.get('ordinary_scene_save_alone_is_sufficient') is False and
            Path(baseline['required_companion']).resolve() == directory/'fixed_mass_audit.json',
            'explicit overridden physical baseline companion missing')
    tro_ref = baseline['source_tro']
    require(tro_ref in [r['artifacts']['tro'] for r in records], 'mass baseline source differs')
    tro = checked(root, tro_ref)['ispatialgym']
    require(baseline['source_campaign_sha256'] == digest(tro['campaign']), 'mass baseline campaign differs')
    mass = [r for r in tro['constraints'] if r['type'] == 'mass']
    require(len(mass) == 2 and {r['object'] for r in mass} == {'tray.n.01_1','tray.n.01_2'}
            and all(r['mass_kg'] == 2. and not r.get('link') for r in mass), 'fixed tray inputs differ')
    expected = audit['expected']
    require(len(expected) == 2 and expected == baseline['intended'], 'overridden actor inventory differs')
    events = audit['events']
    require(events and events[0]['stage'] == 'world_initialized_after_one_override' and
            sum(e['stage'] == 'world_initialized_after_one_override' for e in events) == 1,
            'baseline must be captured once at initialization')
    first = events[0]; lifecycle = first.get('initial_actor_reconstruction', {})
    require(lifecycle.get('native_calls') == ['stop','play'] and
            lifecycle.get('observed_physics_step_calls') == [] and
            lifecycle.get('physics_steps_requested') == 0 and
            lifecycle.get('time_step_settings_before') == lifecycle.get('time_step_settings_after') and
            bool(lifecycle.get('time_step_settings_before')),
            'initial actor reconstruction lacks no-step measured proof')
    require(set(first.get('stock',{})) == set(expected), 'original physical baseline missing')
    for key, target in expected.items():
        stock = first['stock'][key]
        require(math.isfinite(stock['mass_kg']) and stock['mass_kg'] > 0 and target['mass_kg'] == 2.,
                'invalid intended or stock mass')
        require(np.allclose(target['inertia'],np.asarray(stock['inertia'])*(2./stock['mass_kg']),
                            atol=1e-9,rtol=1e-6), 'intended inertia is not the mass-scaled original')
    allowed = {'world_initialized_after_one_override','before_existing_stop_play','after_existing_stop_play',
               'before_episode_restore','after_episode_restore','after_native_mass_constraint'}
    restores = []; restarts = 0; pending_restart = False; pending_episode = None
    for event in events:
        stage = event['stage']
        require(stage in allowed and event.get('passed') is True and set(event['native']) == set(expected),
                'missing or failed native physical observation')
        require(all(matches(event['native'][key], target) for key,target in expected.items()),
                'native/USD mass or inertia persistence differs')
        if stage == 'before_existing_stop_play':
            require(not pending_restart, 'nested restart evidence'); pending_restart = True
        elif stage == 'after_existing_stop_play':
            require(pending_restart, 'restart completion without start'); pending_restart=False; restarts+=1
        elif stage == 'before_episode_restore':
            require(pending_episode is None, 'nested restore evidence'); pending_episode=event['instance_id']
        elif stage == 'after_episode_restore':
            require(pending_episode == event['instance_id'], 'restore completion identity differs')
            restores.append(pending_episode); pending_episode=None
    expected_ids = [checked(root, entry['report'])['instance_id'] for entry in probe['entries']]
    additional = 0
    require(not separate_reconstruction_lifecycle or
            (request.get('integrated_candidate_v1') is True and additional_reconstruction_reference is not None),
            'separate lifecycle evidence requires a declared integrated reconstruction')
    if additional_reconstruction_reference is not None:
        from .training_late_rebuild_probe import DELTA as LATE_DELTA
        require(request.get('late_incoming_physics_rebuild_v1') is True and
                Path(additional_reconstruction_reference['path']).resolve() == directory/'late_rebuild_audit.json',
                'additional reconstruction requires its exact native diagnostic audit')
        late = checked(root, additional_reconstruction_reference)
        require(late['physical_delta'] == LATE_DELTA and late['production_released'] is False and
                late['benchmark_credit'] == 0 and late['explicit_sdk_reconstructions'] == episodes and
                [event['instance_id'] for event in late['events']] == expected_ids,
                'additional reconstruction count or source identities differ')
        for event in late['events']:
            lifecycle = event['lifecycle']
            require(event['status'] == 'completed' and event['static_body_properties_bit_identical'] is True and
                    event['before_static_bodies'] and event['before_static_bodies'] == event['after_static_bodies'] and
                    event['whole_boundary_physics_step_calls'] == [] and event['reference_state_copied'] is False and
                    event['sdk']['explicit_sdk_reconstructions'] == event['sdk']['release_calls'] ==
                    event['sdk']['force_load_calls'] == 1 and event['sdk']['manual_attached_stage'] == 0 and
                    lifecycle['native_calls'] == ['stop','play'] and lifecycle['observed_physics_step_calls'] == [] and
                    lifecycle['physics_steps_requested'] == 0 and
                    lifecycle['time_step_settings_before'] == lifecycle['time_step_settings_after'],
                    'additional reconstruction physical persistence or no-step proof failed')
        additional = episodes
    # The common candidate captures the reconstruction callback before the mass
    # observer is installed. Its independent native audit above owns those
    # lifecycle events; requiring duplicate mass-hook events invents missing work.
    # This mode is explicit, and the caller still verifies the exact runtime proof.
    mass_observed_additional = 0 if separate_reconstruction_lifecycle else additional
    require(len(expected_ids) == episodes and restores == expected_ids and restarts == episodes-1+mass_observed_additional and
            not pending_restart and pending_episode is None, 'incomplete episode/reconstruction persistence matrix')
    return {'status':'verified','references':references,'actors':sorted(expected),
            'episodes':episodes,'initial_actor_reconstructions':1,'reused_actor_reconstructions':restarts,
            'additional_declared_reconstructions':additional,
            'additional_lifecycle_evidence': 'separate_native_audit' if separate_reconstruction_lifecycle else 'mass_and_native_audits',
            'physics_steps_added':0,'production_released':False,'benchmark_credit':0}
