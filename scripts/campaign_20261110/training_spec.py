"""Original campaign training requirements, with explicit unverified release gates.

This is an engineering checklist, never a production authorization. File presence
means implementation exists; it does not turn a requirement into executed proof.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from scripts.campaign_20261110.contracts import file_digest, task_quotas, write_json_atomic

CAMPAIGN = 'campaign_20261110'


def task_rules():
    rules = {}
    for quota in task_quotas():
        task = quota.task
        room_task = task.startswith('camera_relocalization') or task in ('object_scaling', 'mental_rotation')
        large = task.endswith('_large_scale')
        per_class = task == 'configuration_matching-attachment' or task.startswith('configuration_matching-articulation_') and not large or task.startswith('inter_object_relation_satisfaction-') and not large
        if room_task:
            anchoring = '10 shared rooms per task; traversable-area weights; camera rooms must support required long paths'
        elif large:
            anchoring = 'Original scene target (articulation) or partner (relations); prefer other ready scene models, then same-category replacements no larger than original at the same anchor'
        elif per_class:
            anchoring = '10 supports per task class, with overlap allowed; maximum scaled layout fit and usable-edge weights'
        elif task == 'alignment-world_vertical':
            anchoring = '10 opaque wall supports; usable-wall-space weights; exclude transparent glass'
        elif task == 'alignment-axial':
            anchoring = '10 rectangular supports; usable-space weights'
        else:
            anchoring = '10 shared supports; maximum scaled layout fit and usable-edge weights'
        scaling = 'Two target scale draws relative to the actual default in [0.8,1.2]; exact scaled grasps and reach data'
        if task == 'camera_relocalization-standard':
            scaling = 'No task object to scale'
        elif task == 'camera_relocalization-obstructed':
            scaling = 'Per-instance obstacle-scale event probability0.5 with category-specific absolute commonsense bounds; persist exact applied scale'
        elif large:
            scaling += '; scale scene objects only when floor-resting and collision-safe; doors fixed; replacements no larger than original'
        if not room_task and not large and not task.startswith('alignment-'):
            scaling += '; two support Z-only height draws in [0.8,1.2], preserve XY and floor contact; measured height-specific reach data'
        if task in ('configuration_matching-attachment','configuration_matching-block_design'):
            scaling += '; coupled scaling preserves mating fit / board-block fit'
        elif task == 'disembedding':
            scaling += '; puzzle target and context pieces scale together'
        elif task.startswith('inter_object_relation_satisfaction-'):
            scaling += '; target/partner scales must preserve exact symbolic feasibility'
        models = ('No object model expansion' if task == 'camera_relocalization-standard' else
                  'Prefer additional ready models of eval categories, at most10 held-in per category or pair (11 including held-out); expand to11*N_TC with suitable new classes where specified; preserve global holdouts')
        if large:
            models = 'Up to11*N_TC ready target/partner models or pairs, at most10 held-in per category/pair; scene alternatives before size-constrained replacements; nonunique pairs only after unique pairs exhausted'
        if 'articulation_' in task:
            models += '; each movable reachable articulated joint counts separately, with fixed initial-to-goal non-target joints'
        semantics = {
            'camera_relocalization-standard':'Camera exempt from goal grouping; path buckets (0,3],(3,5],(5,10]m.',
            'camera_relocalization-obstructed':'Obstacle goal pose equals initial pose in BDDL, excluded from measure; commonsense household obstacles; camera group exemption.',
            'object_scaling':'Graspable targets, symmetry preferred; draw goal then robot base, grasp and start; base varies throughout room; distance buckets0.05–0.10/0.10–0.15/0.15–0.30m.',
            'mental_rotation':'Asymmetric, semantically reviewed targets;10 additional mental-rotation blocks; goal then variable robot base and distinct starts/grasps; angle buckets30–45/45–75/75–120degrees.',
            'disembedding':'Same start/goal target orientation; trays aligned to support XY and equal edge height; goal target not fixed to tray center; full distractor pools;10 additional LEF target+context-a puzzles; crowdedness<.3/.3–.45/>=.45.',
            'configuration_matching-block_design':'Four identical blocks per instance; board aligned to support XY and fixed across starts/goals;10 extra Kohs face permutations and predefined top patterns; coupled board/block scale; Kohs model holdout, Wechsler category holdout.',
            'configuration_matching-attachment':'Movable tabletop target/partner; verified mating ports and common scale; collect ready eval-category pairs then all suitable attachment pairs.',
            'alignment-world_vertical':'Unambiguous semantic upright direction; one fixed goal orientation per model/support across instances; vary start orientation and push/brace.',
            'alignment-axial':'Unambiguous semantic axis (texture can disambiguate); one fixed goal orientation per model/support; vary start orientation and push/brace.',
        }.get(task)
        if semantics is None and 'articulation_' in task:
            semantics = 'Opening starts fully closed and draws stable goal; closing draws stable start and ends fully closed. Exactly one joint differs initial-to-goal; temporary other-joint motion allowed. Distinct member start angle or grasp.'
        if semantics is None:
            semantics = 'Exact scaled symbolic fit and reviewed natural relative goals; random distance from middle80% of valid range; initial target orientation unrestricted except pure-push large objects; straight pushes; inside target yaw free. Book leans upright on bookcase wall; umbrella ends on hall-tree compartment, never floor.'
        rules[task] = dict(task=task, standard_classes=quota.standard_classes, anchoring=anchoring,
                          models=models, scales=scaling, semantics=semantics,
                          grouping='camera_exempt' if task.startswith('camera_relocalization') else 'max3 distinct successful members; goal/base/perturbations shared; successful eval needs training sibling; successful singleton may be training')
    return rules


def checklist(code_root: Path):
    common = [
        ('G01','Production authorization','Persistent fail-closed training block covers enqueue, direct submit, refill, retry, restart, migration and running producers; bounded canaries have immutable limits.', ['production_gate.py','controller_spool.py']),
        ('G02','Historical eligibility','Preserve original artifacts/IDs/acceptance; exclude noncompliant or unverified cohorts from goal and quota; retain eligible eval siblings/singletons; link compliant replacements.', ['training_eligibility.py','group_lane.py']),
        ('G03','Exact validation','Every released configuration has exact ready assets, native successful demo, own open-loop native replay, hardQA, acceptance and group compatibility.', ['controller_admission.py','replay_gate_quality.py']),
        ('G04','Readiness and global holdouts','All object candidates ready; model/category/room reservations across all roles; structural wall/floor and declared tray/board exceptions only.', ['readiness_catalog.py','reservation_policy.py']),
        ('G05','Group perturbations','Scene-removal event and lighting event independently probability0.5; identical within a goal group; never perturb eval. Unperturbed draws can legitimately qualify.', ['../ispatialgym/sampling/campaign_augmentation.py','goal_groups.py']),
        ('G06','Scene removal','Park only nonstructural, non-task-relevant objects; record exact poses, restore for subsequent instances; remove chairs near table-like task supports.', ['../ispatialgym/sampling/campaign_augmentation.py','private_scene.py']),
        ('G07','Lighting','Fixture intensity sampled normally around default with rendered sensible min/max; vary skybox color; same recorded settings collection/replay.', ['../ispatialgym/sampling/campaign_augmentation.py']),
        ('G08','Scale application','Two exact scales per applicable target/default and two Z-only support heights; couple puzzle, mating and board/block scales; floor-safe scene scaling only.', ['scale_prebuild.py','support_height_profiles.py','support_height_probe.py']),
        ('G09','Scale-specific dependencies','Exact model/asset/absolute scale/rest/support-height combinations load positive physically validated grasps and reach maps; predicted height is not measured proof.', ['scale_anchors.py','controller_scale.py','materialize_contact_dependencies.py']),
        ('G10','Replay restoration','Saved scene, removals, scales, lights and replacement poses reload identically in open-loop physics replay and batched rendering; alternating instances do not contaminate each other.', ['../ispatialgym/sampling/campaign_runtime.py','../ispatialgym/sampling/campaign_augmentation.py']),
        ('G11','Realized coverage','Account actual eligible accepted room/support/model/joint/scale/light/removal coverage versus per-task requirements; candidate lists and nominal template counts are not execution.', ['dashboard_evidence.py','training_eligibility.py']),
        ('G12','Scheduling distribution','Traversable-area/usable-edge or compartment weights, model diversity drive refill; bucket ranges and quotas are descriptive for every split.', ['controller_groups.py','training_candidates.py']),
        ('G13','Failures and denominators','Retain failed sampling start/goal states and demo failures with separate reasons; include failed attempts/startup in known GPU cost; no inferred missing denominator.', ['goal_group_yield.py','../ispatialgym/sampling/campaign_runtime.py']),
        ('G14','Shared and applicable-path release','Validate shared pipeline behavior, then each applicable production path and exact configuration with real bounded canaries, scales/perturbations, exported provenance and strict acceptance. Bulk standalone production remains held until all100 canonical eval classes validate. Pipeline implementation, exact dependencies and explicitly bounded end-to-end canaries including augmentation/batched replay/render restoration proceed independently now. After100/100, release bulk for each configuration only after shared/applicable-path/exact proof; one family does not validate another. Preserve diversity capacity for unfinished configurations.', ['production_gate.py','training_eligibility.py']),
        ('G15','Priority and ownership','Prioritize canonical100 standard template validation; the under_large_01 owner update is merged and requires fresh validation. Each unvalidated template has a responsible engineer and discriminating next experiment. Hold bulk standalone training until100/100 exact canonical validation; keep bounded pipeline canaries and exact dependency preparation progressing in parallel, including required distinct siblings and camera exemptions;1000accepted eval instances are not required. All100 canonical classes are required;99cannot substitute. Training pipeline/path/configuration gates still apply after100/100.', ['standard_recovery.py']),
    ]
    rows=[]
    for key,scope,rule,files in common:
        paths=[code_root/'scripts/campaign_20261110'/p if not p.startswith('../') else code_root/p[3:] for p in files]
        rows.append(dict(id=key,family=scope,requirement=rule,status='untested',
            implementation=[dict(path=str(p.relative_to(code_root)),exists=p.is_file(),sha256=file_digest(p) if p.is_file() else None) for p in paths],
            evidence=[],next_action='Join actual runtime and acceptance evidence; missing or partial implementation does not release this requirement.'))
    for i,(task,rule) in enumerate(task_rules().items(),1):
        for suffix,field in [('A','anchoring'),('M','models'),('S','scales'),('B','semantics'),('G','grouping')]:
            rows.append(dict(id=f'T{i:02}{suffix}',tasks=[task],requirement=rule[field],status='untested',
                implementation=['training_candidates.py','materialize_train_rooms.py','support_height_profiles.py','ispatialgym/sampling/campaign_runtime.py'],
                evidence=[],next_action='Verify materialized and accepted distribution for this exact task; then add bounded missing-path canary.'))
    return rows



EXTRA_REQUIREMENTS = [
    ('G16','Reusable selected model pool','Preload all selected compatible variants, switch active models, park inactive objects and export only active task bindings; prove isolation in one live world.', ['/root'], ['ispatialgym/sampling/sim_utils.py']),
    ('G17','Complete selected dependency preparation','Map every selected configuration of all100 classes to compatible positive grasp/reach dependencies; measured support heights, complete disjoint shards and native reader audits.', ['/root/templates_assets','/root/compute'], ['scripts/campaign_20261110/scale_prebuild.py']),
    ('G18','Applicable family paths','Validate each materially different task path independently; exact under-small evidence does not validate articulation, attachment, block, alignment, disembedding, floating or camera.', ['/root/dataset_contract','/root'], ['scripts/campaign_20261110/training_pipeline_status.py']),
    ('G19','Independent held-out evaluation','Bulk100-class hold must not block model/category evaluation, exact dependencies or bounded canaries; related unfinished held-out streams need active ownership/evidence.', ['/root/templates_assets','/root/compute','/root/dataset_contract'], ['scripts/campaign_20261110/production_gate.py']),
    ('G20','Resource and spending execution','Use useful compatible capacity across both accounts and Vast within3000USD including commitments; distinguish excluded queue from dispatchable work and include startup/failures in cost.', ['/root/compute'], ['scripts/campaign_20261110/vastctl.py']),
]



def compact_facts(value):
    """Counts and short labels only; full provenance stays in hashed source files."""
    allowed=('status','validated','total','state_counts','counts','summary','totals','eligible_accepted',
             'excluded','unverified','coverage_complete','review_inventory_complete','attempts_observed',
             'benchmark_credit','mode','standalone_training_allowed','owner_clarification')
    def small(item,depth=0):
        if item is None or isinstance(item,(bool,int,float)):return item
        if isinstance(item,str):return item[:300]
        if isinstance(item,dict)and depth<3:
            return {str(k):small(v,depth+1)for k,v in list(item.items())[:40]
                    if isinstance(v,(str,bool,int,float,type(None),dict))}
        return None
    return {key:small(value[key])for key in allowed if key in value}


def compact_family(family):
    return {key:family.get(key)for key in ('task','standard_classes','native_attempts_observed','native_written',
        'exact_configurations_with_demo_physics_export','required_configuration_denominator','denominator_status')}|{
        'exact_validation_status':family.get('exact_validation',{}).get('status'),
        'aba_status':{key:value.get('status')for key,value in family.get('aba_restoration',{}).items()},
        'model_pool_status':family.get('model_pool_switch_active_only_export',{}).get('status')}

def compact_model_pool(value):
    """Keep executed stage facts and references without duplicating native captures."""
    attempts=[]
    for attempt in value.get('attempts',[]):
        native=attempt.get('result',{})
        row={key:attempt.get(key,native.get(key)) for key in
             ('job_id','stage','mechanics_passed','independent_load_render_passed','failure')}
        row['observed_stages']=attempt.get('observed_stages',len(native['stages']) if 'stages' in native else None)
        row['evidence']=attempt.get('evidence')
        attempts.append(row)
    verified=value.get('verification',{})
    return {'status':value.get('status'),'required_model_count':value.get('required_model_count'),
        'benchmark_credit':0,'production_released':False,'blocker':value.get('blocker'),
        'unverified':value.get('unverified',[]),'attempts':attempts,
        'verification':{key:verified.get(key)for key in
            ('scope','switch','independent_restore','stages_each','scene_loads_for_switch','pool_rebuilds','clean_exit')},
        'process_accounting':{key:value.get('process_accounting',{}).get(key)for key in
            ('observed_at','job_id','record','query_returncode')}}


def executed_audit(code_root, campaign_root, sources):
    """Join authoritative observations without treating checklist placeholders as failures."""
    code_root,campaign_root=Path(code_root),Path(campaign_root)
    rows=checklist(code_root);observations={};errors=[]
    locations={
        'canonical':'infra/standard_recovery/canonical_validation.json',
        'eligibility':'accounting/current_eligibility.json',
        'training_eligibility':'accounting/training_eligibility.json',
        'pipeline':'training_pipeline_canaries/pipeline_status.json',
        'gate':'controllers/training_production_gate.json',
        'backend_holdout':'infra/capacity_recovery/holdout_backend_gate_audit_r1.json',
        'queue':'infra/capacity_recovery/vast_excluded_queue_audit_r2.json',
        'yield':'infra/goal_group_yield.json',
        'selected_assets':'training_preparation/full_scope_r1/selection_reconciliation.json',
        'dependency_jobs':'training_preparation/full_scope_r1/jobs_asset_geometry.json',
        'full_preparation':'training_preparation/status.json',
        'model_pool':'training_preparation/pool_pair_r1/status.json',
        'exact_loading':'training_preparation/G09_exact_loading_current.json',
        'refill_distribution':'training_preparation/G12_refill_distribution_current.json',
    }
    # The canonical observer has one authoritative output; accept its established
    # validation directory or older monitor location only when exactly present.
    for label,relative in locations.items():
        path=campaign_root/relative
        if label=='canonical' and not path.exists():path=campaign_root/'monitoring/canonical_validation.json'
        if not path.exists():continue
        try:
            value=json.loads(path.read_text())
            if value.get('campaign_id',CAMPAIGN)!=CAMPAIGN:raise ValueError('foreign observation')
            if label in ('exact_loading','refill_distribution'):
                reference=value['source_evidence'];source=Path(reference['path']).resolve()
                if not source.is_relative_to(campaign_root.resolve()) or file_digest(source)!=reference['sha256']:
                    raise ValueError('requirement observation source bytes changed or outside campaign')
            observations[label]={'reference':{'path':str(path),'sha256':file_digest(path)},'value':value,
                                 'observed_at':value.get('observed_at')}
        except Exception as exc:errors.append({'path':str(path),'error':repr(exc)})
    def evidence(label,scope,stage,facts):
        if label not in observations:return None
        source=observations[label]
        return {**source['reference'],'observed_at':source['observed_at'],'scope':scope,'stage':stage,'facts':facts}
    for key,scope,rule,owners,files in EXTRA_REQUIREMENTS:
        rows.append({'id':key,'family':scope,'requirement':rule,'status':'untested','owners':owners,
                     'implementation':[{'path':f,'exists':(code_root/f).is_file(),
                         'sha256':file_digest(code_root/f)if(code_root/f).is_file()else None}for f in files],
                     'evidence':[]})
    pipeline=observations.get('pipeline',{}).get('value',{});families={r['task']:r for r in pipeline.get('family_paths',[])}
    for row in rows:
        key=row['id'];task=(row.get('tasks')or[None])[0]
        row['source_requirements']=sources
        row['evidence']=[];row['scope_status']='unknown_until_evidence_reconciled'
        row['implementation_status']='present' if any(isinstance(x,dict)and x.get('exists') for x in row['implementation']) else 'not_audited'
        row['owners']=row.get('owners') or (['/root/templates_assets','/root/compute']if key.startswith('T')and key[-1]in('A','M','S')else ['/root/dataset_contract'])
        row['next_action']=('Materialize exact selected configurations and map each to positive measured dependencies; stream ready builds independently.'
                            if key.startswith('T')and key[-1]in('A','M','S') else
                            'Join this exact requirement to the scoped native evidence and run a bounded discriminating check for missing behavior.')
        row['exit_criteria']='Executed evidence covers the stated scope; exact configuration proof never substitutes for full selected coverage.'
        if task and task in families:
            family=families[task]
            row['evidence']=[evidence('pipeline',task,'scoped_pipeline',compact_family(family))]
            if family['exact_configurations_with_demo_physics_export']:row['status']='partial'
            row['next_action']='Advance the next dependency-ready configuration of '+task+' through its own native/demo/physics/render gates; retain full selected denominator.'
        mapping={'G01':('gate','dispatch_policy'),'G02':('eligibility','historical_and_current_acceptance'),
            'G03':('canonical','exact_standard_chains'),'G04':('canonical','accepted_exact_template_checks'),
            'G05':('pipeline','recorded_group_perturbations'),'G06':('pipeline','recorded_scene_removal'),
            'G07':('pipeline','recorded_lighting'),'G08':('pipeline','recorded_scales'),
            'G09':('exact_loading','exact_native_dependency_loading'),
            'G10':('pipeline','aba_restoration'),'G11':('training_eligibility','observed_accepted_coverage'),
            'G12':('refill_distribution','weighted_refill_and_soft_bucket_evidence'),
            'G13':('yield','attempt_and_cost_denominators'),'G14':('gate','bulk_and_canary_policy'),
            'G15':('canonical','canonical_progress'),'G18':('pipeline','family_path_coverage'),
            'G16':('model_pool','scoped_model_pool_mechanics'),'G17':('full_preparation','full_selected_dependency_status'),'G19':('backend_holdout','backend_gate_decisions'),'G20':('queue','queue_disposition')}
        if key in mapping:
            label,stage=mapping[key]
            if label in observations:
                value=observations[label]['value']
                facts=compact_facts(value)
                row['evidence'].append(evidence(label,'only the explicitly enumerated observations',stage,facts));row['status']='partial'
        if key=='G16':
            value=observations.get('model_pool',{}).get('value',{})
            row['next_action']=value.get('next_action','Root must execute model-pool preload/activation/active-only export and audit independent reloads.')
            if row['evidence']:
                row['evidence'][-1]['scope']='Only the explicit model pair and native stages; process, independent restore, TRO, demo and action replay remain separate.'
                row['evidence'][-1]['facts']=compact_model_pool(value)

        if key=='G17':row['next_action']='Assets reconciles all selected candidates and exact dependencies; compute dispatches disjoint positive-grasp/reach builds across both accounts.'
        if key in ('G09','G12'):
            row['owners']=['/root/templates_assets','/root'] if key=='G09' else ['/root','/root/compute']
            label='exact_loading' if key=='G09' else 'refill_distribution'
            value=observations.get(label,{}).get('value',{})
            row['next_action']=value.get('next_action') or (
                'Bind each selected configuration to its measured-height, exact model/scale/direction/arm library and native reader/query proof; one demonstrated configuration does not close the full denominator.' if key=='G09' else
                'Implement measured-area/usable-edge/compartment weighted refill across the complete selected inventory; preserve model/scale capacity and descriptive bucket metadata, then run a bounded zero-credit decision/native canary. Missing geometry has an owned preparation dependency.')
            if row['evidence']:
                row['evidence'][-1]['scope']=value.get('scope','Only the exact executed configurations in the hashed source evidence')
                row['evidence'][-1]['source_evidence']=value['source_evidence']
        if key in ('G05','G06','G07','G08','G10','G18'):
            row['owners']=['/root/dataset_contract','/root']
            verified=[batch for batch in pipeline.get('batch_stages',[])if batch.get('status')=='batch_restoration_verified']
            row['next_action']=('Keep verified exact A/B/A evidence scoped; advance each other dependency-ready family/configuration through its own augmentation, demo, physics and rendering checks. Model-pool switching remains separate.' if verified else 'Execute the prepared exact A/B/A physics and rendering probes, then independently validate each other applicable family path and model-pool switching.')
        row['scope_status']='partial_executed_evidence'if row['status']=='partial'else'unknown_no_executed_evidence'
        row['last_observed_progress']=max((str(e.get('observed_at')or'')for e in row['evidence']),default=None)
        row['chain']={'dependencies':[observations[label]['reference'] for label in ('selected_assets','dependency_jobs')if label in observations] if key=='G17' or task else [],
            'owners':row['owners'],'active_work':{'state':'owned_next_action','action':row['next_action']},
            'artifacts':[{'path':e['path'],'sha256':e['sha256'],'observed_at':e.get('observed_at')}for e in row['evidence']],
            'verification':{'state':row['scope_status'],'scope':'explicit evidence only','complete_requirement':False},
            'acceptance':{'state':'observed_current_projection'if key in ('G02','G11','G15')and row['evidence']else'not_claimed',
                          'benchmark_canary_credit':0}}

    return {'schema':'isg-training-requirements-audit-v2','campaign_id':CAMPAIGN,
        'observed_at':datetime.now(timezone.utc).isoformat(),'status':'partial_executed_audit',
        'production_authorization':False,'sources':sources,'requirements':rows,'task_rules':task_rules(),
        'counts':dict(Counter(r['status']for r in rows)),'observation_errors':errors,
        'unknown_is_failure':False,'implementation_presence_is_execution':False,
        'family_evidence_transfers_to_other_tasks':False,
        'superseded_instructions':[
            {'instruction':'Hold all standalone canaries until100canonical','current':'Bulk-only hold; bounded zero-credit canaries and all dependency preparation proceed now','source':sources[-1]if sources else None},
            {'instruction':'Owner-artifact holds for five standard classes','current':'All owner artifacts released; fresh exact native/replay/sibling gates still required','source':sources[-1]if sources else None},
            {'instruction':'Treat130 generated unverified placeholders as actual failed requirements','current':'Placeholders mean unknown; this audit distinguishes scoped execution from missing evidence','source':sources[-1]if sources else None}],
        'independent_workstreams':['remaining_standard_eval','model_holdout','category_room_holdout',
            'full_selected_dependencies','bounded_pipeline_validation','model_pool_implementation'],
        'bulk_release_condition':'100canonical AND shared/applicable-path/exact-configuration proof; not a dependency or holdout barrier'}


def publish_executed_audit(code_root,campaign_root,sources,output):
    value=executed_audit(code_root,campaign_root,sources);output=Path(output)
    if output.exists():
        archive=output.parent/'training_requirements_history'/(str(time.time_ns())+'.json')
        archive.parent.mkdir(parents=True,exist_ok=True);archive.write_bytes(output.read_bytes())
        value['previous_observation']={'path':str(archive),'sha256':file_digest(archive)}
    write_json_atomic(output,value)
    return value

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--original-prompt',type=Path,required=True);p.add_argument('--correction',type=Path,required=True);p.add_argument('--campaign-root',type=Path);p.add_argument('--source',type=Path,action='append',default=[]);p.add_argument('--interval',type=float,default=0)
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    if a.campaign_root:
        sources=[{'path':str(x.resolve()),'sha256':file_digest(x)}for x in [a.original_prompt,a.correction,*a.source]]
        while True:
            value=publish_executed_audit(root,a.campaign_root,sources,a.out)
            print(json.dumps({'observed_at':value['observed_at'],'counts':value['counts']}),flush=True)
            if a.interval<=0:return
            time.sleep(a.interval)
    write_json_atomic(a.out,dict(campaign_id=CAMPAIGN,observed_at=datetime.now(timezone.utc).isoformat(),
        status='blocked_pending_executed_evidence',production_authorization=False,
        sources=[dict(path=str(x.resolve()),sha256=file_digest(x)) for x in (a.original_prompt,a.correction)],
        decisions=['All candidates ready','All bucket ranges and quotas descriptive; native physics and task constraints preserved','Max3 distinct group members; camera exempt',
            'Repeated goals across groups/splits permitted','Kohs model holdout; Wechsler category holdout',
            'Articulation one changed joint refers initial-to-goal','Structural walls/floors and tray/board holdout exceptions',
            'Model holdout891/category109 totals from subsequent owner decision','Budget3000USD; deadline2026-11-12T19:00Z without auto shutdown'],
        task_rules=task_rules(),requirements=checklist(root)))


if __name__=='__main__':main()
