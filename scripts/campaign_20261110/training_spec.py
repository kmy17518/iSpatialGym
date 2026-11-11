"""Original campaign training requirements, with explicit unverified release gates.

This is an engineering checklist, never a production authorization. File presence
means implementation exists; it does not turn a requirement into executed proof.
"""
from __future__ import annotations

import argparse
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
        ('G12','Scheduling distribution','Traversable-area/usable-edge or compartment weights, model diversity and soft train buckets drive refill; strict eval buckets preserved.', ['controller_groups.py','training_candidates.py']),
        ('G13','Failures and denominators','Retain failed sampling start/goal states and demo failures with separate reasons; include failed attempts/startup in known GPU cost; no inferred missing denominator.', ['goal_group_yield.py','../ispatialgym/sampling/campaign_runtime.py']),
        ('G14','Shared and applicable-path release','Validate shared pipeline behavior, then each applicable production path and exact configuration with real bounded canaries, scales/perturbations, exported provenance and strict acceptance. Release configurations incrementally; one family does not validate another path. No all-template, all-map or all100 eval barrier; preserve diversity capacity for unfinished configurations.', ['production_gate.py','training_eligibility.py']),
        ('G15','Priority and ownership','Prioritize canonical100 standard template validation; only under_large_01 remains parked. Each unvalidated template has a responsible engineer and discriminating next experiment. Training may start after the complete training pipeline and exact training template validate, without an all100 eval completion barrier.', ['standard_recovery.py']),
    ]
    rows=[]
    for key,scope,rule,files in common:
        paths=[code_root/'scripts/campaign_20261110'/p if not p.startswith('../') else code_root/p[3:] for p in files]
        rows.append(dict(id=key,family=scope,requirement=rule,status='end_to_end_unverified',
            implementation=[dict(path=str(p.relative_to(code_root)),exists=p.is_file(),sha256=file_digest(p) if p.is_file() else None) for p in paths],
            evidence=[],next_action='Join actual runtime and acceptance evidence; missing or partial implementation does not release this requirement.'))
    for i,(task,rule) in enumerate(task_rules().items(),1):
        for suffix,field in [('A','anchoring'),('M','models'),('S','scales'),('B','semantics'),('G','grouping')]:
            rows.append(dict(id=f'T{i:02}{suffix}',tasks=[task],requirement=rule[field],status='end_to_end_unverified',
                implementation=['training_candidates.py','materialize_train_rooms.py','support_height_profiles.py','ispatialgym/sampling/campaign_runtime.py'],
                evidence=[],next_action='Verify materialized and accepted distribution for this exact task; then add bounded missing-path canary.'))
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--original-prompt',type=Path,required=True);p.add_argument('--correction',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parents[2]
    write_json_atomic(a.out,dict(campaign_id=CAMPAIGN,observed_at=datetime.now(timezone.utc).isoformat(),
        status='blocked_pending_executed_evidence',production_authorization=False,
        sources=[dict(path=str(x.resolve()),sha256=file_digest(x)) for x in (a.original_prompt,a.correction)],
        decisions=['All candidates ready','Train buckets soft only','Max3 distinct group members; camera exempt',
            'Repeated goals across groups/splits permitted','Kohs model holdout; Wechsler category holdout',
            'Articulation one changed joint refers initial-to-goal','Structural walls/floors and tray/board holdout exceptions',
            'Model holdout891/category109 totals from subsequent owner decision','Budget3000USD; deadline2026-11-12T19:00Z without auto shutdown'],
        task_rules=task_rules(),requirements=checklist(root)))


if __name__=='__main__':main()
