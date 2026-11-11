"""Read exact native replay evidence; never promote result-summary success alone."""
import datetime
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assess(gate, gate_path):
    reasons=[];report_path=None;report_hash=None;annotated=None
    if gate.get('success') is not True:reasons.append('gate_result_failed')
    if gate.get('replay_rc')!=0 or gate.get('replay_mode')!='physics_action_open_loop':
        reasons.append('native_action_replay_not_successful')
    try:
        source=Path(gate['instance']);raw=json.loads(source.read_text())
        if digest(source)!=gate.get('tro_sha256'):raise ValueError('source TRO hash differs')
        iid=raw['ispatialgym']['instance_id'];annotated=raw['ispatialgym'].get('campaign') is not None
        parent=Path(gate_path).resolve().parent;identity=gate.get('scene_identity') or {}
        declared=gate.get('replay_meta_paths')
        if declared is not None and (not isinstance(declared,list) or not declared):raise ValueError('invalid replay report list')
        paths=[Path(p).resolve() for p in declared] if declared else list((parent/'replay').rglob('meta.json'))
        if len(set(paths))!=len(paths):raise ValueError('duplicate native replay reports')
        if identity.get('replay_report'):
            report_path=Path(identity['replay_report']).resolve()
            if identity.get('verified_replay') is not True:raise ValueError('scene replay identity not verified')
            if declared is not None and paths.count(report_path)!=1:raise ValueError('scene/native report identity differs')
            paths=[report_path]
        matches=[]
        for path in paths:
            if not path.is_relative_to(parent):raise ValueError('native replay report outside gate attempt')
            value=json.loads(path.read_text())
            if value.get('instance_id')==iid and value.get('mode')=='replay':matches.append((path,value))
        if len(matches)!=1:raise ValueError('exact native replay report is absent or ambiguous')
        report_path,report=matches[0];report_hash=digest(report_path)
        if identity.get('replay_report') and identity.get('replay_report_sha256')!=report_hash:
            raise ValueError('verified native replay report hash differs')
        if report.get('status')!='success' or report.get('success') is not True:reasons.append('native_report_failed')
        qa=report.get('qa') or {}
        if qa.get('enabled') is not True:reasons.append('native_qa_missing_or_disabled')
        hard=set(qa.get('hard_triggered') or [])
        hard.update(key for key,value in qa.get('metrics',{}).items() if value.get('mode')=='hard' and value.get('triggered'))
        if hard:reasons.append('native_hard_qa:'+','.join(sorted(hard)))
        if report.get('goal_override') is not None:reasons.append('native_goal_override')
        if annotated and report.get('loader',{}).get('campaign_restoration',{}).get('scale_verified') is not True:
            reasons.append('native_campaign_restoration_unverified')
        if gate.get('result',{}).get('qa_hard'):reasons.append('result_hard_qa')
        if gate.get('result',{}).get('goal_override') is not None:reasons.append('result_goal_override')
    except (KeyError,TypeError,ValueError,OSError) as error:
        reasons.append('native_evidence_unavailable:'+str(error))
    return {'schema':'isg-native-replay-quality-v1','observed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'passed':not reasons,'reasons':reasons,'annotated_source':annotated,
            'native_report':str(report_path) if report_path else None,'native_report_sha256':report_hash}
