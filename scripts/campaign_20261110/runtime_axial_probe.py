"""Bounded, nonaccepting fixed-goal axial candidate-search experiment.

The source native runtime, goal, strict bucket, and checks remain exact. Only
how many map-feasible candidates reach the existing exact checker changes.
"""
from pathlib import Path
import argparse,hashlib,inspect,json,os,sys,textwrap,time


def expanded_scan(source, limit):
    if limit not in (6,24):raise ValueError('only paired baseline/24-candidate experiment allowed')
    old='len(paired) + len(push_only) >= 6'
    if source.count(old)!=1:raise ValueError('native candidate bound differs')
    return source.replace(old,'len(paired) + len(push_only) >= '+str(limit))


def expanded_candidates(source, limit):
    if limit not in (64,256):raise ValueError('only paired64/256 checked-candidate experiment allowed')
    old='order[:64] if campaign_plan is not None else order[:6]'
    if source.count(old)!=1:raise ValueError('native checked-candidate bound differs')
    return source.replace(old,'order[:'+str(limit)+'] if campaign_plan is not None else order[:6]')


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',type=Path,required=True);a=p.parse_args()
    q=json.loads(a.request.read_text());root=Path(q['root']).resolve();output=Path(q['output']).resolve()
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    assert root.name=='campaign_20261110' and output.is_relative_to(root/'diagnostics')
    assert q['class_id']=='alignment-axial_04' and q['benchmark_credit']==0 and q['writer_disabled']is True
    assert not output.exists();output.mkdir(parents=True)
    for name,digest in q['inputs'].items():assert sha(name)==digest,name
    assert sha(__file__)==q['runner_sha256']
    native=Path(q['native_runtime']);assert Path(os.environ['ISGE_CODE'])==native
    assert os.environ['ISPATIALGYM_CAMPAIGN_CONFIG']==q['config']
    assert os.environ['ISPATIALGYM_CAMPAIGN_GOAL_PLAN']==q['goal_plan']
    assert q['max_draws']==2 and q['scan_limit']in(6,24) and q.get('candidate_limit',64)in(64,256)
    os.environ['ISPATIALGYM_SAMPLING_CACHE']=str(output/'sampling_cache')
    os.environ['ISPATIALGYM_MAP_OVERLAY_SHARDS']=str(output/'map_overlay')
    sys.path.insert(0,str(native))
    from ispatialgym.sampling import multiply_isg
    from ispatialgym.sampling.v2.core import SamplerV2
    from ispatialgym.sampling.v2.tasks.alignment_axial import AlignmentAxial
    import omnigibson as og
    module=sys.modules[AlignmentAxial.__module__]
    assert Path(module.__file__).resolve().is_relative_to(native)
    assert sha(module.__file__)==q['native_module_sha256']
    source=textwrap.dedent(inspect.getsource(AlignmentAxial._scan_candidates));namespace=dict(vars(module))
    exec(compile(expanded_scan(source,q['scan_limit']),str(a.request)+'#exact-search-bound', 'exec'),namespace)
    candidate_source=textwrap.dedent(inspect.getsource(AlignmentAxial.sample_instance))
    exec(compile(expanded_candidates(candidate_source,q.get('candidate_limit',64)),str(a.request)+'#checked-candidate-bound','exec'),namespace)
    original_scan=AlignmentAxial._scan_candidates;original_instance=AlignmentAxial.sample_instance;original_sample=SamplerV2.sample;original_writer=SamplerV2.write_instance
    report={'schema':'isg-runtime-axial-search-probe-v1','request_sha256':sha(a.request),'started_at':time.time(),'native_runtime':str(native),'class_id':q['class_id'],'scan_limit':q['scan_limit'],'draws':[],'scans':[],'benchmark_credit':0,'eligible_for_acceptance':False,'writer_disabled':True}
    report['candidate_limit']=q.get('candidate_limit',64)
    def save():
        report['observed_at']=time.time();report['published_tro_files']=len(list(output.rglob('*-tro_state.json')))
        assert report['published_tro_files']==0
        temp=output/'diagnosis.tmp';temp.write_text(json.dumps(report,indent=2)+'\n');temp.replace(output/'diagnosis.json')
    class StopProbe(BaseException):pass
    def scan(self,rng,init,order):
        result=namespace['_scan_candidates'](self,rng,init,order)
        report['scans'].append({'input_candidates':len(order),'map_feasible_candidates':len(result),'candidate_indices':[int(x[0])for x in result]});save();return result
    def sample(self,rng):
        assert len(report['draws'])<q['max_draws'];row={'started_at':time.time(),'draw':len(report['draws'])};report['draws'].append(row)
        try:
            result=original_sample(self,rng);row['native_sample_passed']=True
            row['manipulation_candidate']=result.summary.get('manipulation_candidate');row['program_check']=result.summary.get('program_check')
            return result
        except Exception as exc:
            row.update(native_sample_passed=False,error=repr(exc));raise
        finally:row['finished_at']=time.time();save()
    def no_write(*args,**kwargs):report['stopped_before_publication']=True;save();raise StopProbe()
    old_shutdown=og.shutdown
    def shutdown(*args,**kwargs):report['finished_at']=time.time();save();return old_shutdown(*args,**kwargs)
    AlignmentAxial._scan_candidates=scan;AlignmentAxial.sample_instance=namespace['sample_instance'];SamplerV2.sample=sample;SamplerV2.write_instance=no_write;og.shutdown=shutdown
    argv=['--split','train','--template',q['template'],'--instances','1','--start-index',str(q['source_index']),'--max-attempts',str(q['max_draws']),'--customs',q['customs'],'--keep-template-scene','--no-write-back','--out',str(output/'native_prefix')]
    if q.get('bddl_root'):argv+=['--bddl-root',q['bddl_root']]
    report['sampler_arguments']=argv;save()
    try:multiply_isg.main(argv)
    except StopProbe:pass
    finally:
        AlignmentAxial._scan_candidates=original_scan;AlignmentAxial.sample_instance=original_instance;SamplerV2.sample=original_sample;SamplerV2.write_instance=original_writer;og.shutdown=old_shutdown
        report['finished_at']=time.time();save()

if __name__=='__main__':main()
