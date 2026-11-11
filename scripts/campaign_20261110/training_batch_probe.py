"""Execute an observational A/B/A probe through existing native production paths.

Inputs are separate validated canaries. Outputs never allocate benchmark quota.
A modern CPU helper must re-audit the resulting evidence before any release.
"""
import argparse
import ast
import json
import os
from pathlib import Path
from scripts.campaign_20261110.contracts import file_digest,require,write_json_atomic


def verify_runtime_compatibility(root, request, records, runtime):
    """Recheck the narrow observation-only runtime against native source bytes."""
    root=Path(root).resolve();runtime=Path(runtime).resolve()
    ref=request.get('runtime_compatibility') or {}
    proof_path=Path(ref.get('path','')).resolve()
    require(proof_path.is_relative_to(root) and proof_path.is_file() and
            file_digest(proof_path)==ref.get('sha256'),'missing exact observational runtime proof')
    proof=json.loads(proof_path.read_text())
    require(proof.get('schema')=='isg-observational-runtime-compatibility-v1', 'invalid observational runtime proof')
    base=root/'code'/('trunk_'+proof['base_commit'][:12])
    require(runtime.is_relative_to(root/'code') and
            (runtime/'.snapshot_commit').read_text().strip()==proof['observational_commit'] and
            (base/'.snapshot_commit').read_text().strip()==proof['base_commit'], 'observational runtime identity changed')
    require(all(r['validation_key']['code_sha256']==file_digest(base/'.snapshot_commit') for r in records),
            'observational runtime does not preserve these canaries native runtime')
    allowed={'ispatialgym/sampling/campaign_runtime.py','ispatialgym/multienv/campaign_rendering.py',
             'scripts/campaign_20261110/training_batch_probe.py'}
    require(set(proof['changed_files'])==allowed and set(proof['changed_payloads'])==allowed,
            'observational runtime changes exceed the approved instrumentation')
    changed=set()
    for folder in ('ispatialgym','scripts'):
        paths={str(p.relative_to(tree)) for tree in (base,runtime) for p in (tree/folder).rglob('*.py')}
        for relative in paths:
            a,b=base/relative,runtime/relative
            if not a.is_file() or not b.is_file() or file_digest(a)!=file_digest(b):changed.add(relative)
    require(changed==allowed,'native implementation changed outside observation-only files')
    for relative in allowed:
        require(file_digest(runtime/relative)==proof['changed_payloads'][relative],'observational payload changed')
    def physical_ast(path):
        tree=ast.parse(path.read_text())
        tree.body=[node for node in tree.body if not
            (isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)) and node.name in ('verify_loaded','restoration_observation'))
            and not (isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and
                t.id.startswith('_RESTORATION_PROBE_') for t in node.targets))]
        return ast.dump(tree,include_attributes=False)
    require(physical_ast(base/'ispatialgym/sampling/campaign_runtime.py')==
            physical_ast(runtime/'ispatialgym/sampling/campaign_runtime.py'),
            'physical campaign runtime changed alongside instrumentation')
    return {'path':str(proof_path),'sha256':ref['sha256']}


def inputs(root,request):
    root=Path(root).resolve();value=json.loads(Path(request).read_text())
    require(value['schema']=='isg-training-batch-probe-request-v1' and value['campaign_id']=='campaign_20261110','invalid native probe request')
    records=[]
    for ref in value['canaries']:
        path=Path(ref['path']).resolve()
        require(path.is_relative_to(root) and file_digest(path)==ref['sha256'],'canary review changed')
        review=json.loads(path.read_text());require(review['validation_passed'] and not review['accepted'],'input not a validation-only canary')
        row=review['export_record']
        for artifact in row['artifacts'].values():
            path=Path(artifact['path']).resolve()
            require(path.is_relative_to(root) and file_digest(path)==artifact['sha256'],'probe input changed')
        records.append(row)
    require(len(records)==2 and records[0]['instance_id']!=records[1]['instance_id'],'probe needs two distinct canaries')
    require(records[0]['render_compatibility_sha256']==records[1]['render_compatibility_sha256'],
            'probe cannot bypass production configuration isolation')
    return value,records


def shared_world_proof(runner, records):
    """Use the production signature and saved-scene checks for template copies."""
    paths=[r['artifacts']['tro']['path'] for r in records]
    signatures=[runner.instance_signature(path) for path in paths]
    require(signatures[0]==signatures[1],'production world signatures differ')
    directories=[runner.resolve_instance(path)[1] for path in paths]
    compatible,reason=runner.template_compatible(directories[0],paths[0],directories[1],paths[1])
    require(compatible,'production saved scenes differ: '+reason)
    return {'signature':signatures[0],'compatible_template_dirs':directories,
            'scene_check':reason,'method':'native instance_signature + template_compatible'}


def run(root,request,output,kind):
    root=Path(root).resolve();output=Path(output).resolve()
    require(output.is_relative_to(root) and not output.exists(),'probe output must be new and private')
    value,records=inputs(root,request);output.mkdir(parents=True)
    runtime=Path(__file__).resolve().parents[2]/'.snapshot_commit'
    require(file_digest(runtime)==value['probe_runtime_sha256'],'probe runtime differs from approved observational snapshot')
    compatibility=verify_runtime_compatibility(root,value,records,runtime.parent)
    os.environ['ISPATIALGYM_CAMPAIGN_RESTORATION_PROBE']='1'
    order=[0,1,0];entries=[]
    common={'schema':'isg-training-restoration-probe-v1','campaign_id':'campaign_20261110','kind':kind,
            'request':{'path':str(Path(request).resolve()),'sha256':file_digest(request)},
            'probe_runtime':{'path':str(runtime),'sha256':file_digest(runtime)},'entries':entries,
            'runtime_compatibility':compatibility,
            'accepted':False,'eligible_for_goal':False,'quota_consumed':0}
    if kind=='physics_action_open_loop':
        # A and B are immutable copies of one exact configuration. Production
        # prebuild checks compare their content and recorded scene, not paths.
        os.environ['ISPATIALGYM_FAST_COLLECTOR_PREBUILD']='1'
        from ispatialgym.oracle import run_isg
        from ispatialgym.fastflags import fast_on
        require(fast_on('COLLECTOR_PREBUILD'),'production template compatibility is disabled')
        world=shared_world_proof(run_isg,records)
        common['world_compatibility']=world
        shared={'signature':world['signature'],
                'compatible_template_dirs':set(map(os.path.abspath,world['compatible_template_dirs']))}
        for ordinal,index in enumerate(order):
            row=records[index];destination=output/('episode_'+str(ordinal))
            argv=['--instance',row['artifacts']['tro']['path'],'--mode','replay',
                  '--replay',row['artifacts']['demo']['path'],'--record','hdf5','--no-goal-override','--out',str(destination)]
            args=run_isg.parse_args(argv);code,result=run_isg.run_instance(args,argv[1],shared)
            report=destination/row['instance_id']/'meta.json'
            entries.append({'canary_review':value['canaries'][index],'returncode':code,'result':result,
                            'report':{'path':str(report),'sha256':file_digest(report)}})
            write_json_atomic(output/'probe.json',common)
            require(code==0 and result.get('success') is True,'native batch physics episode failed')
    else:
        require(kind=='kinematic_observation_rendering','unknown native probe kind')
        from ispatialgym.multienv import replay
        manifest=output/'canaries.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n'for r in records))
        listing=output/'demos.txt';listing.write_text(''.join(str(Path(records[i]['artifacts']['demo']['path']).parent)+'\n'for i in order))
        results=output/'results.jsonl'
        code=replay.main(['--demos','@'+str(listing),'--campaign-manifest',str(manifest),
            '--root',str(output/'rendered'),'--task',records[0]['variant'],'--slots','1',
            '--scene','recorded','--results',str(results),'--repo-prefix','training_probe'])
        rows=[(i,json.loads(line))for i,line in enumerate(results.read_text().splitlines())]
        episodes=[(i,row)for i,row in rows if row.get('tag')=='REPLAY_DEMO']
        outputs=[{'path':str(p),'sha256':file_digest(p)}for p in sorted((output/'rendered').rglob('*'))if p.is_file()]
        require(code==0 and len(episodes)==3 and outputs,'native renderer did not produce three episodes and files')
        for ordinal,(line,row)in enumerate(episodes):
            require(Path(row['demo']).name==records[order[ordinal]]['instance_id'],'render order differs')
            entries.append({'canary_review':value['canaries'][order[ordinal]],
                'report':{'path':str(results),'sha256':file_digest(results),'jsonl_index':line},'outputs':outputs})
        write_json_atomic(output/'probe.json',common)
    # Recheck all source bytes after running; no source state or goal rewrites.
    inputs(root,request)
    return common


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('root','request','output'):p.add_argument('--'+name,required=True)
    p.add_argument('--kind',required=True,choices=('physics_action_open_loop','kinematic_observation_rendering'))
    a=p.parse_args()
    try:print(json.dumps(run(a.root,a.request,a.output,a.kind)),flush=True)
    except BaseException as error:
        # OG shutdown may terminate the interpreter before Python prints a
        # traceback. Preserve the actual diagnostic failure first.
        import traceback
        destination=Path(a.output)
        if destination.is_dir():
            write_json_atomic(destination/'failure.json',{'schema':'isg-training-batch-probe-failure-v1',
                'kind':a.kind,'error':repr(error),'traceback':traceback.format_exc(),
                'accepted':False,'eligible_for_goal':False,'quota_consumed':0})
        raise
    finally:
        import sys
        og=sys.modules.get('omnigibson')
        if og is not None and hasattr(og,'shutdown'):og.shutdown()
