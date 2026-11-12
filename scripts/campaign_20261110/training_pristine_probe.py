"""Explicit physical diagnostic on immutable validated inputs, never release proof."""
import ast
import copy
import json
from pathlib import Path
from .contracts import digest,file_digest,require

RESTORE_OPTIONS = {'pristine_mass_before_reset_v1': True, 'reuse_native_lifecycle_v1': True,
                   'complete_scene_snapshot_v1': True}

ALLOWED = {'ispatialgym/oracle/run_isg.py','ispatialgym/oracle/campaign_restore.py',
           'ispatialgym/sampling/campaign_runtime.py','ispatialgym/multienv/campaign_rendering.py',
           'scripts/campaign_20261110/training_batch_probe.py',
           'scripts/campaign_20261110/training_pristine_probe.py'}


def verify_candidate(root, request, records, runtime):
    root=Path(root).resolve();runtime=Path(runtime).resolve();ref=request['physical_candidate']
    path=Path(ref['path']).resolve()
    require(path.is_relative_to(root) and file_digest(path)==ref['sha256'],'physical diagnostic proof changed')
    proof=json.loads(path.read_text())
    require(request.get('diagnostic_only') is True and request.get('production_released') is False
            and proof.get('schema') in ('isg-pristine-mass-diagnostic-runtime-v1',
                                        'isg-pristine-mass-diagnostic-runtime-v2')
            and proof.get('production_released') is False and proof.get('source_files_unchanged') is True,
            'physical diagnostic cannot become observational release evidence')
    base=root/'code'/('trunk_'+proof['source_commit'][:12])
    require((base/'.snapshot_commit').read_text().strip()==proof['source_commit']
            and (runtime/'.snapshot_commit').read_text().strip()==proof['candidate_commit'],
            'physical diagnostic runtime differs')
    require(all(r['validation_key']['code_sha256']==file_digest(base/'.snapshot_commit')for r in records),
            'physical diagnostic source runtime differs')
    allowed = ALLOWED
    if proof['schema'] == 'isg-pristine-mass-diagnostic-runtime-v2':
        implementation = proof['production_restoration_runtime']
        stamp = Path(implementation['path']).resolve()
        require(stamp.is_relative_to(root/'code') and stamp.name == '.snapshot_commit'
                and file_digest(stamp) == implementation['sha256'],
                'production restoration runtime binding changed')
        allowed = ALLOWED | {'scripts/campaign_20261110/training_canary_scope.py'}
        for relative in ('ispatialgym/oracle/run_isg.py', 'ispatialgym/oracle/campaign_restore.py',
                         'scripts/campaign_20261110/training_canary_scope.py'):
            require(file_digest(runtime/relative) == file_digest(stamp.parent/relative),
                    'diagnostic did not execute the exact production restoration implementation')
        require(request.get('diagnostic_restore_options') == proof.get('diagnostic_restore_options') == RESTORE_OPTIONS,
                'diagnostic restoration options differ')
    else:
        require(not request.get('diagnostic_restore_options'), 'v1 diagnostic cannot add restoration options')
    require(set(proof['changed_payloads']) == allowed, 'physical diagnostic change scope differs')
    changed=set()
    for folder in ('ispatialgym','scripts'):
        paths={str(p.relative_to(tree))for tree in(base,runtime)for p in(tree/folder).rglob('*.py')}
        for relative in paths:
            a,b=base/relative,runtime/relative
            if not a.is_file()or not b.is_file()or file_digest(a)!=file_digest(b):changed.add(relative)
    require(changed == allowed, 'physical diagnostic includes unrelated implementation changes')
    for relative,sha in proof['changed_payloads'].items():
        require(file_digest(runtime/relative)==sha,'physical diagnostic payload changed')
    def physical_ast(path):
        tree=ast.parse(path.read_text());tree.body=[n for n in tree.body if not
            (isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))and n.name in('verify_loaded','restoration_observation'))
            and not(isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id.startswith('_RESTORATION_PROBE_')for t in n.targets))]
        return ast.dump(tree,include_attributes=False)
    require(physical_ast(base/'ispatialgym/sampling/campaign_runtime.py')==physical_ast(runtime/'ispatialgym/sampling/campaign_runtime.py'),
            'physical campaign settings changed alongside pristine diagnostic')
    return {'path':str(path),'sha256':ref['sha256'],'kind':'explicit_physical_diagnostic','production_released':False}


def install_override(runner, request=None):
    """Change only the named in-memory option; never rewrite source files/keys."""
    options = (request or {}).get('diagnostic_restore_options')
    require(options is None or options == RESTORE_OPTIONS, 'unsupported diagnostic restoration options')
    overrides = dict(options or {'pristine_mass_before_reset_v1': True})
    initialize=runner.InstanceLoader.__init__
    def controlled(loader,*args,**kwargs):
        initialize(loader,*args,**kwargs)
        campaign=loader.tro.meta.raw['campaign'];source=copy.deepcopy(campaign)
        require(campaign['campaign_id']=='campaign_20261110' and campaign['split']=='train'
                and campaign['class_id']=='disembedding_07' and campaign['task']=='disembedding'
                and campaign['runtime']['purpose']=='training_pipeline_canary', 'physical diagnostic source scope differs')
        options=campaign.get('scene_setup',{}).get('restoration_options',{})
        require(options.get('serialized_root_motion_v1') is True and not options.get('serialized_mass_before_reset_v1')
                and not options.get('pristine_mass_before_reset_v1'), 'physical diagnostic requires the original fde input')
        changed=copy.deepcopy(campaign)
        changed.setdefault('scene_setup',{}).setdefault('restoration_options',{}).update(overrides)
        loader.tro.meta.raw['campaign']=changed
        loader.report['physical_diagnostic_override']={'option':'pristine_mass_before_reset_v1','value':True,
            'effective_option_overrides':overrides,'source_campaign_sha256':digest(source),'effective_campaign_sha256':digest(changed),
            'source_tro':{'path':loader.tro_path,'sha256':file_digest(loader.tro_path)},
            'source_files_unchanged':True,'source_validation_key_unchanged':True,'production_released':False}
    runner.InstanceLoader.__init__=controlled
    return initialize
