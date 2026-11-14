"""Authorize the unchanged canonical lid retreat for one exact pipeline source."""
import json
import os
from dataclasses import asdict
from pathlib import Path

from .contracts import digest, file_digest, require


def authorized(template, *, environment=None):
    environment = os.environ if environment is None else environment
    scope = template.scene_setup.get('bounded_lid_pipeline_recovery')
    if template.split != 'train' or scope is None:
        return False
    root = Path(environment['ISGF_ROOT']).resolve()
    config_path = Path(environment['ISPATIALGYM_CAMPAIGN_CONFIG']).resolve()
    require(config_path.is_relative_to(root), 'lid pipeline configuration outside campaign')
    config = json.loads(config_path.read_text())
    require(scope.get('schema') == 'isg-exact-canonical-lid-pipeline-recovery-v1'
            and scope.get('production_released') is False
            and type(scope.get('benchmark_credit')) is int and scope['benchmark_credit'] == 0
            and type(scope.get('start')) is int and scope['start'] > 0
            and type(scope.get('count')) is int and scope['count'] == 1,
            'lid pipeline recovery must be one explicit zero-credit source')
    require(config.get('purpose') == 'training_production' and config.get('split') == 'train'
            and digest(config['template']) == digest(asdict(template))
            and template.scene_setup.get('source_alias') ==
                'configuration_matching-attachment-lid-stockpot-desk',
            'lid pipeline recovery configuration or original variant differs')
    source_ref = scope['canonical_configuration']
    source_path = Path(source_ref['path']).resolve()
    require(source_path.is_relative_to(root) and file_digest(source_path) == source_ref['sha256'],
            'lid pipeline canonical source changed')
    source = json.loads(source_path.read_text())['configuration']['template']
    require(source['split'] == 'eval_standard'
            and source['class_id'] == template.class_id == 'configuration_matching-attachment_04'
            and source['task'] == template.task == 'configuration_matching-attachment'
            and digest(source['objects']) == digest([asdict(o) for o in template.objects]),
            'lid pipeline source model, scale, asset or task differs from canonical')
    partner = [o for o in template.objects if o.role == 'partner']
    require(len(partner) == 1 and (partner[0].category, partner[0].model) == ('stockpot', 'oshwps')
            and len(partner[0].default_scale) == 3
            and all(abs(float(v) - .65) <= 1e-7 for v in partner[0].default_scale),
            'lid pipeline recovery belongs to its exact canonical stockpot')
    for obj in template.objects:
        scale = config['perturbations']['scales'][obj.instance]
        require(scale['mode'] == 'fixed' and scale['applied'] == scale['default']
                and digest(scale['applied']) == digest(obj.default_scale),
                'lid pipeline recovery does not authorize changed scale')
    from .production_gate import check_environment
    decision = check_environment(root, environment, kind='source_validation',
                                 start=scope['start'], count=scope['count'])
    require(decision.get('allowed') is True and decision.get('category') == 'validation_canary'
            and decision.get('config_sha256') == file_digest(config_path)
            and decision.get('authorization', {}).get('purpose') == 'training_pipeline_canary',
            'lid pipeline recovery lacks exact bounded source authorization')
    return True
