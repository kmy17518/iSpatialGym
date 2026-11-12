"""Replay transport must never discover alternate collection inputs."""
import ast
from pathlib import Path
import pytest
from scripts.campaign_20261110.infra_vast.campaign_inputs import collector_prebuild_patterns


def test_exact_replay_needs_no_sampler_template_identity():
    assert collector_prebuild_patterns({'ALIAS_KEY':'object_scaling-bottle_of_pop'},
        replay=True,primary_root='/campaign',legacy_root='/legacy')==[]


def test_replay_does_not_discover_alternate_prebuild_even_if_identity_exists():
    assert collector_prebuild_patterns({'ALIAS_KEY':'class','TEMPLATE_ID':'template','ISGF_ROOT':'/other'},
        replay=True,primary_root='/campaign',legacy_root='/legacy')==[]


def test_collection_retains_original_patterns_and_required_identity():
    env={'ALIAS_KEY':'class','TEMPLATE_ID':'template','ISGF_ROOT':'/other'}
    assert collector_prebuild_patterns(env,replay=False,primary_root='/campaign',legacy_root='/legacy')==[
        '/other/out/*/class/g*/inst/*/template','/campaign/out/*/class/g*/inst/*/template',
        '/legacy/out/class/g*/inst/*/template']
    with pytest.raises(KeyError,match='TEMPLATE_ID'):
        collector_prebuild_patterns({'ALIAS_KEY':'class'},replay=False,primary_root='/campaign',legacy_root='/legacy')


def test_actual_stage_replay_skips_collection_discovery_with_exact_closure_retained():
    """Execute the real staging discovery block, not a mirrored implementation."""
    source=Path(__file__).parents[1]/'infra_vast/vastctl.py'
    tree=ast.parse(source.read_text());stage=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=='stage')
    begin=next(i for i,n in enumerate(stage.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='pats'for t in n.targets))
    end=next(i for i,n in enumerate(stage.body[begin:],begin)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='lst'for t in n.targets))
    class ForbiddenGlob:
        def glob(self,*args):raise AssertionError('replay searched unrelated collection history')
    env={'ALIAS_KEY':'class','FAMILY':'camera_relocalization'}
    scope={'env':env,'replay':True,'F':'/campaign','T85':'/legacy','jd':'/exact',
        'collector_prebuild_patterns':collector_prebuild_patterns,'glob':ForbiddenGlob()}
    exec(compile(ast.Module(body=stage.body[begin:end],type_ignores=[]),str(source),'exec'),scope)
    assert scope['n_pre']==0
    assert any(isinstance(n,ast.Call)and isinstance(n.func,ast.Name)and n.func.id=='transport_inputs'for n in ast.walk(stage))
