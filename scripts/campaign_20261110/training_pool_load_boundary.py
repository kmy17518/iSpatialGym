"""Exact source transformation for the explicit incoming pool-state boundary.

Attachment cleanup's native snapshot loads remain untouched. Only the final
incoming scene.load_state call gets a named loader method; this is not a reset,
solver reconstruction, timestep change or extra physics operation.
"""
from pathlib import Path
from .contracts import require

PAYLOAD = 'ispatialgym/oracle/run_isg.py'
DELTA = {'name':'explicit_incoming_pool_state_boundary_v1',
         'operation':'extract_existing_final_scene_load_into_loader_method',
         'cleanup_snapshot_loads':'unchanged_native_operations_outside_pool_scope',
         'incoming_pool_transition':'once_after_completed_attachment_cleanup',
         'native_load_arguments_changed':False,'extra_physics_steps':0,
         'thresholds_changed':False,'reference_state_copied':False}
CALL = '        self.env.scene.load_state(scene_state, serialized=False)\n'
MARKER = '    # -- the loader contract --------------------------------------------------------------------------------------\n'
METHOD = '''    def _load_reused_scene_state(self, scene_state):
        """Load the incoming snapshot after attachment cleanup has finished.

        Cleanup may independently restore its own native snapshot. Keep that
        lifecycle separate from the single incoming episode state boundary.
        """
        return self.env.scene.load_state(scene_state, serialized=False)

'''


def source_with_boundary(source):
    require(source.count(CALL)==1 and source.count(MARKER)==1 and
            'def _load_reused_scene_state(' not in source, 'native incoming load boundary is ambiguous')
    return source.replace(CALL,'        self._load_reused_scene_state(scene_state)\n',1).replace(MARKER,METHOD+MARKER,1)


def verify_runtime(request, proof, base, runtime):
    require(request.get('incoming_pool_load_boundary_v1') is True and
            request.get('incoming_pool_load_boundary') == proof.get('incoming_pool_load_boundary') == DELTA,
            'incoming pool load boundary requires its exact operation declaration')
    require((Path(runtime)/PAYLOAD).read_text()==source_with_boundary((Path(base)/PAYLOAD).read_text()),
            'native loader changed beyond exact incoming-state method extraction')
    return {PAYLOAD}
