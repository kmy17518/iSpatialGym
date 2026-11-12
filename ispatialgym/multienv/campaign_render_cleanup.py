"""Native writer cleanup at a bounded diagnostic failure boundary.

Ordinary failures remain errors. Only explicit diagnostic exception types can
continue to the next recorded episode, using the same existing writer/world.
"""
from contextlib import contextmanager


class ControlledRenderFailure(RuntimeError):
    pass


class ControlledRenderInterrupt(KeyboardInterrupt):
    pass


@contextmanager
def processing_cleanup(writers):
    """Abort every writer on any unexpected episode-processing exception."""
    try:
        yield
    except BaseException as error:
        cleanup_errors=[]
        for writer in writers:
            try:writer.abort()
            except BaseException as failure:cleanup_errors.append(repr(failure))
        for writer in writers:
            try:writer.close()
            except BaseException as failure:cleanup_errors.append(repr(failure))
        if cleanup_errors and hasattr(error,'add_note'):
            error.add_note('Renderer cleanup errors: '+repr(cleanup_errors))
        raise


def abort_after_render(writer,index,fault):
    """Inject only after an existing render; use the unchanged native abort."""
    cls={'failure':ControlledRenderFailure,'interruption':ControlledRenderInterrupt}[fault]
    try:raise cls('explicit diagnostic after existing render')
    except BaseException as error:
        writer.abort()
        pending=bool(writer.dataset is not None and writer.dataset.has_pending_frames())
        if writer.current_traj_history or pending or not writer.aborted:
            raise RuntimeError('native renderer abort did not clear episode buffers') from error
        return error,{'exception_type':type(error).__name__,'error':repr(error),'frames_observed':index,
            'native_writer_abort_called':True,'trajectory_buffer_empty':True,'dataset_pending_frames':False,
            'writer_aborted':True,'physics_steps_added':0,'render_ticks_added':0,
            'expected_controlled_failure':True,'boundary':'after_existing_render'}


def write_frame(writer, demo, index, observation, *, fault=None):
    """Record one unchanged frame; abort native buffers if that operation fails."""
    try:
        if index == 0:
            writer.begin(observation)
        else:
            writer.step(demo.action[index-1], observation, float(demo.reward[index-1]),
                        bool(demo.terminated[index-1]), bool(demo.truncated[index-1]))
        if fault and index == 1:
            cls={'failure':ControlledRenderFailure,'interruption':ControlledRenderInterrupt}[fault]
            raise cls('explicit diagnostic after one recorded native action')
    except BaseException as error:
        writer.abort()
        pending=bool(writer.dataset is not None and writer.dataset.has_pending_frames())
        proof={'exception_type':type(error).__name__,'error':repr(error),
               'boundary':'after_native_writer_action',
               'frames_observed':index+1,'native_writer_abort_called':True,
               'trajectory_buffer_empty':not writer.current_traj_history,
               'dataset_pending_frames':pending,'writer_aborted':writer.aborted,
               'physics_steps_added':0,'render_ticks_added':0,
               'expected_controlled_failure':type(error) is {'failure':ControlledRenderFailure,
                   'interruption':ControlledRenderInterrupt}.get(fault)}
        if not proof['trajectory_buffer_empty']or pending or not writer.aborted:
            raise RuntimeError('native renderer abort did not clear episode buffers') from error
        return error,proof
    return None,None
