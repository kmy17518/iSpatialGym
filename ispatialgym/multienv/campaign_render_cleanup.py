"""Native writer cleanup at a bounded diagnostic failure boundary.

Ordinary failures remain errors. Only explicit diagnostic exception types can
continue to the next recorded episode, using the same existing writer/world.
"""


class ControlledRenderFailure(RuntimeError):
    pass


class ControlledRenderInterrupt(KeyboardInterrupt):
    pass


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
