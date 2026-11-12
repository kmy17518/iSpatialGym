"""Read-only exact dataset checks for completed model-pool render episodes.

Cold exports are independent native renders, not collection-time proprioception.
The native renderer/state/video proof remains a separate required witness.
"""
import json
from pathlib import Path

from .contracts import file_digest, require


def exact_episode_rows(actual, cold, source, index, offset):
    import numpy as np
    require(set(actual) == set(cold) and len(actual) == len(cold) > 0,
            'export episode schema or frame count differs')
    count = len(actual)
    for key, values in [('episode_index', np.full(count, index)),
                        ('index', np.arange(offset, offset + count)),
                        ('frame_index', np.arange(count))]:
        require(np.array_equal(np.asarray(actual[key]), values), 'export episode indexing differs: ' + key)
    checked = []
    for key in actual:
        if key in ('episode_index', 'index'): continue
        a, b = np.stack(actual[key].to_numpy()), np.stack(cold[key].to_numpy())
        require(a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b),
                'export differs from exact own cold data: ' + key)
        require(np.isfinite(a).all(), 'export contains nonfinite values: ' + key)
        checked.append(key)
    for native, column, dtype in [('action', 'action', np.float32), ('reward', 'next.reward', np.float32),
                                  ('terminated', 'next.terminated', bool), ('truncated', 'next.truncated', bool)]:
        a = np.stack(actual[column].to_numpy()); b = np.asarray(source[native], dtype=dtype)
        require(a.shape == b.shape and np.array_equal(a, b), 'export changed exact recorded ' + native)
    return {'frames':count, 'cold_exact_fields':checked, 'source_action_reward_done_exact':True}


def verify_dataset(dataset, records, cold_datasets):
    import h5py
    import numpy as np
    import pandas as pd
    dataset = Path(dataset)
    require(len(records) == len(cold_datasets) > 0, 'exact completed source order required')
    def ref(path): return {'path':str(path), 'sha256':file_digest(path)}
    def table(directory):
        paths = sorted(directory.rglob('*.parquet'))
        require(paths, 'native export table absent: ' + str(directory))
        return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True), paths
    info_path = dataset/'meta/info.json'; info = json.loads(info_path.read_text())
    data, data_files = table(dataset/'data'); episodes, episode_files = table(dataset/'meta/episodes')
    data = data.sort_values('index'); episodes = episodes.sort_values('episode_index')
    require(info['total_episodes'] == len(records) == len(episodes)
            and info['total_frames'] == len(data) and info['total_tasks'] == 1
            and np.array_equal(episodes.episode_index.to_numpy(), np.arange(len(records)))
            and np.array_equal(data['index'].to_numpy(), np.arange(len(data))),
            'native export counts or global indexes differ')
    variants = {r['variant'] for r in records}; require(len(variants) == 1, 'mixed exported task labels')
    tasks_path = dataset/'meta/tasks.parquet'; tasks = pd.read_parquet(tasks_path)
    require(tasks.index.tolist() == list(variants) and tasks.task_index.tolist() == [0],
            'native export task dictionary differs')
    video_keys = [key for key, value in info['features'].items() if value['dtype'] == 'video']
    require(len(video_keys) == 6, 'native six-camera-stream schema changed')
    observed = []; offset = 0; sources = []
    for index, (record, cold_path) in enumerate(zip(records, cold_datasets)):
        cold_path = Path(cold_path); cold_info = json.loads((cold_path/'meta/info.json').read_text())
        for key in ('features', 'fps', 'robot_type', 'cam_intrinsics', 'omnigibson_git_hash'):
            require(info[key] == cold_info[key], 'native export metadata differs from cold: ' + key)
        cold, cold_files = table(cold_path/'data'); cold = cold.sort_values('index')
        actual = data[data.episode_index == index]; ep = episodes.iloc[index]
        source = record['artifacts']['demo']; require(ref(source['path']) == source, 'source recording changed')
        with h5py.File(source['path'], 'r') as h:
            values = {key:h['data/demo_0'][key][()] for key in ('action','reward','terminated','truncated')}
        proof = exact_episode_rows(actual, cold, values, index, offset)
        count = proof['frames']
        require(int(ep['length']) == count and int(ep['dataset_from_index']) == offset
                and int(ep['dataset_to_index']) == offset + count and list(ep['tasks']) == [record['variant']],
                'native episode metadata or task differs')
        for key in video_keys:
            start = float(ep['videos/'+key+'/from_timestamp']); end = float(ep['videos/'+key+'/to_timestamp'])
            # Timeline metadata is serialized float64; allow only arithmetic roundoff.
            require(abs(start - offset/info['fps']) <= 1e-12
                    and abs(end - (offset+count)/info['fps']) <= 1e-12,
                    'native video episode offsets include missing or aborted frames')
        observed.append({'episode_index':index, 'instance_id':record['instance_id'],
                         'source_demo':source, 'cold_export_files':[ref(p) for p in cold_files], **proof})
        sources.append(source); offset += count
    require(offset == len(data), 'native export contains an unbound episode or aborted prefix')
    files = [info_path,tasks_path,*data_files,*episode_files]
    return {'status':'exact_source_and_cold_export_verified', 'dataset':str(dataset),
            'episodes':observed, 'total_frames':offset, 'files':[ref(p) for p in files],
            'source_recordings':sources, 'proprio_mode':'native_loaded_state',
            'collection_time_behavior_proprio_verified':False, 'production_released':False,
            'benchmark_credit':0, 'scope':'Exact actions/rewards/dones, loaded proprio, camera poses, timestamps, '
                'episode/task/schema/intrinsics and no extra aborted rows. Video decoding and native state proof separate.'}
