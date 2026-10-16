"""Single-environment view of OmniGibson's vectorized ``Environment`` API.

``Environment.step`` / ``reset`` / ``get_obs`` return per-env lists even for one environment; these helpers return the
values of env 0 (results already in the single-env form pass through unchanged).
"""


def step(env, action, **kwargs):
    """``env.step`` -> ``(obs, reward, terminated, truncated, info)`` of env 0."""
    obs, reward, terminated, truncated, info = env.step(action, **kwargs)
    if isinstance(info, list):
        return obs[0], float(reward[0]), bool(terminated[0]), bool(truncated[0]), info[0]
    return obs, reward, terminated, truncated, info


def reset(env, **kwargs):
    """``env.reset`` -> ``(obs, {"obs_info": obs_info})`` of env 0."""
    obs, info = env.reset(**kwargs)
    if isinstance(obs, list):
        return obs[0], {"obs_info": info["obs_info"][0]}
    return obs, info


def get_obs(env):
    """``env.get_obs`` -> ``(obs, obs_info)`` of env 0."""
    obs, info = env.get_obs()
    if isinstance(obs, list):
        return obs[0], info[0]
    return obs, info
