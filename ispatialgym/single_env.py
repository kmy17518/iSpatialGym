"""Single-environment view of OmniGibson's vectorized ``Environment`` API.

Since the vector-env refactor (BEHAVIOR-1K #2329) ``Environment.step`` / ``reset`` / ``get_obs`` return per-env lists and
``(num_envs,)`` tensors even for one env. iSpatialGym drives one env per ``Environment`` through these helpers, which
return env 0's values with the pre-refactor types; results already in the single-env form (the unit tests' stand-in
environments) pass through unchanged.
"""


def step(env, action, **kwargs):
    """``env.step`` -> ``(obs, reward, terminated, truncated, info)`` of env 0 (``float`` reward, ``bool`` flags)."""
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
