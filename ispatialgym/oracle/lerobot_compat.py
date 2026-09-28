"""BEHAVIOR's LeRobot writer against the installed lerobot (wensi-ai/lerobot release/b1k, 0.5.2).

``omnigibson.envs.lerobot_data_wrapper.LeRobotDataWrapper.create_dataset`` (BEHAVIOR-1K main) adds two keys to the
dataset info, ``meta.info["cam_intrinsics"]`` and ``meta.info["omnigibson_git_hash"]``, then ``write_info``s it. That lerobot's
``DatasetInfo`` is a dataclass: an unknown key raises ``KeyError`` on assignment, is not written to ``info.json`` and is
dropped when a dataset is loaded (resumed). :func:`install` lets ``DatasetInfo`` carry extra keys through all three, so
BEHAVIOR's writer runs unmodified and ``info.json`` holds the two keys it adds. Every process that writes, resumes or
aggregates these datasets (collection, replay_obs_isg.py, the aggregation script) installs it.
"""

from __future__ import annotations

import dataclasses


def install() -> None:
    from lerobot.datasets.utils import DatasetInfo

    if getattr(DatasetInfo, "_isg_extra_keys", False):
        return
    known = {f.name for f in dataclasses.fields(DatasetInfo)}
    setitem, getitem, to_dict, from_dict = DatasetInfo.__setitem__, DatasetInfo.__getitem__, DatasetInfo.to_dict, DatasetInfo.from_dict.__func__

    def __setitem__(self, key, value):
        if key in known:
            return setitem(self, key, value)
        self.__dict__.setdefault("_extra", {})[key] = value

    def __getitem__(self, key):
        extra = self.__dict__.get("_extra", {})
        return extra[key] if key in extra and key not in known else getitem(self, key)

    def _to_dict(self):
        d = to_dict(self)
        d.update(self.__dict__.get("_extra", {}))
        return d

    def _from_dict(cls, data):
        info = from_dict(cls, data)
        info.__dict__["_extra"] = {k: v for k, v in data.items() if k not in known}
        return info

    DatasetInfo.__setitem__ = __setitem__
    DatasetInfo.__getitem__ = __getitem__
    DatasetInfo.to_dict = _to_dict
    DatasetInfo.from_dict = classmethod(_from_dict)
    DatasetInfo._isg_extra_keys = True


__all__ = ["install"]
