"""Files of the campaign that minyeong cannot read (shiyuc-written, mode 600: 953 of the eval-1000 inputs on 2026-11-15) are read
from the shiyuc-made mirror /vision/u/shiyuc/minyeong/isgrend_mirror/<absolute path> (copies, mode 644; the originals are never
touched). Packing keeps the original absolute path as the archive name, so the replay sees the original paths."""
import os

MIRROR = os.environ.get("ISGREND_MIRROR", "/vision/u/shiyuc/minyeong/isgrend_mirror")


def readable(path: str) -> str:
    if path is None or os.access(path, os.R_OK):
        return path
    m = MIRROR + os.path.abspath(path)
    if os.access(m, os.R_OK):
        return m
    raise PermissionError(f"{path}: unreadable and not mirrored (as shiyuc: cp -p it to {m} and chmod 644)")
