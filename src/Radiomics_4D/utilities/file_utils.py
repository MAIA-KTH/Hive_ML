import os
from pathlib import Path

from .log_utils import get_logger

logger = get_logger(__name__)


def subfiles(folder, join=True, prefix=None, suffix=None, sort=True):
    if join:
        l = os.path.join  # noqa: E741
    else:
        l = lambda x, y: y  # noqa: E741, E731
    res = [
        l(folder, i.name)
        for i in Path(folder).iterdir()
        if i.is_file() and (prefix is None or i.name.startswith(prefix)) and (suffix is None or i.name.endswith(suffix))
    ]
    if sort:
        res.sort()
    return res


def subfolders(folder, join=True, sort=True):
    if join:
        l = os.path.join  # noqa: E741
    else:
        l = lambda x, y: y  # noqa: E741, E731
    res = [l(folder, i.name) for i in Path(folder).iterdir() if i.is_dir()]
    if sort:
        res.sort()
    return res
