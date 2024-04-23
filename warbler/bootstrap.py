from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING
from warbler.constant import DATASET, DIRECTORIES

if TYPE_CHECKING:
    from collections.abc import Callable


def bootstrap(func: Callable) -> Callable:
    @wraps(func)
    def wrapper() -> Callable:
        for directory in DIRECTORIES:
            individual = DATASET.joinpath(directory.stem)
            individual.mkdir(parents=True, exist_ok=True)

            recordings = individual.joinpath('recordings')
            segmentation = individual.joinpath('segmentation')

            recordings.mkdir(parents=True, exist_ok=True)
            segmentation.mkdir(parents=True, exist_ok=True)

        return func

    return wrapper()
