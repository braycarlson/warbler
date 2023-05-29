"""
Bootstrap
---------

"""

from __future__ import annotations

from constant import (
    DATASET,
    DIRECTORIES,
    LOGS,
    INPUT,
    OUTPUT,
    PDF,
    PICKLE
)
from functools import wraps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


def bootstrap(func: Callable) -> Callable:
    """A decorator that creates paths before executing the decorated function.

    Args:
        func: The function to be decorated.

    Returns:
        The decorated function.

    """

    @wraps(func)
    def wrapper() -> Callable:
        INPUT.mkdir(parents=True, exist_ok=True)
        OUTPUT.mkdir(parents=True, exist_ok=True)

        DATASET.mkdir(parents=True, exist_ok=True)
        LOGS.mkdir(parents=True, exist_ok=True)
        PDF.mkdir(parents=True, exist_ok=True)
        PICKLE.mkdir(parents=True, exist_ok=True)

        for directory in DIRECTORIES:
            individual = DATASET.joinpath(directory.stem)
            individual.mkdir(parents=True, exist_ok=True)

            recordings = individual.joinpath('recordings')
            segmentation = individual.joinpath('segmentation')

            recordings.mkdir(parents=True, exist_ok=True)
            segmentation.mkdir(parents=True, exist_ok=True)

        return func

    return wrapper()
