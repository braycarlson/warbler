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


def bootstrap(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
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
