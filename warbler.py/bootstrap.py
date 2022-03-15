from functools import wraps
from path import (
    DIRECTORIES,
    LOGS,
    DATA,
    INDIVIDUALS,
    NOTES,
    PICKLE,
    PRINTOUT,
    SPECTROGRAM,
)


def bootstrap(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Data
        LOGS.mkdir(parents=True, exist_ok=True)
        DATA.mkdir(parents=True, exist_ok=True)
        NOTES.mkdir(parents=True, exist_ok=True)

        # Analyze
        PRINTOUT.mkdir(parents=True, exist_ok=True)

        # Segment
        PICKLE.mkdir(parents=True, exist_ok=True)

        SPECTROGRAM.mkdir(parents=True, exist_ok=True)
        SPECTROGRAM.joinpath('good').mkdir(parents=True, exist_ok=True)
        SPECTROGRAM.joinpath('mediocre').mkdir(parents=True, exist_ok=True)
        SPECTROGRAM.joinpath('bad').mkdir(parents=True, exist_ok=True)

        for directory in DIRECTORIES:
            (
                DATA
                .joinpath(directory.stem)
                .mkdir(parents=True, exist_ok=True)
            )

        for individual in INDIVIDUALS:
            # Create a folder to save converted .wav files
            individual.joinpath('wav').mkdir(parents=True, exist_ok=True)

            # Create a folder to save metadata
            individual.joinpath('json').mkdir(parents=True, exist_ok=True)

            # Create a folder to save baseline/custom parameters
            individual.joinpath('parameter').mkdir(parents=True, exist_ok=True)

            # Create a folder to save segmented notes
            individual.joinpath('notes').mkdir(parents=True, exist_ok=True)

        return func

    return wrapper()
