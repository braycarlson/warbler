from functools import wraps
from pathlib import Path


# The current-working directory of the project
CWD = Path.cwd().parent.parent

# The drive of the project
ROOT = Path(CWD.drive).joinpath('/')

# The directory for the original dataset
DATASET = ROOT.joinpath('dataset/logue/adelaideswarbler')

# The directory for saving logs
LOGS = CWD.joinpath('logs')

# The directory for saving the processed dataset
DATA = CWD.joinpath('data')

# The directory for saving all segmented notes
NOTES = CWD.joinpath('notes')

# The root directory for the package
PACKAGE = CWD.joinpath('warbler.py')

# The directory for the cluster package
CLUSTER = PACKAGE.joinpath('cluster')

# The directory for the inspect package
INSPECT = PACKAGE.joinpath('inspect')

# The directory for generated spectrograms to be inspected
SPECTROGRAMS = INSPECT.joinpath('spectrograms')

# The directory for the noise package
NOISE = PACKAGE.joinpath('noise')

# The directory for the segment package
SEGMENT = PACKAGE.joinpath('segment')

# Get each individual's directory from the original dataset
DIRECTORIES = sorted([
    directory
    for directory in DATASET.glob('*/')
    if directory.is_dir()
])

# Get each individual's directory from the processed dataset
INDIVIDUALS = sorted([
    individual
    for individual in DATA.glob('*/')
    if individual.is_dir()
])


def bootstrap(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        LOGS.mkdir(parents=True, exist_ok=True)

        DATA.mkdir(parents=True, exist_ok=True)

        NOTES.mkdir(parents=True, exist_ok=True)

        SPECTROGRAMS.mkdir(parents=True, exist_ok=True)

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

            # Create a folder to save segmented notes
            individual.joinpath('notes').mkdir(parents=True, exist_ok=True)

            # Create a directory for the good/bad waveforms/spectrogram(s)
            threshold = individual.joinpath('threshold')
            threshold.mkdir(parents=True, exist_ok=True)

            threshold.joinpath('good').mkdir(parents=True, exist_ok=True)
            threshold.joinpath('bad').mkdir(parents=True, exist_ok=True)

        return func

    return wrapper()
