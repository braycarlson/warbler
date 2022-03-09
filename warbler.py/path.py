import pickle

from functools import wraps
from pathlib import Path


# The current-working directory of the project
CWD = Path.cwd().parent.parent

# The drive of the project
ROOT = Path(CWD.anchor)

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

# The directory for the analyze package
ANALYZE = PACKAGE.joinpath('analyze')

# The directory for the Luscinia printouts
PRINTOUT = ANALYZE.joinpath('printout')

# The directory for generated spectrograms to be inspected
SPECTROGRAM = ANALYZE.joinpath('spectrogram')

# The directory for spectrograms deemed good
GOOD = SPECTROGRAM.joinpath('good')

# The directory for spectrograms deemed mediocre
MEDIOCRE = SPECTROGRAM.joinpath('mediocre')

# The directory for spectrograms deemed bad
BAD = SPECTROGRAM.joinpath('bad')

# The directory for the noise package
NOISE = PACKAGE.joinpath('noise')

# The directory for the segment package
SEGMENT = PACKAGE.joinpath('segment')

# The directory for the custom parameters
PARAMETER = SEGMENT.joinpath('parameter')

# The directory for pickled songs
PICKLE = SEGMENT.joinpath('pickle')

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

# Get each individual's custom parameters
PARAMETERS = sorted([
    parameter.stem
    for parameter in PARAMETER.glob('*.json')
    if parameter.is_file()
])

# Get each individual's directory from the printouts
PRINTOUTS = sorted([
    individual
    for individual in PRINTOUT.glob('*/')
    if individual.is_dir()
])

IGNORE = set()

bad = PICKLE.joinpath('bad.pkl')

if bad.is_file():
    with open(bad, 'rb') as handle:
        ignore = pickle.load(handle)

        for file in ignore:
            IGNORE.add(
                Path(file['filename']).stem
            )

error = PICKLE.joinpath('error.pkl')

if error.is_file():
    with open(error, 'rb') as handle:
        ignore = pickle.load(handle)

        for file in ignore:
            IGNORE.add(file)


def bootstrap(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Data
        LOGS.mkdir(parents=True, exist_ok=True)
        DATA.mkdir(parents=True, exist_ok=True)
        NOTES.mkdir(parents=True, exist_ok=True)

        # Analyze
        PRINTOUT.mkdir(parents=True, exist_ok=True)
        SPECTROGRAM.mkdir(parents=True, exist_ok=True)
        GOOD.mkdir(parents=True, exist_ok=True)
        MEDIOCRE.mkdir(parents=True, exist_ok=True)
        BAD.mkdir(parents=True, exist_ok=True)

        # Segment
        PARAMETER.mkdir(parents=True, exist_ok=True)
        PICKLE.mkdir(parents=True, exist_ok=True)

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
