import pathlib
import pickle

from natsort import os_sorted
from pathlib import Path, PurePath


def relative(path: str) -> pathlib.PurePath:
    """Calculates the relative path from the current working directory to the given path.

    Args:
        path: The path for which the relative path needs to be calculated.

    Returns:
        A PurePath object representing the relative path from the current working directory.

    """

    pure = PurePath(path)
    return pure.relative_to(CWD)


# Find the "venv" folder relative to file
def walk(file: pathlib.Path):
    """Recursively finds the 'venv' folder relative to the given file.

    Args:
        file: The file from which the 'venv' folder needs to be found.

    Returns:
        The parent directory of the 'venv' folder if found, else None.

    """

    for path in file.parents:
        if path.is_dir():
            venv = list(
                path.glob('venv')
            )

            for environment in venv:
                return environment.parent

            walk(path.parent)

    return None


file = Path.cwd()
CWD = walk(file)

# The drive of the project
ROOT = Path(CWD.anchor)

# The directory for the original dataset
ORIGINAL = ROOT.joinpath('dataset/logue/adelaideswarbler')

# The directory for saving logs
LOGS = CWD.joinpath('logs')

# The directory for input
INPUT = CWD.joinpath('input')

# The directory for converted recordings
DATASET = INPUT.joinpath('dataset')

# The directory for Luscinia printouts
LUSCINIA = INPUT.joinpath('printout')

# The directory for Luscinia printouts
SPREADSHEET = INPUT.joinpath('spreadsheet')

# The directory for output
OUTPUT = CWD.joinpath('output')

# The directory for the .pdf files
PDF = OUTPUT.joinpath('pdf')

# The directory for pickled and compressed output
PICKLE = OUTPUT.joinpath('pickle')

# The directory for projection(s)
PROJECTION = OUTPUT.joinpath('projection')

# The directory for generated spectrograms to be inspected
SPECTROGRAM = OUTPUT.joinpath('spectrogram')

# The directory for hyperparameter tuning
TUNING = OUTPUT.joinpath('tuning')

# The root directory for the package
PACKAGE = CWD.joinpath('warbler.py')

# The directory for the analysis package
ANALYSIS = PACKAGE.joinpath('analysis')

# The directory for the AVGN package
AVGN = PACKAGE.joinpath('avgn')

# The directory for the pipeline package
PIPELINE = PACKAGE.joinpath('pipeline')

# The directory for the settings file(s)
SETTINGS = PACKAGE.joinpath('settings')

# A set of filenames to ignore in further processing
IGNORE = set()

bad = PICKLE.joinpath('bad.pkl')

if bad.exists():
    with open(bad, 'rb') as handle:
        ignore = pickle.load(handle)

        for file in ignore:
            file = file.get('filename')
            IGNORE.add(file)

error = PICKLE.joinpath('error.pkl')

if error.exists():
    with open(error, 'rb') as handle:
        ignore = pickle.load(handle)

        for file in ignore:
            file = file.get('filename')
            IGNORE.add(file)

# Get each individual's directory from the original dataset
DIRECTORIES = os_sorted([
    directory
    for directory in ORIGINAL.glob('*/')
    if directory.is_dir()
])

# Get each individual's directory from the processed dataset
INDIVIDUALS = os_sorted([
    individual
    for individual in DATASET.glob('*/')
    if individual.is_dir()
])

# Get each individual's songs from the processed dataset
RECORDINGS = os_sorted([
    file
    for individual in INDIVIDUALS
    for file in individual.glob('recordings/*.wav')
    if file.is_file()
])

# Get each individual's segmentation parameters from the processed dataset
SEGMENTATIONS = os_sorted([
    file
    for individual in INDIVIDUALS
    for file in individual.glob('segmentation/*.json')
    if file.is_file()
])

# Get each individual's .pdf
PDFS = os_sorted([
    individual
    for individual in OUTPUT.glob('pdf/*.pdf')
    if individual.is_file()
])

# Get each individual's Luscinia spectrograms
IMAGES = os_sorted([
    individual
    for individual in INPUT.glob('printout/*/*.jpg')
    if individual.is_file()
])
