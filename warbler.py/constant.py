import pickle

from natsort import os_sorted
from pathlib import Path, PurePath


def relative(path):
    pure = PurePath(path)
    return pure.relative_to(CWD)


# The current-working directory of the project
condition = (
    Path.cwd().joinpath('setup.py').is_file() or
    Path.cwd().parent.joinpath('notebook').is_dir()
)

if condition:
    CWD = Path.cwd().parent
else:
    CWD = Path.cwd().parent.parent

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

# The directory for generated spectrograms to be inspected
SPECTROGRAM = OUTPUT.joinpath('spectrogram')

# The root directory for the package
PACKAGE = CWD.joinpath('warbler.py')

# The directory for the cluster package
CLUSTER = PACKAGE.joinpath('cluster')

# The directory for the analyze package
ANALYZE = PACKAGE.joinpath('analyze')

# The directory for the segment package
SEGMENT = PACKAGE.joinpath('segment')

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

# Get each individual's metadata from the processed dataset
METADATA = os_sorted([
    file
    for individual in INDIVIDUALS
    for file in individual.glob('metadata/*.json')
    if file.is_file()
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
