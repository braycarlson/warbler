import pickle

from natsort import os_sorted
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

# The directory for the segment package
SEGMENT = PACKAGE.joinpath('segment')

# The file for the baseline parameters
BASELINE = SEGMENT.joinpath('parameter.json')

# The directory for pickled songs
PICKLE = SEGMENT.joinpath('pickle')

# The directory for generated spectrograms to be inspected
SPECTROGRAM = SEGMENT.joinpath('spectrogram')

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
            IGNORE.add(file)

# Get each individual's directory from the original dataset
DIRECTORIES = os_sorted([
    directory
    for directory in DATASET.glob('*/')
    if directory.is_dir()
])

# Get each individual's directory from the processed dataset
INDIVIDUALS = os_sorted([
    individual
    for individual in DATA.glob('*/')
    if individual.is_dir()
])

# Get each individual's songs from the processed dataset
SONGS = os_sorted([
    file
    for individual in INDIVIDUALS
    for file in individual.glob('wav/*.wav')
    if file.is_file()
])

# Get each individual's parameters from the processed dataset
PARAMETERS = os_sorted([
    file
    for individual in INDIVIDUALS
    for file in individual.glob('parameter/*.json')
    if file.is_file()
])

# Get each individual's metadata from the processed dataset
METADATA = os_sorted([
    file
    for individual in INDIVIDUALS
    for file in individual.glob('json/*.json')
    if file.is_file()
])

# Get each individual's .pdf
PDFS = os_sorted([
    individual
    for individual in ANALYZE.glob('pdf/*.pdf')
    if individual.is_file()
])

# Get each individual's Luscinia spectrograms
IMAGES = os_sorted([
    individual
    for individual in ANALYZE.glob('luscinia/*/*.jpg')
    if individual.is_file()
])

# Get each individual's directory from the printouts
PRINTOUTS = os_sorted([
    individual
    for individual in PRINTOUT.glob('*/')
    if individual.is_dir()
])
