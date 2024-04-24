from __future__ import annotations

import json
import pickle

from natsort import os_sorted
from pathlib import Path


def walk(file: Path) -> Path | None:
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
LOGS.mkdir(parents=True, exist_ok=True)

# The directory for input
INPUT = CWD.joinpath('input')
INPUT.mkdir(parents=True, exist_ok=True)

# The directory for converted recordings
DATASET = INPUT.joinpath('dataset')
DATASET.mkdir(parents=True, exist_ok=True)

# The directory for Luscinia printouts
LUSCINIA = INPUT.joinpath('printout')
LUSCINIA.mkdir(parents=True, exist_ok=True)

# The directory for Luscinia printouts
SPREADSHEET = INPUT.joinpath('spreadsheet')
SPREADSHEET.mkdir(parents=True, exist_ok=True)

# The directory for output
OUTPUT = CWD.joinpath('output')
OUTPUT.mkdir(parents=True, exist_ok=True)

# The directory for the .pdf files
PDF = OUTPUT.joinpath('pdf')
PDF.mkdir(parents=True, exist_ok=True)

# The directory for the Parquet output
PARQUET = OUTPUT.joinpath('parquet')
PARQUET.mkdir(parents=True, exist_ok=True)

# The directory for projection(s)
PROJECTION = OUTPUT.joinpath('projection')
PROJECTION.mkdir(parents=True, exist_ok=True)

# The directory for generated spectrograms to be inspected
SPECTROGRAM = OUTPUT.joinpath('spectrogram')
SPECTROGRAM.mkdir(parents=True, exist_ok=True)

# The directory for hyperparameter tuning
TUNING = OUTPUT.joinpath('tuning')
TUNING.mkdir(exist_ok=True, parents=True)

# The root directory for the package
PACKAGE = CWD.joinpath('warbler')

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

bad = PARQUET.joinpath('bad.pkl')

if bad.exists():
    with open(bad, 'rb') as handle:
        ignore = pickle.load(handle)

        for file in ignore:
            file = file.get('filename')
            IGNORE.add(file)

error = PARQUET.joinpath('error.pkl')

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

def object_hook(file: str) -> dict[int, list[float]]:
    return {
        int(key): value
        for key, value in file.items()
    }

path = SETTINGS.joinpath('palette.json')

with open(path, 'r') as handle:
    file = handle.read()

    MASTER = json.loads(
        file,
        object_hook=object_hook
    )
