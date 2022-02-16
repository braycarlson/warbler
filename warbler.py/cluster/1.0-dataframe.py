import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random

from functions.audio_functions import generate_mel_spectrogram, read_wavfile
from path import CWD, DATA, PARAMETERS
from pathlib import Path


# Set important paths

# Project directory, by default set to parent directory
P_DIR = str(CWD)

# Audio directory, which contains audio (.wav) files
AUDIO_IN = CWD.joinpath('notes')


# Information about the info_file.csv

# Name of column that contains labels
LABEL_COL = "label"

NA_DESCRIPTORS = ["None", None]

# All vocalizations without label will be relabelled to "unknown"
NEW_NA_INDICATOR = "Unknown"


# Other parameters

# Do you want to plot example spectrograms?
PLOT_EXAMPLES = True

# Should bandpass-filtered spectrograms be generated?
BANDPASS_FILTER = False

# Should median-subtracted spectrograms be generated
# (Reduce impulse noise)?
MEDIAN_SUB = False

# Should time-stretched spectrograms be generated
# (All stretched to max. duration in dataset)?
STRETCH = False


# Function for plotting example spectrograms
def plot_examples(df, random_indices, spec_type):
    if spec_type in df.columns:
        df_subset = df.iloc[random_indices, :]
        specs = df_subset[spec_type].values

        plt.figure(
            figsize=(len(random_indices) * 3, 3))

        for i, spec in enumerate(specs):
            plt.subplot(1, len(random_indices), i + 1)
            plt.imshow(spec, origin='lower')

        plt.suptitle(spec_type)
        plt.tight_layout()

        plt.show()
    else:
        print(f"Error: {spec_type} not in df columns")


info_file = DATA.joinpath('info_file.csv')

if info_file.is_file():
    df = pd.read_csv(info_file, sep=";")
else:
    print(f"Input file missing: {info_file}")
    print("Will create default input file without labels")

    audiofiles = os.listdir(AUDIO_IN)

    if len(audiofiles) > 0:
        df = pd.DataFrame(
            {
                'filename': [Path(x).stem for x in audiofiles],
                'label': ["Unknown"] * len(audiofiles)
            }
        )


audiofiles = df['filename'].values
files_in_audio_directory = os.listdir(AUDIO_IN)

raw_audio, samplerate_hz = map(
    list,
    zip(*[read_wavfile(x) for x in audiofiles])
)

df['raw_audio'] = raw_audio
df['samplerate_hz'] = samplerate_hz

# Removing NA rows

nrows = df.shape[0]
df.dropna(subset=['raw_audio'], inplace=True)
print("Dropped ", nrows-df.shape[0], " rows due to missing/failed audio")

# Extract duration of calls
df['duration_s'] = [x.shape[0] for x in df['raw_audio']]/df['samplerate_hz']

# Minimum duration of calls in seconds
MIN_DUR = 0

# Maximum duration of calls in seconds
MAX_DUR = 1

print(f"Dropped {df.loc[df['duration_s'] < MIN_DUR, :].shape[0]} rows below {MIN_DUR} s")
df = df.loc[df['duration_s'] >= MIN_DUR, :]
print(f"Dropped {df.loc[df['duration_s'] < MAX_DUR, :].shape[0]} rows above {MAX_DUR} s")
df = df.loc[df['duration_s'] <= MAX_DUR, :]

# Number of mel bins (usually 20-40)
# The frequency bins are transformed to this number of logarithmically spaced mel bins.
N_MELS = 40

# Length of audio chunk when applying STFT in seconds
# FFT_WIN * samplerate = number of audio datapoints that go in one fft (=n_fft)
FFT_WIN = 0.03

# Hop_length in seconds
# FFT_HOP * samplerate = n of audio datapoints between successive ffts (=hop_length)
FFT_HOP = FFT_WIN / 8

# Name of window function
# Each frame of audio is windowed by a window function.
# We use the window function 'hanning',
WINDOW = 'hann'

# Lower bound for frequency (in Hz) when generating Mel filterbank
FMIN = 0

# Upper bound for frequency (in Hz) when generating Mel filterbank
# this is set to 0.5 times the samplerate (-> Nyquist rule)
# If input files have different samplerates, the lowest samplerate is used
# to ensure all spectrograms have the same frequency resolution.
FMAX = int(np.min(df['samplerate_hz']) / 2)


spec_params_file = PARAMETERS.joinpath('spec_params.py')

lines = [
    'N_MELS = ' + str(N_MELS),
    'FFT_WIN = ' + str(FFT_WIN),
    'FFT_HOP = ' + str(FFT_HOP),
    'WINDOW = "' + str(WINDOW) + '"',
    'FMIN = ' + str(FMIN),
    'FMAX = ' + str(FMAX)
]

with open(spec_params_file, 'w') as f:
    for line in lines:
        f.write(line)
        f.write('\n')


spectrograms = df.apply(
    lambda row: generate_mel_spectrogram(
            data=row['raw_audio'],
            rate=row['samplerate_hz'],
            n_mels=N_MELS,
            window=WINDOW,
            fft_win=FFT_WIN,
            fft_hop=FFT_HOP,
            fmax=FMAX
        ),
    axis=1
)


df['spectrograms'] = spectrograms


# Removing NA rows
nrows = df.shape[0]
df.dropna(subset=['spectrograms'], inplace=True)
print(f"Dropped {nrows-df.shape[0]} rows due to failed spectrogram generation")

# Randomly choose some example vocalizations
n_examples = 10

random_indices = [
    random.randint(0, df.shape[0]) for x in [0] * n_examples
]

plot_examples(df, random_indices, 'spectrograms')

# Original labels are saved in "original_label" column
df['original_label'] = df[LABEL_COL]

# "Unknown" for all NA labels
df['label'] = [
    "Unknown" if x in NA_DESCRIPTORS else x for x in df[LABEL_COL]
]

# Double-check
labels = df['label'].fillna(NEW_NA_INDICATOR)

# Transform to strings and save in df as "label" column
df['label'] = labels.astype(str)

pkl = DATA.joinpath('df.pkl')
df.to_pickle(pkl)
