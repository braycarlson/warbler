import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random

from functions.audio_functions import (
    butter_bandpass_filter,
    generate_mel_spectrogram,
    read_wavfile
)
from parameters import Parameters
from path import DATA, NOTES
from pathlib import Path


# Function for plotting example spectrograms
def plot_examples(df, random_indices, spec_type):
    if spec_type in df.columns:
        df_subset = df.iloc[random_indices, :]
        specs = df_subset[spec_type].values

        plt.figure(
            figsize=(
                len(random_indices) * 3, 3
            )
        )

        for i, spec in enumerate(specs):
            plt.subplot(1, len(random_indices), i + 1)
            plt.imshow(spec, origin='lower')

        plt.suptitle(spec_type)
        plt.tight_layout()

        plt.show()
    else:
        print(f"Error: {spec_type} not in df columns")


# Dataframe
file = Path('dataframe.json')
dataframe = Parameters(file)


# Parameters
file = Path('parameters.json')
parameters = Parameters(file)


# Information about the notes.csv
NOTES_FILE = DATA.joinpath('notes.csv')

if NOTES_FILE.is_file():
    df = pd.read_csv(NOTES_FILE, sep=";")
else:
    print(f"Input file missing: {NOTES_FILE}")

    audiofiles = os.listdir(NOTES)

    if len(audiofiles) > 0:
        df = pd.DataFrame(
            {
                'filename': [Path(x).stem for x in audiofiles],
                'label': ['Unknown'] * len(audiofiles)
            }
        )


audiofiles = df['filename'].values
files_in_audio_directory = os.listdir(NOTES)

raw_audio, samplerate_hz = map(
    list,
    zip(*[read_wavfile(path) for path in df['path'].values])
)

df['raw_audio'] = raw_audio
df['samplerate_hz'] = samplerate_hz

# Removing NA rows

nrows = df.shape[0]
df.dropna(subset=['raw_audio'], inplace=True)
print(f"Dropped {nrows - df.shape[0]} rows due to missing/failed audio")

# Extract duration of calls
df['duration_s'] = [x.shape[0] for x in df['raw_audio']] / df['samplerate_hz']


# print(f"Dropped {df.loc[df['duration_s'] < parameters.minimum_duration, :].shape[0]} rows below {parameters.minimum_duration} s")
# df = df.loc[df['duration_s'] >= parameters.minimum_duration, :]

# print(f"Dropped {df.loc[df['duration_s'] < parameters.maximum_duration, :].shape[0]} rows above {parameters.maximum_duration} s")
# df = df.loc[df['duration_s'] <= parameters.maximum_duration, :]

parameters.update(
    'fft_hop',
    parameters.fft_win / 8
)

parameters.update(
    'fmax',
    int(np.min(df['samplerate_hz']) / 2)
)

parameters.save()

spectrograms = df.apply(
    lambda row: generate_mel_spectrogram(
            data=row['raw_audio'],
            rate=row['samplerate_hz'],
            n_mels=parameters.mel_bins,
            window=parameters.window,
            fft_win=parameters.fft_win,
            fft_hop=parameters.fft_hop,
            fmax=parameters.fmax
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


if parameters.bandpass_filter:
    # Create filtered audio
    df['filtered_audio'] = df.apply(
        lambda row:
        butter_bandpass_filter(
            data=row['raw_audio'],
            lowcut=parameters.lowcut,
            highcut=parameters.highcut,
            sr=row['samplerate_hz'],
            order=6
        ),
        axis=1
    )

    # Create spectrograms from filtered audio
    df['filtered_spectrograms'] = df.apply(
        lambda row:
        generate_mel_spectrogram(
            data=row['filtered_audio'],
            rate=row['samplerate_hz'],
            n_mels=parameters.mel_bins,
            window=parameters.window,
            fft_win=parameters.fft_win,
            fft_hop=parameters.fft_hop,
            fmax=parameters.highcut,
            fmin=parameters.lowcut
        ),
        axis=1
    )

plot_examples(df, random_indices, 'filtered_spectrograms')

# Original labels are saved in "original_label" column
df['original_label'] = df[dataframe.label_column]

# "Unknown" for all NA labels
df['label'] = [
    'Unknown' if x in dataframe.na_descriptor else x
    for x in df[dataframe.label_column]
]

# Double-check
labels = df['label'].fillna(dataframe.na_label)

# Transform to strings and save in df as "label" column
df['label'] = labels.astype(str)

pkl = DATA.joinpath('df.pkl')
df.to_pickle(pkl)

dataframe.close()
parameters.close()
