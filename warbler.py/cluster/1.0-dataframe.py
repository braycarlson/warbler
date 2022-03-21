import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random

from functions.audio_functions import generate_mel_spectrogram, read_wavfile
from parameters import Parameters
from path import CLUSTER, DATA, NOTES


path = CLUSTER.joinpath('parameters.json')
parameters = Parameters(path)

path = CLUSTER.joinpath('dataframe.json')
dataframe = Parameters(path)


def plot_examples(df, random_indices, spec_type):
    if spec_type in df.columns:

        df_subset = df.iloc[random_indices, :]
        specs = df_subset[spec_type].values

        plt.figure(figsize=(len(random_indices) * 3, 3))

        for i, spec in enumerate(specs):
            plt.subplot(1, len(random_indices), i + 1)
            plt.imshow(spec, origin='lower')

        plt.suptitle(spec_type)
        plt.tight_layout()
    else:
        print(f"Error: {spec_type} not in df columns")


NA_DESCRIPTORS = [
    0,
    np.nan,
    'NA',
    'na',
    'not available',
    'None',
    'Unknown',
    'unknown',
    None,
    ''
]

if not NOTES.is_dir():
    print('No notes directory found')

file = DATA.joinpath('notes.csv')
notes = [file.name for file in NOTES.glob('*.wav')]

if file.is_file():
    df = pd.read_csv(file, sep=';')
else:
    if len(notes) > 0:
        df = pd.DataFrame(
            {
                'filename': [note for note in notes],
                'label': ['unknown'] * len(notes)
            }
        )

print(df)

filenames = df['filename'].values

missing_files = list(
    set(filenames) - set(notes)
)

if len(missing_files) > 0:
    length = len(missing_files)

    print(f"Warning: {length} files with no matching note in notes directory")
    print(missing_files)

paths = [
    NOTES.joinpath(filename).as_posix()
    for filename in filenames
]

raw_audio, samplerate_hz = map(
    list,
    zip(*[read_wavfile(path) for path in paths])
)

df['raw_audio'] = raw_audio
df['samplerate_hz'] = samplerate_hz

# Removing NA rows

nrows = df.shape[0]

df.dropna(
    subset=['raw_audio'],
    inplace=True
)

print(f"Dropped {nrows - df.shape[0]} rows due to missing/failed audio")

# Extract duration of calls
df['duration_s'] = [x.shape[0] for x in df['raw_audio']] / df['samplerate_hz']

print(f"Dropped {df.loc[df['duration_s'] < parameters.minimum_duration, :].shape[0]} rows below {parameters.minimum_duration}s (min_dur)")
df = df.loc[df['duration_s'] >= parameters.minimum_duration, :]

print(f"Dropped {df.loc[df['duration_s'] > parameters.maximum_duration, :].shape[0]} rows above {parameters.maximum_duration}s (max_dur)")
df = df.loc[df['duration_s'] <= parameters.maximum_duration, :]


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
    lambda row:
    generate_mel_spectrogram(
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

df['spectrogram'] = spectrograms

# Removing NA rows

nrows = df.shape[0]

df.dropna(
    subset=['spectrogram'],
    inplace=True
)

print(f"Dropped {nrows - df.shape[0]} rows due to failed spectrogram generation")


# Randomly choose some example vocalizations

n_examples = 3

random_indices = [
    random.randint(0, df.shape[0]) for x in [0] * n_examples
]

if parameters.plot_examples:
    plot_examples(df, random_indices, 'spectrogram')

if parameters.median_subtracted:
    df['denoised_spectrograms'] = [
        spectrogram - np.median(
            spectrogram,
            axis=0)
        for spectrogram in df['spectrogram']
    ]

if parameters.median_subtracted and parameters.plot_examples:
    plot_examples(
        df,
        random_indices,
        'denoised_spectrograms'
    )

if parameters.bandpass_filter:
    from functions.audio_functions import butter_bandpass_filter

    # create filtered audio
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

    # create spectrograms from filtered audio
    df['filtered_spectrogram'] = df.apply(
        lambda row:
        generate_mel_spectrogram(
            data=row['filtered_audio'],
            rate=row['samplerate_hz'],
            n_mels=parameters.mel_bins_filtered,
            window=parameters.window,
            fft_win=parameters.fft_win,
            fft_hop=parameters.fft_hop,
            fmax=parameters.highcut,
            fmin=parameters.lowcut
        ),
        axis=1
    )

if parameters.bandpass_filter and parameters.plot_examples:
    plot_examples(
        df,
        random_indices,
        'filtered_spectrogram'
    )

if parameters.stretch:
    from functions.audio_functions import generate_stretched_mel_spectrogram

    MAX_DURATION = np.max(df['duration_s'])

    df['stretched_spectrograms'] = df.apply(
        lambda row:
        generate_stretched_mel_spectrogram(
            row['raw_audio'],
            row['samplerate_hz'],
            row['duration_s'],
            parameters.mel_bins,
            parameters.window,
            parameters.fft_win,
            parameters.fft_hop,
            parameters.maximum_duration
        ),
        axis=1
    )

if parameters.stretch and parameters.plot_examples:
    plot_examples(
        df,
        random_indices,
        'stretched_spectrograms'
    )

if 'filtered_spectrogram' in df.columns:
    df['denoised_filtered_spectrogram'] = [
        spectrogram - np.median(spectrogram, axis=0)
        for spectrogram in df['filtered_spectrogram']
    ]

    if parameters.plot_examples:
        plot_examples(
            df,
            random_indices,
            'denoised_filtered_spectrogram'
        )


df['original_label'] = df[dataframe.label_column]

df['label'] = [
    'unknown' if x in NA_DESCRIPTORS else
    x for x in df[dataframe.label_column]
]

labels = df['label'].fillna(dataframe.na_label)
df['label'] = labels.astype(str)

df.to_pickle(
    DATA.joinpath('df.pkl')
)
