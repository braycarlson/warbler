import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from avgn.dataset import DataSet
from avgn.signalprocessing.create_spectrogram_dataset import (
    create_label_df,
    get_row_audio,
    log_resize_spec,
    make_spec,
    pad_spectrogram,
)
from avgn.utils.hparams import HParams
from avgn.visualization.spectrogram import draw_spec_set
from joblib import Parallel, delayed
from parameters import Parameters
from path import INDIVIDUALS
from pathlib import Path
from tqdm.autonotebook import tqdm


# [Optional]
# Normalize the spectrograms into uint8
# This will make the dataset smaller
def norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


# Parameters
file = Path('parameters.json')
parameters = Parameters(file)

# Create a set of parameters for processing the dataset
hparams = HParams(
    n_fft=parameters.n_fft,
    hop_length_ms=parameters.hop_length_ms,
    win_length_ms=parameters.win_length_ms,
    ref_level_db=parameters.ref_level_db,
    pre=parameters.pre,
    min_level_db=parameters.min_level_db,
    min_level_db_floor=parameters.min_level_db_floor,
    db_delta=parameters.db_delta,
    silence_threshold=parameters.silence_threshold,
    min_silence_for_spec=parameters.min_silence_for_spec,
    max_vocal_for_spec=parameters.max_vocal_for_spec,
    min_syllable_length_s=parameters.min_syllable_length_s,
    spectral_range=parameters.spectral_range,

    num_mel_bins=parameters.num_mel_bins,
    mel_lower_edge_hertz=parameters.mel_lower_edge_hertz,
    mel_upper_edge_hertz=parameters.mel_upper_edge_hertz,
    butter_lowcut=parameters.butter_lowcut,
    butter_highcut=parameters.butter_highcut,
    mask_spec=parameters.mask_spec,
    nex=parameters.nex,
    n_jobs=parameters.n_jobs,
    verbosity=parameters.verbosity
)


dataset = DataSet(INDIVIDUALS, hparams=hparams)

n_jobs = parameters.n_jobs
verbosity = parameters.n_jobs

with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
    syllable_dfs = parallel(
        delayed(create_label_df)(
            dataset.data_files[key].data,
            hparams=dataset.hparams,
            labels_to_retain=["labels", "sequence_num"],
            unit="notes",
            dict_features_to_retain=[],
            key=key,
        )
        for key in tqdm(dataset.data_files.keys())
    )

syllable_df = pd.concat(syllable_dfs)

with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
    syllable_dfs = parallel(
        delayed(get_row_audio)(
            syllable_df[syllable_df.key == key],
            dataset.data_files[key].data['wav_loc'],
            dataset.hparams
        )
        for key in tqdm(syllable_df.key.unique())
    )

syllable_df = pd.concat(syllable_dfs)

# Get rid of syllables that are zero seconds,
# which will produce errors in segmentation
df_mask = np.array(
    [len(i) > 0 for i in tqdm(syllable_df.audio.values)]
)
syllable_df = syllable_df[np.array(df_mask)]


syllable_df['audio'] = [
    librosa.util.normalize(i) for i in syllable_df.audio.values
]

# Plot some example audio
nrows = 5
ncols = 10
zoom = 2

fig, axs = plt.subplots(
    ncols=ncols,
    nrows=nrows,
    figsize=(ncols * zoom, nrows + zoom / 1.5)
)

for i, syll in tqdm(enumerate(syllable_df['audio'].values), total=nrows*ncols):
    ax = axs.flatten()[i]
    ax.plot(syll)

    if i == nrows * ncols - 1:
        break

plt.show()

with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
    # Create spectrograms
    syllables_spec = parallel(
        delayed(make_spec)(
            syllable,
            rate,
            hparams=dataset.hparams,
            mel_matrix=dataset.mel_matrix,
            use_mel=True,
            use_tensorflow=False,
        )
        for syllable, rate in tqdm(
            zip(syllable_df.audio.values, syllable_df.rate.values),
            total=len(syllable_df),
            desc="Getting syllable spectrograms",
            leave=False,
        )
    )

plt.matshow(syllables_spec[10])


# A hyperparameter where larger = higher dimensional spectrogram
log_scaling_factor = 10

with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
    syllables_spec = parallel(
        delayed(log_resize_spec)(spec, scaling_factor=log_scaling_factor)
        for spec in tqdm(
            syllables_spec,
            desc="scaling spectrograms",
            leave=False
        )
    )

# Lets take a look at these spectrograms
draw_spec_set(
    syllables_spec,
    zoom=1,
    maxrows=10,
    colsize=25
)

plt.show()

syllables_spec = [
    (norm(i) * 255).astype('uint8') for i in tqdm(syllables_spec)
]

syll_lens = [np.shape(i)[1] for i in syllables_spec]
plt.hist(syll_lens)

plt.show()

pad_length = np.max(syll_lens)

with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
    syllables_spec = parallel(
        delayed(pad_spectrogram)(spec, pad_length)
        for spec in tqdm(
            syllables_spec, desc="padding spectrograms", leave=False
        )
    )

draw_spec_set(syllables_spec, zoom=1, maxrows=10, colsize=25)
plt.show()

# What is the dimensionality of the dataset
print(np.shape(syllables_spec))

# convert to uint8 to save space
syllables_spec = [
    (norm(i) * 255).astype('uint8') for i in tqdm(syllables_spec)
]

syllable_df['spectrogram'] = syllables_spec
