import json
import librosa
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataframe.dataframe import (
    create_label_df,
    Dataset,
    get_row_audio,
    log_resize_spec,
    make_spec,
    norm,
    pad_spectrogram,
)
from avgn.visualization.spectrogram import draw_spec_set
from joblib import delayed, Parallel
from path import DATA, INDIVIDUALS, SEGMENT
from scipy.io import wavfile
from tqdm.autonotebook import tqdm


dataset = Dataset(INDIVIDUALS)


with Parallel(n_jobs=-1, verbose=1) as parallel:
    syllable_dfs = parallel(
        delayed(create_label_df)(
            dataset.datafiles[key].data,
            labels_to_retain=["labels", "filename"],
            unit="notes",
            dict_features_to_retain=[],
            key=key,
        )
        for key in tqdm(dataset.datafiles.keys())
    )

syllable_df = pd.concat(syllable_dfs)

with Parallel(n_jobs=-1, verbose=1) as parallel:
    syllable_dfs = parallel(
        delayed(get_row_audio)(
            syllable_df[syllable_df.key == key],
            dataset.datafiles[key].data['wav_loc'],
            key
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

region = {
    k: v for k, v in syllable_df.groupby('filename')
}

for filename in region.keys():
    ic = region[filename].index
    bc = region[filename].get('indv')
    ac = region[filename].get('audio').tolist()
    rc = region[filename].get('rate')

    notes = []

    for index, bird, audio, rate in zip(ic, bc, ac, rc):
        index = str(index).zfill(2)
        rate = int(rate)

        directory = DATA.joinpath(bird, 'notes')
        path = directory.joinpath(filename + '_' + index + '.wav')

        wavfile.write(
            path,
            rate,
            audio
        )

        path = path.as_posix()
        notes.append(path)

    directory = DATA.joinpath(bc[0], 'json')
    path = directory.joinpath(filename + '.json')

    with open(path, 'r') as file:
        try:
            data = json.load(file)
        except json.decoder.JSONDecodeError:
            print(f"Unable to open: {path}")
            continue

        template = data.get('indvs').get(bc[0]).get('notes')
        template['files'] = notes

    with open(path, 'w+') as file:
        text = json.dumps(data, indent=2)
        file.write(text)


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

with Parallel(n_jobs=-1, verbose=1) as parallel:
    # Create spectrograms
    syllables_spec = parallel(
        delayed(make_spec)(
            syllable,
            rate,
            key,
            use_mel=True,
            use_tensorflow=False,
        )
        for syllable, rate, key in tqdm(
            zip(
                syllable_df.audio.values,
                syllable_df.rate.values,
                syllable_df.key.values,
            ),
            total=len(syllable_df),
            desc="Getting syllable spectrograms",
            leave=False,
        )
    )

plt.matshow(syllables_spec[10])

# A hyperparameter where larger = higher dimensional spectrogram
log_scaling_factor = 10

with Parallel(n_jobs=-1, verbose=1) as parallel:
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

with Parallel(n_jobs=-1, verbose=1) as parallel:
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

file = 'aw.pickle'

syllable_df.to_pickle(
    SEGMENT.joinpath(file)
)
