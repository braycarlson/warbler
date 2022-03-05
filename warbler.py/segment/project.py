import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from avgn.signalprocessing.create_spectrogram_dataset import (
    flatten_spectrograms
)
from avgn.visualization.projections import scatter_spec
from avgn.visualization.quickplots import draw_projection_plots
from avgn.visualization.spectrogram import draw_spec_set
from cuml.manifold.umap import UMAP as cumlUMAP
from path import SEGMENT
from tqdm.autonotebook import tqdm


def norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


df_loc = SEGMENT.joinpath('aw.pickle')


print(df_loc)

syllable_df = pd.read_pickle(df_loc)

print(syllable_df[:3])

print(len(syllable_df))


fig, axs = plt.subplots(
    ncols=4,
    figsize=(24, 6)
)

axs[0].hist(
    [np.max(i) for i in syllable_df.spectrogram.values],
    bins=50
)

axs[0].set_title('max')

axs[1].hist(
    [np.sum(i) for i in syllable_df.spectrogram.values],
    bins=50
)

axs[1].set_title('sum')

axs[2].hist(
    (syllable_df.end_time - syllable_df.start_time).values,
    bins=50
)

axs[2].set_title('len')

axs[3].hist(
    [np.min(i) for i in syllable_df.spectrogram.values],
    bins=50
)

axs[3].set_title('min')

len(syllable_df)

specs = list(syllable_df.spectrogram.values)
specs = [norm(i) for i in specs]
specs_flattened = flatten_spectrograms(specs)
np.shape(specs_flattened)

draw_spec_set(specs, zoom=1, maxrows=10, colsize=25)

print(syllable_df.indv.unique())

for indv in tqdm(syllable_df.indv.unique()):
    indv_df = syllable_df[syllable_df.indv == indv]
    print(indv, len(indv_df))

    specs = list(indv_df.spectrogram.values)
    draw_spec_set(specs, zoom=1, maxrows=10, colsize=25)

    specs_flattened = flatten_spectrograms(specs)

    cuml_umap = cumlUMAP(min_dist=0.25)
    z = list(cuml_umap.fit_transform(specs_flattened))

    # fit = umap.UMAP()
    # z = list(fit.fit_transform(specs_flattened))
    indv_df["umap"] = z

    indv_df["syllables_sequence_id"] = None
    indv_df["syllables_sequence_pos"] = None
    indv_df = indv_df.sort_values(by=["key", "start_time"])
    for ki, key in enumerate(indv_df.key.unique(), 0):
        indv_df.loc[indv_df.key == key, "syllables_sequence_id"] = ki
        indv_df.loc[indv_df.key == key, "syllables_sequence_pos"] = np.arange(
            np.sum(indv_df.key == key)
        )

    draw_projection_plots(indv_df, label_column="indv")

    scatter_spec(
        np.vstack(z),
        specs,
        column_size=15,
        # x_range = [-5.5, 7],
        # y_range = [-10, 10],
        pal_color="hls",
        color_points=False,
        enlarge_points=20,
        figsize=(10, 10),
        scatter_kwargs={
            # 'labels': list(indv_df.labels.values),
            'alpha': 0.25,
            's': 1,
            'show_legend': False
        },
        matshow_kwargs={
            'cmap': plt.cm.Greys
        },
        line_kwargs={
            'lw': 1,
            'ls': "solid",
            'alpha': 0.25,
        },
        draw_lines=True
    )

    path = SEGMENT.joinpath(indv + '.png')

    plt.savefig(
        path,
        dpi=300,
        bbox_inches='tight',
        pad_inches=0,
        transparent=True,
    )

    plt.show()
