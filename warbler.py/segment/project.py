import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap

from avgn.signalprocessing.create_spectrogram_dataset import (
    flatten_spectrograms
)
from avgn.visualization.projections import scatter_spec, scatter_projections
from avgn.visualization.quickplots import draw_projection_plots
from avgn.visualization.spectrogram import draw_spec_set
# from cuml.manifold.umap import UMAP as cumlUMAP
from path import PICKLE, SEGMENT
from tqdm import tqdm


def norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


syllable_df = pd.read_pickle(
    PICKLE.joinpath('aw.pkl')
)

for indv in tqdm(syllable_df.indv.unique()):
    indv_df = syllable_df[syllable_df.indv == indv]
    indv_df = indv_df.sort_values(by=['key', 'start_time'])

    specs = list(indv_df.spectrogram.values)

    specs_flattened = flatten_spectrograms(specs)
    fit = umap.UMAP(min_dist=0.50)
    z = list(fit.fit_transform(specs_flattened))
    indv_df['umap'] = z

    scatter_spec(
        np.vstack(z),
        specs,
        column_size=15,
        pal_color='hls',
        color_points=False,
        enlarge_points=20,
        figsize=(10, 10),
        scatter_kwargs={
            'labels': list(indv_df.labels.values),
            'alpha': 0.25,
            's': 1,
            'show_legend': False
        },
        matshow_kwargs={
            'cmap': plt.cm.Greys
        },
        line_kwargs={
            'lw': 1,
            'ls': 'solid',
            'alpha': 0.25,
        },
        draw_lines=True
    )

    plt.show()






# specs = list(syllable_df.spectrogram.values)
# specs = [norm(i) for i in specs]
# specs_flattened = flatten_spectrograms(specs)
# np.shape(specs_flattened)

# fit = umap.UMAP()
# embedding = fit.fit_transform(specs_flattened)
# z = np.vstack(list(embedding))

# fig, ax = plt.subplots(
#     figsize=(15, 15)
# )

# scatter_projections(
#     projection=z,
#     alpha=1,
#     labels=syllable_df['indv'].values,
#     s=5,
#     ax=ax
# )
# plt.show()


# scatter_spec(
#     z,
#     specs,
#     column_size=15,
#     color_points=False,
#     enlarge_points=20,
#     figsize=(10, 10),
#     scatter_kwargs={
#         'labels': syllable_df.labels.values,
#         'alpha': 0.25,
#         's': 1,
#         'show_legend': False,
#         'color_palette': 'bright'
#     },
#     matshow_kwargs={
#         'cmap': plt.cm.Greys
#     },
#     line_kwargs={
#         'lw': 1,
#         'ls': 'solid',
#         'alpha': 0.5,
#     },
#     draw_lines=True
# )

# plt.show()
# plt.close()
