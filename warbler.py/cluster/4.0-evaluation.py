import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import scipy
import seaborn as sns
import librosa
import librosa.display

from functions.plot_functions import umap_2Dplot, umap_3Dplot
from functions.evaluation_functions import plot_within_without
from functions.evaluation_functions import nn, sil
# from IPython.display import Image
from path import DATA
from sklearn.neighbors import NearestNeighbors
from spec_params import FMIN, FMAX, FFT_HOP


DF_PATH = DATA.joinpath('df_umap.pkl')

LABEL_COL = 'label'
NA_INDICATOR = "unknown"

distinct_colors_20 = [
    '#e6194b',
    '#3cb44b',
    '#ffe119',
    '#4363d8',
    '#f58231',
    '#911eb4',
    '#46f0f0',
    '#f032e6',
    '#bcf60c',
    '#fabebe',
    '#008080',
    '#e6beff',
    '#9a6324',
    '#fffac8',
    '#800000',
    '#aaffc3',
    '#808000',
    '#ffd8b1',
    '#000075',
    '#808080',
    '#ffffff',
    '#000000'
]

# Load dataframe
df = pd.read_pickle(DF_PATH)

# labels
labels = df[LABEL_COL]

print(df.columns)

umap_2Dplot(
    x=df['UMAP1'],
    y=df['UMAP2'],
    scat_labels=labels,
    mycolors=distinct_colors_20,
    outname=None,
    showlegend=True
)

umap_3Dplot(
    x=df['UMAP1'],
    y=df['UMAP2'],
    z=df['UMAP3'],
    scat_labels=labels,
    mycolors=distinct_colors_20,
    outname=None,
    showlegend=True
)

labelled_df = df.loc[df[LABEL_COL] != NA_INDICATOR, :]

UMAP_COLS = [
    x for x in labelled_df.columns
    if 'UMAP' in x
]

print(f"Found {len(UMAP_COLS)} UMAP columns in df, using all {len(UMAP_COLS)} for subsequent analyses.")


labels = labelled_df[LABEL_COL]
embedding = np.asarray(labelled_df[UMAP_COLS])

knn = 5

nn_stats = nn(embedding, np.asarray(labels), k=knn)

# Summary scores
print("Evaluation score S (unweighted average of same-class probability P for all classes):", round(nn_stats.get_S(), 3))
print("Evaluation score Snorm (unweighted average of normalized same-class probability Pnorm for all classes)::", round(nn_stats.get_Snorm(), 3))

nn_stats.plot_heat_S(
    vmin=0,
    vmax=100,
    center=50,
    cmap=sns.color_palette(
        "Greens",
        as_cmap=True
    ),
    cbar=None,
    outname=None
)

nn_stats.plot_heat_fold(
    center=1,
    cmap=sns.diverging_palette(
        20,
        145,
        as_cmap=True
    ),
    cbar=None,
    outname=None
)

nn_stats.plot_heat_Snorm(
    vmin=-13,
    vmax=13,
    center=1,
    cmap=sns.diverging_palette(
        20,
        145,
        as_cmap=True
    ),
    cbar=None,
    outname=None
)

plot_within_without(
    embedding=embedding,
    labels=labels,
    distance_metric='euclidean',
    outname=None,
    xmin=0,
    xmax=12,
    ymax=0.5,
    nbins=50,
    nrows=6,
    ncols=4,
    density=True
)


sil_stats = sil(embedding, labels)
sil_stats.plot_sil(outname=None)

sil_stats.get_avrg_score()

DISPLAY_COL = 'spectrograms'

knn = 5

nbrs = NearestNeighbors(
    metric='euclidean',
    n_neighbors=knn + 1,
    algorithm='brute'
).fit(embedding)

distances, indices = nbrs.kneighbors(embedding)

indices = indices[:, 1:]
distances = distances[:, 1:]

n_examples = 8

fig = plt.figure(
    figsize=(20, 20)
)

k = 1

random.seed(1)

example_indices = random.sample(
    list(
        range(embedding.shape[0])
    ),
    n_examples
)

for i, ind in enumerate(example_indices):
    img_of_interest = labelled_df.iloc[ind, :][DISPLAY_COL]
    embedding_of_interest = embedding[ind, :]
    plt.subplot(n_examples, knn + 1, k)
    sr = labelled_df.iloc[ind, :].samplerate_hz

    librosa.display.specshow(
        img_of_interest,
        sr=sr,
        hop_length=int(FFT_HOP * sr),
        fmin=FMIN,
        fmax=FMAX,
        y_axis='mel',
        x_axis='s',
        cmap='viridis'
    )

    k = k + 1

    nearest_neighbors = indices[ind]

    for neighbor in nearest_neighbors:
        neighbor_embedding = embedding[neighbor, :]

        dist_to_original = scipy.spatial.distance.euclidean(
            embedding_of_interest,
            neighbor_embedding
        )

        neighbor_img = labelled_df.iloc[neighbor, :][DISPLAY_COL]

        plt.subplot(n_examples, knn + 1, k)
        sr = labelled_df.iloc[neighbor, :].samplerate_hz

        librosa.display.specshow(
            neighbor_img,
            sr=sr,
            hop_length=int(FFT_HOP * sr),
            fmin=FMIN,
            fmax=FMAX,
            y_axis='mel',
            x_axis='s',
            cmap='viridis'
        )

        k = k + 1

plt.tight_layout()


n_examples = 8
major_tick_interval = 20
f_to_s = FFT_HOP
rotate_x = 0

fig = plt.figure(
    figsize=(20, 20)
)

k = 1

# randomly choose
random.seed(1)

example_indices = random.sample(
    list(
        range(embedding.shape[0])
    ),
    n_examples
)

# adjust! this is specific to your N_MELS and samplerate!
freq_label_list = ['512', '1024', '2048']

for i, ind in enumerate(example_indices):
    img_of_interest = labelled_df.iloc[ind, :][DISPLAY_COL]
    embedding_of_interest = embedding[ind, :]
    plt.subplot(n_examples, knn + 1, k)

    # Align specs to left
    ax = plt.gca()
    ax.set_anchor('W')

    plt.imshow(
        img_of_interest,
        interpolation='nearest',
        origin='lower',
        aspect='equal'
    )

    major_xticks = np.arange(
        0,
        img_of_interest.shape[1],
        major_tick_interval
    )

    major_xtick_labels = ["" for x in major_xticks]

    major_yticks = [10, 20, 30]
    major_ytick_labels = freq_label_list

    if i == (n_examples - 1):
        major_xtick_labels = [round(x * f_to_s, 2) for x in major_xticks]
        plt.xlabel('Time (s)')

    plt.ylabel('Hz')
    plt.xticks(major_xticks, major_xtick_labels, rotation=rotate_x)
    plt.yticks(major_yticks, major_ytick_labels,)

    k = k + 1

    nearest_neighbors = indices[ind]

    for neighbor in nearest_neighbors:
        neighbor_embedding = embedding[neighbor, :]

        dist_to_original = scipy.spatial.distance.euclidean(
            embedding_of_interest,
            neighbor_embedding
        )

        neighbor_img = labelled_df.iloc[neighbor, :][DISPLAY_COL]

        plt.subplot(n_examples, knn + 1, k)
        plt.imshow(
            neighbor_img,
            interpolation='nearest',
            origin='lower',
            aspect='equal'
        )

        ax = plt.gca()
        ax.set_anchor('W')

        major_xticks = np.arange(
            0,
            neighbor_img.shape[1],
            major_tick_interval
        )

        major_xtick_labels = ["" for x in major_xticks]

        major_yticks = [10, 20, 30]
        major_ytick_labels = ["" for x in major_yticks]

        if k >= (n_examples * (knn + 1) - knn):
            major_xtick_labels = [round(x * f_to_s, 2) for x in major_xticks]
            plt.xlabel('Time (s)')

        plt.xticks(major_xticks, major_xtick_labels, rotation=rotate_x)

        k = k + 1

plt.show()

# G = nn_stats.draw_simgraph(outname=None)

# Image(
#     G.draw(
#         format='png',
#         prog='neato'
#     ),
#     width=400,
#     height=600
# )
