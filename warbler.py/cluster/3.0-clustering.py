import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

from parameters import Parameters
from path import DATA
from pathlib import Path
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score
from functions.plot_functions import umap_3Dplot


# Import functions for calculating unadjusted Rand score

# Resource:
# https://stackoverflow.com/questions/49586742/rand-index-function-clustering-performance-evaluation

def rand_index_score(clusters, classes):
    tp_plus_fp = scipy.special.comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = scipy.special.comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]
    tp = sum(
        scipy.special.comb(
            np.bincount(A[A[:, 0] == i, 1]), 2
        ).sum() for i in set(clusters)
    )
    fp = tp_plus_fp - tp
    fn = tp_plus_fn - tp
    tn = scipy.special.comb(len(A), 2) - tp - fp - fn
    return (tp + tn) / (tp + fp + fn + tn)


def calc_rand(pred, true):
    classnames, pred = np.unique(pred, return_inverse=True)
    classnames, true = np.unique(true, return_inverse=True)
    return rand_index_score(pred, true)


distinct_colors_22 = [
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


# Dataframe
file = Path('dataframe.json')
dataframe = Parameters(file)

# Load dataframe with UMAP coordinates
pkl = DATA.joinpath('df_umap.pkl')
df = pd.read_pickle(pkl)

# Detect columns with UMAP coordinates
UMAP_COLS = [x for x in df.columns if 'UMAP' in x]

# Save labels and embedding as variables

# labels
labels = df[dataframe.label_column]

# UMAP coordinates
embedding = np.asarray(df[UMAP_COLS])

print(f"Found: {len(UMAP_COLS)} UMAP columns.")
print(f"Label in column: {dataframe.label_column}")


# Clustering
HDBSCAN = hdbscan.HDBSCAN(
    min_cluster_size=int(0.01 * embedding.shape[0]),
    cluster_selection_method='leaf'
).fit(embedding)

# These are the predicted clusters labels
hdb_labels = HDBSCAN.labels_

# Add predicted cluster labels to dataframe
df['HDBSCAN'] = hdb_labels


# For some evaluations, it makes sense to remove the datapoints labelled as noise
# "Noise" datapoints are by default labelled as -1 by HDBSCAN

hdb_labels_no_noise = hdb_labels.copy()

assigned = np.full(
    (
        len(hdb_labels_no_noise),
    ),
    False
)

assigned[
    np.where(hdb_labels_no_noise != -1)[0]
] = True

# the predicted cluster labels without noise datapoints
hdb_labels_no_noise = hdb_labels_no_noise[assigned]

# The dataframe without noise datapoints
df_no_noise = df.loc[assigned]

# The UMAP coordinates without noise datapoints
embedding_no_noise = embedding[assigned, :]


# evaluate clustering

print("************************")
print("HDBSCAN:")
print("************************")
cluster_labels = hdb_labels
true_labels = df[dataframe.label_column]
embedding_data = embedding


print(f"RI: {calc_rand(cluster_labels, true_labels)}")
print(f"ARI: {adjusted_rand_score(cluster_labels, true_labels)}")
print(f"SIL: {silhouette_score(embedding_data, cluster_labels)}")
print(f"N_clust: {len(list(set(cluster_labels)))}")

print("************************")
print("HDBSCAN-no-noise:")
print("************************")
cluster_labels = hdb_labels_no_noise
true_labels = df_no_noise[dataframe.label_column]
embedding_data = embedding_no_noise


print(f"RI: {calc_rand(cluster_labels, true_labels)}")
print(f"ARI: {adjusted_rand_score(cluster_labels, true_labels)}")
print(f"SIL: {silhouette_score(embedding_data, cluster_labels)}")
print(f"N_clust: {len(list(set(cluster_labels)))}")


hdb_colors = ['#d9d9d9'] + distinct_colors_22

umap_3Dplot(
    # x-axis coordinates
    x=df['UMAP1'],
    # y-axis coordinates
    y=df['UMAP2'],
    # z-axis coordinates
    z=df['UMAP3'],
    # labels (if available)
    scat_labels=hdb_labels,
    # sns.Palette color scheme name or list of colors
    mycolors=hdb_colors,
    # filename (with path) where figure will be saved.
    # Default: None -> figure not saved
    outname=None,
    # show legend if True else no legend
    showlegend=True
)

plt.show()

umap_3Dplot(
    # x-axis coordinates
    x=df['UMAP1'],
    # y-axis coordinates
    y=df['UMAP2'],
    # z-axis coordinates
    z=df['UMAP3'],
    # labels (if available)
    scat_labels=labels,
    # sns.Palette color scheme name or list of colors
    mycolors=distinct_colors_22,
    # filename (with path) where figure will be saved.
    # Default: None -> figure not saved
    outname=None,
    # show legend if True else no legend
    showlegend=True
)

plt.show()

n_specs = 5

clusters = sorted(
    list(
        set(hdb_labels)
    )
)

plt.figure(figsize=(
    n_specs * 2,
    len(clusters) * 2)
)

k = 1
specs = {}

for cluster in clusters:
    example = df[df['HDBSCAN'] == cluster].sample(
        n=n_specs,
        random_state=2204
    )
    specs = example[dataframe.input_column].values
    labels = example[dataframe.label_column].values

    i = 0

    plt.subplot(
        len(clusters),
        n_specs + 1,
        k
    )

    k += 1
    plt.axis('off')

    plt.text(
        0.5,
        0.4,
        'Cl. ' + str(cluster),
        fontsize=16
    )

    for spec, label in zip(specs, labels):
        plt.subplot(len(clusters), n_specs + 1, k)
        plt.imshow(
            spec,
            interpolation='nearest',
            aspect='equal',
            origin='lower'
        )
        plt.axis('off')
        title = str(label)
        plt.title(title)
        k += 1

plt.tight_layout()


# Column name of variable by which to analyze cluster content.
# Select any variable of interest in DF
analyze_by = dataframe.label_column

# Variable of cluster labels
cluster_labels = hdb_labels

cluster_labeltypes = sorted(
    list(
        set(cluster_labels)
    )
)

by_types = sorted(
    list(
        set(df[analyze_by])
    )
)

stats_tab = np.zeros(
    (
        len(cluster_labeltypes),
        len(by_types)
    )
)

for i, clusterlabel in enumerate(cluster_labeltypes):
    label_df = df.loc[cluster_labels == clusterlabel]

    for j, by_type in enumerate(by_types):
        stats_tab[i, j] = sum(label_df[analyze_by] == by_type)

stats_tab_df = pd.DataFrame(
    stats_tab,
    index=cluster_labeltypes,
    columns=by_types
)

# absolute vals
plt.figure(
    figsize=(
        int(len(cluster_labeltypes) / 2),
        int(len(by_types))
    )
)
ax = sns.heatmap(
    stats_tab_df,
    annot=True,
    cmap='viridis',
    fmt='.0f',
    cbar=False
)
ax.set_xlabel(analyze_by, fontsize=16)
ax.set_ylabel("Cluster label", fontsize=16)
ax.tick_params(labelsize=16)


# Rowsums
stats_tab_norm = np.zeros(
    (stats_tab.shape)
)
rowsums = np.sum(stats_tab, axis=1)

for i in range(stats_tab.shape[0]):
    stats_tab_norm[i, :] = stats_tab[i, :] / rowsums[i]

stats_tab_norm_df = pd.DataFrame(
    stats_tab_norm,
    index=cluster_labeltypes,
    columns=by_types
) * 100

plt.figure(
    figsize=(
        int(len(cluster_labeltypes) / 2),
        int(len(by_types))
    )
)

ax = sns.heatmap(
    stats_tab_norm_df,
    annot=True,
    cmap='viridis',
    fmt='.1f',
    cbar=False
)

ax.set_xlabel(analyze_by, fontsize=16)
ax.set_ylabel("Cluster label", fontsize=16)
ax.tick_params(labelsize=16)


# Colsums
stats_tab_norm = np.zeros(
    (stats_tab.shape)
)
colsums = np.sum(stats_tab, axis=0)

for i in range(stats_tab.shape[1]):
    stats_tab_norm[:, i] = stats_tab[:, i] / colsums[i]

stats_tab_norm_df = pd.DataFrame(
    stats_tab_norm,
    index=cluster_labeltypes,
    columns=by_types
) * 100

plt.figure(
    figsize=(
        int(len(cluster_labeltypes) / 2),
        int(len(by_types))
    )
)

ax = sns.heatmap(
    stats_tab_norm_df,
    annot=True,
    cmap='viridis',
    fmt='.1f',
    cbar=False
)

ax.set_xlabel(analyze_by, fontsize=16)
ax.set_ylabel("Cluster label", fontsize=16)
ax.tick_params(labelsize=16)

dataframe.close()
