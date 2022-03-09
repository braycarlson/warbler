import hdbscan
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns

from functions.plot_functions import umap_3Dplot
from parameters import Parameters
from path import CLUSTER, DATA
from sklearn.metrics.cluster import adjusted_rand_score
from sklearn.metrics import silhouette_score


path = CLUSTER.joinpath('parameters.json')
parameters = Parameters(path)

path = CLUSTER.joinpath('dataframe.json')
dataframe = Parameters(path)


# Import functions for calculating unadjusted Rand score
# Resource: https://stackoverflow.com/questions/49586742/rand-index-function-clustering-performance-evaluation

def rand_index_score(clusters, classes):
    tp_plus_fp = scipy.special.comb(np.bincount(clusters), 2).sum()
    tp_plus_fn = scipy.special.comb(np.bincount(classes), 2).sum()
    A = np.c_[(clusters, classes)]

    tp = sum(scipy.special.comb(np.bincount(
                    A[A[:, 0] == i, 1]
            ), 2).sum() for i in set(clusters)
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

df = pd.read_pickle(
    DATA.joinpath('df_umap.pkl')
)

UMAP_COLS = [x for x in df.columns if 'UMAP' in x]

labels = df[dataframe.label_column]
embedding = np.asarray(df[UMAP_COLS])

print(f"Found {len(UMAP_COLS)} UMAP columns. Label in column: {dataframe.label_column}")

# clustering
HDBSCAN = hdbscan.HDBSCAN(
    min_cluster_size=int(0.01 * embedding.shape[0]),
    cluster_selection_method='leaf'
).fit(embedding)

hdb_labels = HDBSCAN.labels_
df['HDBSCAN'] = hdb_labels

hdb_labels_no_noise = hdb_labels.copy()

assigned = np.full(
    (len(hdb_labels_no_noise), ),
    False
)

assigned[np.where(hdb_labels_no_noise != -1)[0]] = True

hdb_labels_no_noise = hdb_labels_no_noise[assigned]
df_no_noise = df.loc[assigned]
embedding_no_noise = embedding[assigned, :]

print('HDBSCAN:')
cluster_labels = hdb_labels
true_labels = df[dataframe.label_column]
embedding_data = embedding

print("RI: ", calc_rand(cluster_labels, true_labels))
print("ARI: ", adjusted_rand_score(cluster_labels, true_labels))
print("SIL: ", silhouette_score(embedding_data, cluster_labels))
print("N_clust: ", len(list(set(cluster_labels))))


print('HDBSCAN-no-noise:')
cluster_labels = hdb_labels_no_noise
true_labels = df_no_noise[dataframe.label_column]
embedding_data = embedding_no_noise

print("RI: ", calc_rand(cluster_labels, true_labels))
print("ARI: ", adjusted_rand_score(cluster_labels, true_labels))
print("SIL: ", silhouette_score(embedding_data, cluster_labels))
print("N_clust: ", len(list(set(cluster_labels))))


hdb_colors = ['#d9d9d9'] + distinct_colors_22

umap_3Dplot(
    x=df['UMAP1'],
    y=df['UMAP2'],
    z=df['UMAP3'],
    scat_labels=hdb_labels,
    mycolors=hdb_colors,
    outname=None,
    showlegend=True
)

umap_3Dplot(
    x=df['UMAP1'],
    y=df['UMAP2'],
    z=df['UMAP3'],
    scat_labels=labels,
    mycolors=distinct_colors_22,
    outname=None,
    showlegend=True
)

n_specs = 5

clusters = sorted(
    list(
        set(hdb_labels)
    )
)

plt.figure(figsize=(n_specs * 2, len(clusters) * 2))

k = 1
specs = {}

for cluster in clusters:
    example = df[df['HDBSCAN'] == cluster].sample(n=n_specs, random_state=2204)
    specs = example[dataframe.input_column].values
    labels = example[dataframe.label_column].values
    i = 0

    plt.subplot(len(clusters), n_specs + 1, k)
    k = k + 1
    plt.axis('off')
    plt.text(0.5, 0.4, 'Cl. ' + str(cluster), fontsize=16)

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

analyze_by = dataframe.label_column
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


# rowsums
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


# colsums
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

plt.show()
