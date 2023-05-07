from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from joblib import Parallel, delayed
from matplotlib import gridspec
from nltk.metrics.distance import edit_distance
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform


def palplot(pal, size=1, ax=None):
    n = len(pal)

    if ax is None:
        _, ax = plt.subplots(
            1,
            1,
            figsize=(n * size, size)
        )

    ax.imshow(
        np.arange(n).reshape(1, n),
        cmap=matplotlib.colors.ListedColormap(list(pal)),
        interpolation='nearest',
        aspect='auto'
    )

    ax.set_xticks(np.arange(n) - 0.5)
    ax.set_yticks([-0.5, 0.5])

    # Ensure nice border between colors
    ax.set_xticklabels(
        ['' for _ in range(n)]
    )

    # The proper way to set no ticks
    ax.yaxis.set_major_locator(
        ticker.NullLocator()
    )

    ax.xaxis.set_major_locator(
        ticker.NullLocator()
    )


def song_barcode(onsets, offsets, labels, label_dict, label_pal_dict, resolution=0.01):
    begin = np.min(onsets)
    end = np.max(offsets)

    transition = (
        np.zeros(
            int(
                (end - begin) / resolution
            )
        )
        .astype('str')
        .astype('object')
    )

    for start, stop, label in zip(onsets, offsets, labels):
        transition[
            int((start - begin) / resolution) : int((stop - begin) / resolution)
        ] = label_dict[label]

    color = [
        label_pal_dict[i]
        if i in label_pal_dict else [1, 1, 1]
        for i in transition
    ]

    color = np.expand_dims(color, 1)

    return transition, color


def plot_sorted_barcodes(
    color_lists,
    trans_lists,
    max_list_len=600,
    seq_len=100,
    nex=50,
    n_jobs=4,
    figsize=(25, 6),
    ax=None,
):
    """
    Cluster and plot barcodes sorted

    max_list_len = 600 # maximum length to visualize
    seq_len = 100 # maximumim length to compute lev distance
    nex = 50 # only show up to NEX examples
    """

    # Subset dataset
    color_lists = color_lists[:nex]
    trans_lists = trans_lists[:nex]

    # get length of lists
    list_lens = [len(i) for i in trans_lists]

    # set max list length
    if max_list_len is None:
        max_list_len = np.max(list_lens)

    # make a matrix for color representations of syllables
    color_item = np.ones((max_list_len, len(list_lens), 3))

    for li, _list in enumerate(color_lists):
        color_item[: len(_list), li, :] = np.squeeze(_list[:max_list_len])

    color_items = color_item.swapaxes(0, 1)

    # make a list of symbols padded to equal length
    trans_lists = np.array(trans_lists, dtype='object')

    cut_lists = [
        list(i[:seq_len].astype('str'))
        if len(i) >= seq_len
        else list(i) + list(np.zeros(seq_len - len(i)).astype('str'))
        for i in trans_lists
    ]

    cut_lists = [
        ''.join(np.array(i).astype('str'))
        for i in cut_lists
    ]

    # Create a distance matrix
    dist = np.zeros(
        (
            len(cut_lists),
            len(cut_lists)
        )
    )

    items = [
        (i, j) for i in range(1, len(cut_lists))
        for j in range(0, i)
    ]

    distances = Parallel(n_jobs=n_jobs)(
        delayed(edit_distance)(cut_lists[i], cut_lists[j])
        for i, j in items
    )

    for distance, (i, j) in zip(distances, items):
        dist[i, j] = distance
        dist[j, i] = distance

    # Hierarchical clustering
    dists = squareform(dist)
    linkage_matrix = linkage(dists, 'single')

    # Make plot
    if ax == None:
        fig = plt.figure(figsize=figsize)

        gs = gridspec.GridSpec(
            1,
            2,
            width_ratios=[1, 10],
            wspace=0,
            hspace=0
        )

        ax0 = plt.subplot(gs[0])
        ax = plt.subplot(gs[1])

        dn = dendrogram(
            linkage_matrix,
            p=6,
            truncate_mode='none',
            get_leaves=True,
            orientation='left',
            no_labels=True,
            link_color_func=lambda k: 'k',
            ax=ax0,
            show_contracted=False,
        )

        ax0.axis('off')
    else:
        dn = dendrogram(
            linkage_matrix,
            p=6,
            truncate_mode='none',
            get_leaves=True,
            orientation='left',
            no_labels=True,
            link_color_func=lambda k: 'k',
            # ax=ax0,
            show_contracted=False,
            no_plot=True,
        )

    ax.imshow(
        color_item.swapaxes(0, 1)[np.array(dn['leaves'])],
        aspect='auto',
        interpolation=None,
        origin='lower',
    )

    ax.axis('off')

    return ax
