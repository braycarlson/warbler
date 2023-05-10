"""
HDBSCAN
-------

"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from constant import PROJECTION
from datatype.dataset import Dataset
from hdbscan import HDBSCAN


def main() -> None:
    PROJECTION.mkdir(parents=True, exist_ok=True)

    dataset = Dataset('segment')
    dataframe = dataset.load()

    coordinates = (
        dataframe.umap_x_2d,
        dataframe.umap_y_2d
    )

    embedding = np.column_stack(coordinates)
    length = len(embedding)

    # The smallest size we would expect a cluster to be
    min_cluster_size = int(length * 0.01)

    cluster = HDBSCAN(
        cluster_selection_method='leaf',
        gen_min_span_tree=True,
        min_cluster_size=min_cluster_size
    ).fit(embedding)

    # Condensed tree
    selection_palette = sns.color_palette('deep', min_cluster_size)

    cluster.condensed_tree_.plot(
        select_clusters=True,
        selection_palette=selection_palette
    )

    file = PROJECTION.joinpath('hdbscan_condensed_tree_2d.png')

    plt.title(
        'HDBSCAN: Condensed Tree (2D)',
        fontweight=600,
        fontsize=12,
        pad=25
    )

    plt.savefig(
        file,
        bbox_inches='tight',
        dpi=300,
        format='png'
    )

    # Minimum spanning tree
    cluster.minimum_spanning_tree_.plot(
        edge_alpha=0.6,
        edge_cmap='viridis',
        edge_linewidth=2,
        node_alpha=0.1,
        node_size=1.0
    )

    file = PROJECTION.joinpath('hdbscan_minimum_spanning_tree_2d.png')

    plt.title(
        'HDBSCAN: Minimum Spanning Tree (2D)',
        fontweight=600,
        fontsize=12,
        pad=25
    )

    plt.savefig(
        file,
        bbox_inches='tight',
        dpi=300,
        format='png'
    )

    # # Single linkage tree
    # cluster.single_linkage_tree_.plot()

    # file = PROJECTION.joinpath('hdbscan_single_linkage_tree_2d.png')

    # plt.title(
    #     'HDBSCAN: Single Linkage Tree (2D)',
    #     fontweight=600,
    #     fontsize=12,
    #     pad=25
    # )

    # plt.savefig(
    #     file,
    #     bbox_inches='tight',
    #     dpi=300,
    #     format='png'
    # )

    dataframe['hdbscan_label_2d'] = cluster.labels_

    dataset.save(dataframe)


if __name__ == '__main__':
    main()
