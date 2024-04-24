from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import seaborn as sns

from hdbscan import HDBSCAN
from warbler.constant import PROJECTION
from warbler.datatype.dataset import Dataset


def main() -> None:
    plt.style.use('science')

    dataset = Dataset('segment')
    dataframe = dataset.load()

    coordinates = (
        dataframe.umap_x_3d,
        dataframe.umap_y_3d,
        dataframe.umap_z_3d
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

    file = PROJECTION.joinpath('hdbscan_condensed_tree_3d.png')

    plt.title(
        'HDBSCAN: Condensed Tree (3D)',
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

    plt.close()

    # Minimum spanning tree
    cluster.minimum_spanning_tree_.plot(
        edge_alpha=0.6,
        edge_cmap='viridis',
        edge_linewidth=2,
        node_alpha=0.1,
        node_size=1.0
    )

    file = PROJECTION.joinpath('hdbscan_minimum_spanning_tree_3d.png')

    plt.title(
        'HDBSCAN: Minimum Spanning Tree (3D)',
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

    plt.close()

    # # Single linkage tree
    # cluster.single_linkage_tree_.plot()

    # file = PROJECTION.joinpath('hdbscan_single_linkage_tree_3d.png')

    # plt.title(
    #     'HDBSCAN: Single Linkage Tree (3D)',
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

    # plt.close()

    dataframe['hdbscan_label_3d'] = cluster.labels_

    dataset.save(dataframe)


if __name__ == '__main__':
    main()
