import matplotlib.pyplot as plt
import numpy as np

from constant import PROJECTION
from datatype.dataset import Dataset
from datatype.network import (
    Builder,
    Network,
    plot_network_graph
)


def main():
    dataset = Dataset('segment')
    dataframe = dataset.load()

    unique = dataframe.hdbscan_label_2d.unique()

    settings = {
        'ignore': [-1],
        'min_cluster_sample': 0,
        'min_connection': 0.05,
    }

    folders = dataframe.folder.unique()

    for folder in folders[:1]:
        individual = dataframe[dataframe.folder == folder]

        coordinates = [
            individual.umap_x_2d,
            individual.umap_y_2d
        ]

        embedding = np.column_stack(coordinates)

        label = individual.hdbscan_label_2d.to_numpy()
        sequence = individual.sequence.to_numpy()

        network = Network()

        network.builder = Builder()
        network.embedding = embedding
        network.label = label
        network.sequence = sequence
        network.settings = settings
        network.unique = unique

        network.construct()
        network.show()

    # dataset = Dataset('segment')
    # dataframe = dataset.load()

    # unique = dataframe.folder.unique()

    # original = dataframe.hdbscan_label_2d.unique()
    # n_colors = len(original)

    # for folder in unique[:1]:
    #     print(f"Processing: {folder}")

    #     individual = dataframe[dataframe.folder == folder]

    #     coordinates = [
    #         individual.umap_x_2d,
    #         individual.umap_y_2d
    #     ]

    #     embedding = np.column_stack(coordinates)

    #     sequence = individual.sequence.to_numpy()
    #     label = individual.hdbscan_label_2d.to_numpy()

    #     figsize = (9, 8)
    #     _, ax = plt.subplots(figsize=figsize)

    #     plot_network_graph(
    #         label,
    #         embedding,
    #         sequence,
    #         ax=ax,
    #         color_palette='tab20',
    #         n_colors=n_colors,
    #         original=original
    #     )

    #     ax.axis('off')

    #     title = f"Network Graph of {folder}"

    #     ax.set_title(
    #         title,
    #         fontsize=18,
    #         pad=25
    #     )

    #     plt.show()


if __name__ == '__main__':
    main()
