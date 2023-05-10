"""
Network
-------

"""

import numpy as np

from datatype.dataset import Dataset
from datatype.network import Builder, NetworkHDBSCAN


def dataset() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    unique = dataframe.hdbscan_label_2d.unique()

    settings = {
        'ignore': [-1],
        'min_cluster_sample': 0,
        'min_connection': 0.05,
        'name': 'Adelaide\'s warbler',
    }

    coordinates = [
        dataframe.umap_x_2d,
        dataframe.umap_y_2d
    ]

    embedding = np.column_stack(coordinates)

    label = dataframe.hdbscan_label_2d.to_numpy()
    sequence = dataframe.sequence.to_numpy()

    network = NetworkHDBSCAN()

    network.builder = Builder()
    network.embedding = embedding
    network.label = label
    network.sequence = sequence
    network.settings = settings
    network.unique = unique

    component = network.build()

    figure = component.get('figure')

    # network.show()

    filename = 'network_hdbscan_dataset.png'

    network.save(
        figure=figure,
        filename=filename
    )


def individual() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    unique = dataframe.hdbscan_label_2d.unique()

    settings = {
        'ignore': [-1],
        'min_cluster_sample': 0,
        'min_connection': 0.05,
        'name': 'Adelaide\'s warbler',
    }

    folders = dataframe.folder.unique()

    for folder in folders:
        settings['name'] = folder

        individual = dataframe[dataframe.folder == folder]

        coordinates = [
            individual.umap_x_2d,
            individual.umap_y_2d
        ]

        embedding = np.column_stack(coordinates)

        label = individual.hdbscan_label_2d.to_numpy()
        sequence = individual.sequence.to_numpy()

        network = NetworkHDBSCAN()

        network.builder = Builder()
        network.embedding = embedding
        network.label = label
        network.sequence = sequence
        network.settings = settings
        network.unique = unique

        component = network.build()

        figure = component.get('figure')

        # network.show()

        filename = f"network_hdbscan_{folder}.png"

        network.save(
            figure=figure,
            filename=filename
        )


def main() -> None:
    # Create a network graph for the entire dataset
    dataset()

    # Create a network graph for each individual
    individual()


if __name__ == '__main__':
    main()
