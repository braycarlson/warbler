"""
Network: FCM
------------

"""

import numpy as np

from constant import SETTINGS
from datatype.dataset import Dataset
from datatype.settings import Settings
from datatype.network import Builder, NetworkHDBSCAN


def dataset() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    # Load default settings
    path = SETTINGS.joinpath('network.json')
    settings = Settings.from_file(path)

    unique = dataframe.hdbscan_label_2d.unique()

    # # Mask the "noise"
    # dataframe = (
    #     dataframe[dataframe.hdbscan_label_2d > -1]
    #     .reset_index(drop=True)
    # )

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

    network.show()

    filename = 'network_hdbscan_dataset.png'

    network.save(
        figure=figure,
        filename=filename
    )


def individual() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    # Load default settings
    path = SETTINGS.joinpath('network.json')
    settings = Settings.from_file(path)

    unique = dataframe.hdbscan_label_2d.unique()

    # # Mask the "noise"
    # dataframe = (
    #     dataframe[dataframe.hdbscan_label_2d > -1]
    #     .reset_index(drop=True)
    # )

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

        network.show()

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
