from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from warbler.constant import SETTINGS
from warbler.datatype.dataset import Dataset
from warbler.datatype.settings import Settings
from warbler.datatype.network import Builder, NetworkFCM


def dataset(save: bool = False) -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    # Load default settings
    path = SETTINGS.joinpath('network.json')
    settings = Settings.from_file(path)

    unique = dataframe.fcm_label_2d.unique()

    coordinates = [
        dataframe.umap_x_2d,
        dataframe.umap_y_2d
    ]

    embedding = np.column_stack(coordinates)

    label = dataframe.fcm_label_2d.to_numpy()
    sequence = dataframe.sequence.to_numpy()

    builder = Builder(
        embedding=embedding,
        label=label,
        sequence=sequence,
        settings=settings,
        unique=unique
    )

    network = NetworkFCM(builder=builder)

    component = network.build()

    if save:
        figure = component.get('figure')
        filename = 'network_fcm_dataset.png'

        network.save(
            figure=figure,
            filename=filename
        )
    else:
        network.show()


def individual(save: bool = False) -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    # Load default settings
    path = SETTINGS.joinpath('network.json')
    settings = Settings.from_file(path)

    unique = dataframe.fcm_label_2d.unique()

    folders = dataframe.folder.unique()

    for folder in folders:
        settings['name'] = folder

        individual = dataframe[dataframe.folder == folder]

        coordinates = [
            individual.umap_x_2d,
            individual.umap_y_2d
        ]

        embedding = np.column_stack(coordinates)

        label = individual.fcm_label_2d.to_numpy()
        sequence = individual.sequence.to_numpy()

        builder = Builder(
            embedding=embedding,
            label=label,
            sequence=sequence,
            settings=settings,
            unique=unique
        )

        network = NetworkFCM(builder=builder)

        component = network.build()

        if save:
            figure = component.get('figure')
            filename = f"network_fcm_{folder}.png"

            network.save(
                figure=figure,
                filename=filename
            )
        else:
            network.show()


def main() -> None:
    plt.style.use('science')

    # Create a network graph for the entire dataset
    dataset(save=False)

    # Create a network graph for each individual
    individual(save=False)


if __name__ == '__main__':
    main()
