from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from warbler.constant import SETTINGS
from warbler.datatype.dataset import Dataset
from warbler.datatype.settings import Settings
from warbler.datatype.scatter import Builder, ScatterHDBSCAN


def dataset(save: bool = False) -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    # Load default settings
    path = SETTINGS.joinpath('scatter.json')
    settings = Settings.from_file(path)

    # Change the default to accomodate a larger dataset
    settings['scatter']['alpha'] = 0.50
    settings['scatter']['s'] = 10

    unique = dataframe.hdbscan_label_2d.unique()

    # Mask the "noise"
    dataframe = (
        dataframe[dataframe.hdbscan_label_2d > -1]
        .reset_index(drop=True)
    )

    coordinates = [
        dataframe.umap_x_2d,
        dataframe.umap_y_2d
    ]

    embedding = np.column_stack(coordinates)
    label = dataframe.hdbscan_label_2d.to_numpy()

    builder = Builder(
        embedding=embedding,
        label=label,
        settings=settings,
        unique=unique
    )

    scatter = ScatterHDBSCAN(builder=builder)

    component = scatter.build()

    if save:
        figure = component.get('figure')
        filename = 'scatter_hdbscan_dataset.png'

        scatter.save(
            figure=figure,
            filename=filename
        )
    else:
        scatter.show()


def individual(save: bool = False) -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    # Load default settings
    path = SETTINGS.joinpath('scatter.json')
    settings = Settings.from_file(path)

    # Change the default to accomodate a smaller dataset
    settings['scatter']['alpha'] = 0.50
    settings['scatter']['s'] = 10

    unique = dataframe.hdbscan_label_2d.unique()

    # Mask the "noise"
    dataframe = (
        dataframe[dataframe.hdbscan_label_2d > -1]
        .reset_index(drop=True)
    )

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

        builder = Builder(
            embedding=embedding,
            label=label,
            settings=settings,
            unique=unique
        )

        scatter = ScatterHDBSCAN(builder=builder)

        component = scatter.build()

        if save:
            figure = component.get('figure')
            filename = f"scatter_hdbscan_{folder}.png"

            scatter.save(
                figure=figure,
                filename=filename
            )
        else:
            scatter.show()


def main() -> None:
    plt.style.use('science')

    # Create a scatter plot for the entire dataset
    dataset(save=False)

    # Create a scatter plot for each individual
    individual(save=False)


if __name__ == '__main__':
    main()
