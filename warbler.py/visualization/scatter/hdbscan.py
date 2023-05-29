"""
Scatter: HDBSCAN
----------------

"""

from __future__ import annotations

import numpy as np

from constant import SETTINGS
from datatype.dataset import Dataset
from datatype.settings import Settings
from datatype.scatter import Builder, ScatterHDBSCAN


def dataset() -> None:
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

    scatter = ScatterHDBSCAN()

    scatter.builder = Builder()
    scatter.embedding = embedding
    scatter.label = label
    scatter.settings = settings
    scatter.unique = unique

    component = scatter.build()

    figure = component.get('figure')

    scatter.show()

    filename = 'scatter_hdbscan_dataset.png'

    scatter.save(
        figure=figure,
        filename=filename
    )


def individual() -> None:
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

        scatter = ScatterHDBSCAN()

        scatter.builder = Builder()
        scatter.embedding = embedding
        scatter.label = label
        scatter.settings = settings
        scatter.unique = unique

        component = scatter.build()

        figure = component.get('figure')

        scatter.show()

        filename = f"scatter_hdbscan_{folder}.png"

        scatter.save(
            figure=figure,
            filename=filename
        )


def main() -> None:
    # Create a scatter plot for the entire dataset
    dataset()

    # Create a scatter plot for each individual
    individual()


if __name__ == '__main__':
    main()
