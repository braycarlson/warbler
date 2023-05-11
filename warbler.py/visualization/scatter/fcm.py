"""
Scatter: FCM
------------

"""

import numpy as np

from constant import SETTINGS
from datatype.dataset import Dataset
from datatype.scatter import Builder, ScatterFCM
from datatype.settings import Settings


def dataset() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    # Load default settings
    path = SETTINGS.joinpath('scatter.json')
    settings = Settings.from_file(path)

    # Change the default to accomodate a larger dataset
    settings['scatter']['alpha'] = 0.50
    settings['scatter']['s'] = 10

    unique = dataframe.fcm_label_2d.unique()

    coordinates = [
        dataframe.umap_x_2d,
        dataframe.umap_y_2d
    ]

    embedding = np.column_stack(coordinates)
    label = dataframe.fcm_label_2d.to_numpy()

    scatter = ScatterFCM()

    scatter.builder = Builder()
    scatter.embedding = embedding
    scatter.label = label
    scatter.settings = settings
    scatter.unique = unique

    component = scatter.build()

    figure = component.get('figure')

    scatter.show()

    filename = 'scatter_fcm_dataset.png'

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

    folders = dataframe.folder.unique()
    unique = dataframe.fcm_label_2d.unique()

    for folder in folders:
        settings['name'] = folder

        individual = dataframe[dataframe.folder == folder]

        coordinates = [
            individual.umap_x_2d,
            individual.umap_y_2d
        ]

        embedding = np.column_stack(coordinates)
        label = individual.fcm_label_2d.to_numpy()

        scatter = ScatterFCM()

        scatter.builder = Builder()
        scatter.embedding = embedding
        scatter.label = label
        scatter.settings = settings
        scatter.unique = unique

        component = scatter.build()

        figure = component.get('figure')

        scatter.show()

        filename = f"scatter_fcm_{folder}.png"

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
