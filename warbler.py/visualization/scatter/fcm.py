"""
Scatter: FCM
------------

"""

import numpy as np

from datatype.dataset import Dataset
from datatype.scatter import Builder, ScatterFCM


def dataset() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    unique = dataframe.fcm_label_2d.unique()

    coordinates = [
        dataframe.umap_x_2d,
        dataframe.umap_y_2d
    ]

    embedding = np.column_stack(coordinates)
    label = dataframe.fcm_label_2d.to_numpy()

    settings = {
        'scatter': {
            'alpha': 0.50,
            's': 10
        },
        'name': 'Adelaide\'s warbler',
    }

    scatter = ScatterFCM()

    scatter.builder = Builder()
    scatter.embedding = embedding
    scatter.label = label
    scatter.settings = settings
    scatter.unique = unique

    component = scatter.build()

    figure = component.get('figure')

    # scatter.show()

    filename = 'scatter_fcm_dataset.png'

    scatter.save(
        figure=figure,
        filename=filename
    )


def individual() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    settings = {
        'scatter': {
            'alpha': 0.50,
            's': 10
        },
        'name': 'Adelaide\'s warbler',
    }

    folders = dataframe.folder.unique()
    unique = dataframe.fcm_label_2d.unique()

    for folder in folders:
        settings['scatter']['alpha'] = 0.75
        settings['scatter']['s'] = 25
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

        # scatter.show()

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
