"""
Transition
----------

"""

import matplotlib.pyplot as plt
import numpy as np

from datatype.dataset import Dataset
from datatype.transition import Builder, Transition


def dataset() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    settings = {
        'colorline': {
            'alpha': 0.25,
            'cmap': plt.get_cmap('cubehelix'),
            'linewidth': 0.5,
            'norm': plt.Normalize(0.0, 1.0),
        },
        'figure': {
            'figsize': (10, 10)
        },
        'name': 'Adelaide\'s warbler',
        'padding': 0.1
    }

    coordinates = [
        dataframe.umap_x_2d,
        dataframe.umap_y_2d
    ]

    embedding = np.column_stack(coordinates)
    sequence = dataframe.sequence.tolist()

    transition = Transition()

    transition.builder = Builder()
    transition.embedding = embedding
    transition.sequence = sequence
    transition.settings = settings

    component = transition.build()

    figure = component.get('figure')

    transition.show()

    filename = 'transition_dataset.png'

    transition.save(
        figure=figure,
        filename=filename
    )


def individual() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    unique = dataframe.folder.unique()

    settings = {
        'colorline': {
            'alpha': 0.25,
            'cmap': plt.get_cmap('cubehelix'),
            'linewidth': 6,
            'norm': plt.Normalize(0.0, 1.0),
        },
        'figure': {
            'figsize': (10, 10)
        },
        'name': 'Adelaide\'s warbler',
        'padding': 0.1
    }

    for folder in unique:
        settings['name'] = folder

        individual = dataframe[dataframe.folder == folder]

        coordinates = [
            individual.umap_x_2d,
            individual.umap_y_2d
        ]

        embedding = np.column_stack(coordinates)
        sequence = individual.sequence.tolist()

        transition = Transition()

        transition.builder = Builder()
        transition.embedding = embedding
        transition.sequence = sequence
        transition.settings = settings

        component = transition.build()

        figure = component.get('figure')

        # transition.show()

        filename = f"transition_{folder}.png"

        transition.save(
            figure=figure,
            filename=filename
        )


def main() -> None:
    # Create a transition plot for the entire dataset
    dataset()

    # Create a transition plot for each individual
    individual()


if __name__ == '__main__':
    main()
