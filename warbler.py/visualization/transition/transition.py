"""
Transition
----------

"""

import matplotlib.pyplot as plt
import numpy as np

from constant import SETTINGS
from datatype.dataset import Dataset
from datatype.transition import Builder, Transition
from datatype.settings import Settings


def dataset() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    # Load default settings
    path = SETTINGS.joinpath('transition.json')
    settings = Settings.from_file(path)

    # Change the default to accomodate a larger dataset
    settings['colorline']['alpha'] = 0.25
    settings['colorline']['cmap'] = plt.get_cmap('cubehelix')
    settings['colorline']['linewidth'] = 0.5
    settings['colorline']['norm'] = plt.Normalize(0.0, 1.0)

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

    # Load default settings
    path = SETTINGS.joinpath('transition.json')
    settings = Settings.from_file(path)

    # Change the default to accomodate a smaller dataset
    settings['colorline']['alpha'] = 0.25
    settings['colorline']['cmap'] = plt.get_cmap('cubehelix')
    settings['colorline']['linewidth'] = 6
    settings['colorline']['norm'] = plt.Normalize(0.0, 1.0)

    unique = dataframe.folder.unique()

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

        transition.show()

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
