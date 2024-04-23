from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from warbler.constant import SETTINGS
from warbler.datatype.dataset import Dataset
from warbler.datatype.transition import Builder, Transition
from warbler.datatype.settings import Settings


def dataset(save: bool = False) -> None:
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

    builder = Builder(
        embedding=embedding,
        sequence=sequence,
        settings=settings
    )

    transition = Transition(builder=builder)

    component = transition.build()

    if save:
        figure = component.get('figure')
        filename = 'transition_dataset.png'

        transition.save(
            figure=figure,
            filename=filename
        )
    else:
        transition.show()


def individual(save: bool = False) -> None:
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

        builder = Builder(
            embedding=embedding,
            sequence=sequence,
            settings=settings
        )

        transition = Transition(builder=builder)

        component = transition.build()

        if save:
            figure = component.get('figure')
            filename = f"transition_{folder}.png"

            transition.save(
                figure=figure,
                filename=filename
            )
        else:
            transition.show()


def main() -> None:
    plt.style.use('science')

    # Create a transition plot for the entire dataset
    dataset(save=False)

    # Create a transition plot for each individual
    individual(save=False)


if __name__ == '__main__':
    main()
