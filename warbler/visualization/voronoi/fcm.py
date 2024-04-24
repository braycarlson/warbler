from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from warbler.constant import SETTINGS
from warbler.datatype.dataset import Dataset
from warbler.datatype.settings import Settings
from warbler.datatype.voronoi import Builder, VoronoiFCM


def dataset(save: bool = False) -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    # Load default settings
    path = SETTINGS.joinpath('voronoi.json')
    settings = Settings.from_file(path)

    settings.cluster = 'Fuzzy C-Means Clustering'
    settings.is_label = False

    unique = dataframe.fcm_label_2d.unique()

    by = ['duration']

    ascending = [False]

    dataframe = dataframe.sort_values(
        ascending=ascending,
        by=by
    )

    coordinates = [
        dataframe.umap_x_2d,
        dataframe.umap_y_2d
    ]

    embedding = np.column_stack(coordinates)

    spectrogram = dataframe.filter_array.to_numpy()
    label = dataframe.fcm_label_2d.to_numpy()

    builder = Builder(
        embedding=embedding,
        label=label,
        settings=settings,
        spectrogram=~spectrogram,
        unique=unique
    )

    voronoi = VoronoiFCM(builder=builder)

    component = voronoi.build()

    if save:
        figure = component.get('figure')
        filename = 'voronoi_fcm_dataset.png'

        voronoi.save(
            figure=figure,
            filename=filename
        )
    else:
        voronoi.show()


def individual(save: bool = False) -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    # Load default settings
    path = SETTINGS.joinpath('voronoi.json')
    settings = Settings.from_file(path)

    settings.cluster = 'Fuzzy C-Means Clustering'
    settings.is_label = False

    unique = dataframe.fcm_label_2d.unique()

    folders = dataframe.folder.unique()

    for folder in folders:
        settings['name'] = folder

        individual = dataframe[dataframe.folder == folder]

        by = ['duration']

        ascending = [False]

        individual = individual.sort_values(
            ascending=ascending,
            by=by
        )

        coordinates = [
            individual.umap_x_2d,
            individual.umap_y_2d
        ]

        embedding = np.column_stack(coordinates)

        spectrogram = individual.filter_array.to_numpy()
        label = individual.fcm_label_2d.to_numpy()

        builder = Builder(
            embedding=embedding,
            label=label,
            settings=settings,
            spectrogram=~spectrogram,
            unique=unique
        )

        voronoi = VoronoiFCM(builder=builder)

        component = voronoi.build()

        if save:
            figure = component.get('figure')
            filename = f"voronoi_fcm_{folder}.png"

            voronoi.save(
                figure=figure,
                filename=filename
            )
        else:
            voronoi.show()


def main() -> None:
    plt.style.use('science')

    # Create a voronoi plot for the entire dataset
    dataset(save=False)

    # # Create a voronoi plot for each individual
    # individual(save=False)


if __name__ == '__main__':
    main()
