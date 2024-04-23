from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pickle
import scienceplots

from warbler.constant import SETTINGS
from warbler.datatype.dataset import Dataset
from warbler.datatype.settings import Settings
from warbler.datatype.voronoi import Builder, VoronoiHDBSCAN


def dataset(save: bool = False) -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    # Deserialize the filter_array
    dataframe['filter_array'] = dataframe['filter_array'].apply(
        lambda x: pickle.loads(
            bytes.fromhex(x)
        )
    )

    # Load default settings
    path = SETTINGS.joinpath('voronoi.json')
    settings = Settings.from_file(path)

    settings.cluster = 'HDBSCAN Clustering'

    unique = dataframe.hdbscan_label_2d.unique()

    # Mask the "noise"
    dataframe = (
        dataframe[dataframe.hdbscan_label_2d > -1]
        .reset_index(drop=True)
    )

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
    label = dataframe.hdbscan_label_2d.to_numpy()

    builder = Builder(
        embedding=embedding,
        label=label,
        settings=settings,
        spectrogram=~spectrogram,
        unique=unique
    )

    voronoi = VoronoiHDBSCAN(builder=builder)

    component = voronoi.build()

    if save:
        figure = component.get('figure')
        filename = 'voronoi_hdbscan_dataset.png'

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
        label = individual.hdbscan_label_2d.to_numpy()

        builder = Builder(
            embedding=embedding,
            label=label,
            settings=settings,
            spectrogram=~spectrogram,
            unique=unique
        )

        voronoi = VoronoiHDBSCAN(builder=builder)

        component = voronoi.build()

        if save:
            figure = component.get('figure')
            filename = f"voronoi_hdbscan_{folder}.png"

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
