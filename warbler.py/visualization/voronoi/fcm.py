"""
Voronoi: FCM
------------

"""

import numpy as np

from constant import SETTINGS
from datatype.dataset import Dataset
from datatype.settings import Settings
from datatype.voronoi import Builder, VoronoiFCM


def dataset() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    # Load default settings
    path = SETTINGS.joinpath('voronoi.json')
    settings = Settings.from_file(path)

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

    spectrogram = dataframe.original_array.to_numpy()
    label = dataframe.fcm_label_2d.to_numpy()

    voronoi = VoronoiFCM()

    voronoi.builder = Builder()
    voronoi.embedding = embedding
    voronoi.label = label
    voronoi.spectrogram = ~spectrogram
    voronoi.settings = settings
    voronoi.unique = unique

    component = voronoi.build()

    figure = component.get('figure')

    voronoi.show()

    filename = 'voronoi_fcm_dataset.png'

    voronoi.save(
        figure=figure,
        filename=filename
    )


def individual() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    # Load default settings
    path = SETTINGS.joinpath('voronoi.json')
    settings = Settings.from_file(path)

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

        spectrogram = individual.original_array.to_numpy()
        label = individual.fcm_label_2d.to_numpy()

        voronoi = VoronoiFCM()

        voronoi.builder = Builder()
        voronoi.embedding = embedding
        voronoi.label = label
        voronoi.settings = settings
        voronoi.spectrogram = ~spectrogram
        voronoi.unique = unique

        component = voronoi.build()

        figure = component.get('figure')

        voronoi.show()

        filename = f"voronoi_fcm_{folder}.png"

        voronoi.save(
            figure=figure,
            filename=filename
        )


def main() -> None:
    # Create a voronoi plot for the entire dataset
    dataset()

    # Create a voronoi plot for each individual
    individual()


if __name__ == '__main__':
    main()
