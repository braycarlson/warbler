"""
Voronoi
-------

"""

import numpy as np

from datatype.dataset import Dataset
from datatype.voronoi import Builder, Voronoi


def dataset() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    unique = dataframe.hdbscan_label_2d.unique()

    # Mask the "noise"
    dataframe = (
        dataframe[dataframe.hdbscan_label_2d > -1]
        .reset_index(drop=True)
    )

    by = ['duration']

    ascending = [False]

    dataframe.sort_values(
        ascending=ascending,
        by=by,
        inplace=True
    )

    coordinates = [
        dataframe.umap_x_2d,
        dataframe.umap_y_2d
    ]

    embedding = np.column_stack(coordinates)

    spectrogram = dataframe.original_array.to_numpy()
    label = dataframe.hdbscan_label_2d.to_numpy()

    voronoi = Voronoi()

    voronoi.builder = Builder()
    voronoi.embedding = embedding
    voronoi.label = label
    voronoi.spectrogram = ~spectrogram
    voronoi.unique = unique

    component = voronoi.build()

    figure = component.collection.get('figure')

    voronoi.show()

    filename = 'voronoi_dataset.png'

    voronoi.save(
        figure=figure,
        filename=filename
    )


def individual() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    settings = {}

    folders = dataframe.folder.unique()
    unique = dataframe.hdbscan_label_2d.unique()

    for folder in folders:
        settings['name'] = folder

        individual = dataframe[dataframe.folder == folder]

        # Mask the "noise"
        individual = (
            individual[individual.hdbscan_label_2d > -1]
            .reset_index(drop=True)
        )

        by = ['duration']

        ascending = [False]

        individual.sort_values(
            ascending=ascending,
            by=by,
            inplace=True
        )

        coordinates = [
            individual.umap_x_2d,
            individual.umap_y_2d
        ]

        embedding = np.column_stack(coordinates)

        spectrogram = individual.original_array.to_numpy()
        label = individual.hdbscan_label_2d.to_numpy()

        voronoi = Voronoi()

        voronoi.builder = Builder()
        voronoi.embedding = embedding
        voronoi.label = label
        voronoi.settings = settings
        voronoi.spectrogram = ~spectrogram
        voronoi.unique = unique

        component = voronoi.build()

        figure = component.collection.get('figure')

        voronoi.show()

        filename = f"voronoi_{folder}.png"

        voronoi.save(
            figure=figure,
            filename=filename
        )


def main():
    # Create a voronoi plot for the entire dataset
    dataset()

    # Create a voronoi plot for each individual
    individual()


if __name__ == '__main__':
    main()
