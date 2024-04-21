"""
Voronoi: Mean
-------------

"""

from __future__ import annotations

import numpy as np

from constant import SETTINGS
from datatype.dataset import Dataset
from datatype.gradient import (
    GradientBuilder,
    VoronoiGradient
)
from datatype.settings import Settings


def dataset() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    # Load default settings
    path = SETTINGS.joinpath('gradient.json')
    settings = Settings.from_file(path)

    settings.cluster = 'Mean Frequency Clustering'
    settings.colorbar['label'] = 'Frequency (Hz)'
    settings.is_diagonal = False
    settings.y = 'y'

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
    label = dataframe['mean'].to_numpy()

    voronoi = VoronoiGradient()

    voronoi.builder = GradientBuilder()
    voronoi.embedding = embedding
    voronoi.label = label
    voronoi.spectrogram = ~spectrogram
    voronoi.settings = settings

    component = voronoi.build()

    figure = component.get('figure')

    # voronoi.show()

    filename = 'voronoi_mean_dataset.png'

    voronoi.save(
        figure=figure,
        filename=filename
    )


def main() -> None:
    # Create a voronoi plot for the entire dataset
    dataset()


if __name__ == '__main__':
    main()
