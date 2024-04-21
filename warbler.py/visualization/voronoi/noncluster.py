"""
Voronoi: Uncluster
------------------

"""

from __future__ import annotations

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

    settings.cluster = 'Dimensionality Reduction'
    settings.is_color = False
    settings.is_legend = False

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

    voronoi = VoronoiFCM()

    voronoi.builder = Builder()
    voronoi.embedding = embedding
    voronoi.spectrogram = ~spectrogram
    voronoi.settings = settings

    component = voronoi.build()

    figure = component.get('figure')

    # voronoi.show()

    filename = 'voronoi_noncluster_dataset.png'

    voronoi.save(
        figure=figure,
        filename=filename
    )


def main() -> None:
    # Create a voronoi plot for the entire dataset
    dataset()


if __name__ == '__main__':
    main()
