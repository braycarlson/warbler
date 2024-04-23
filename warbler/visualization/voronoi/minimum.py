from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pickle
import scienceplots

from warbler.constant import SETTINGS
from warbler.datatype.dataset import Dataset
from warbler.datatype.gradient import (
    GradientBuilder,
    VoronoiGradient
)
from warbler.datatype.settings import Settings


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
    path = SETTINGS.joinpath('gradient.json')
    settings = Settings.from_file(path)

    settings.cluster = 'Minimum Frequency Clustering'
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
    label = dataframe.minimum.to_numpy()

    builder = GradientBuilder(
        embedding=embedding,
        label=label,
        settings=settings,
        spectrogram=~spectrogram
    )

    voronoi = VoronoiGradient(builder=builder)

    component = voronoi.build()

    if save:
        figure = component.get('figure')
        filename = 'voronoi_minimum_dataset.png'

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


if __name__ == '__main__':
    main()
