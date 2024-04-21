from __future__ import annotations

import numpy as np
import pandas as pd

from ast import literal_eval
from constant import SETTINGS, TUNING
from datatype.dataset import Dataset
from datatype.scorer import (
    CalinskiHarabaszScore,
    DaviesBouldinIndex,
    PartitionCoefficient,
    PartitionEntropyCoefficient,
    SilhouetteScore,
    SumOfSquaredErrors,
    XieBeniIndex
)
from datatype.settings import Settings
from datatype.voronoi import Builder, VoronoiFCM
from fcmeans import FCM


def main() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    x = np.array(
        [
            dataframe.umap_x_2d,
            dataframe.umap_y_2d
        ]
    ).transpose()

    path = TUNING.joinpath('tuning/search.csv')
    search = pd.read_csv(path)

    strategies = [
        CalinskiHarabaszScore(),
        DaviesBouldinIndex(),
        PartitionCoefficient(),
        PartitionEntropyCoefficient(),
        SilhouetteScore(),
        SumOfSquaredErrors(),
        XieBeniIndex()
    ]

    for strategy in strategies:
        ascending = not strategy.maximize
        by, filename = repr(strategy), repr(strategy)

        temporary = search.sort_values(
            ascending=ascending,
            by=by
        )

        parameter = temporary.iloc[0].parameter
        parameter = literal_eval(parameter)

        print(f"{by}: {parameter}")

        fcm = FCM(**parameter)
        fcm.fit(x)

        label = fcm.predict(x)
        dataframe['fcm_label_2d'] = label

        # Load default settings
        path = SETTINGS.joinpath('voronoi.json')
        settings = Settings.from_file(path)
        settings.name = str(strategy)

        unique = dataframe.fcm_label_2d.unique()

        by = ['duration']

        ascending = [False]

        temporary = dataframe.sort_values(
            ascending=ascending,
            by=by
        )

        coordinates = [
            temporary.umap_x_2d,
            temporary.umap_y_2d
        ]

        embedding = np.column_stack(coordinates)

        spectrogram = temporary.filter_array.to_numpy()
        label = temporary.fcm_label_2d.to_numpy()

        voronoi = VoronoiFCM()

        voronoi.builder = Builder()
        voronoi.embedding = embedding
        voronoi.label = label
        voronoi.spectrogram = ~spectrogram
        voronoi.settings = settings
        voronoi.unique = unique

        component = voronoi.build()

        figure = component.get('figure')

        filename = filename + '.png'

        voronoi.save(
            figure=figure,
            filename=filename
        )


if __name__ == '__main__':
    main()
