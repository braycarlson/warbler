from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from constant import PROJECTION, SETTINGS
from datatype.dataset import Dataset
from datatype.scorer import (
    CalinskiHarabaszScore,
    DaviesBouldinIndex,
    FukuyamaSugenoIndex,
    PartitionCoefficient,
    PartitionCoefficientDifference,
    PartitionEntropyCoefficient,
    Scorer,
    SilhouetteScore,
    SumOfSquaredErrors,
    XieBeniIndex
)
from datatype.search import GridSearch
from datatype.settings import Settings
from datatype.voronoi import Builder, VoronoiFCM
from fcmeans import FCM


def main() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    dataframe = dataframe.drop(
        ['fcm_label_2d'],
        axis=1,
        errors='ignore'
    )

    x = np.array(
        [
            dataframe.umap_x_2d,
            dataframe.umap_y_2d
        ]
    ).transpose()

    grid = {
        'm': np.arange(1.5, 2.5, 0.1),
        'max_iter': np.arange(50, 200, 50),
        'n_clusters': np.arange(2, 20)
    }

    # grid = {
    #     'm': np.arange(1.5, 2.5, 1.0),
    #     'max_iter': np.arange(50, 200, 50),
    #     'n_clusters': np.arange(2, 5)
    # }

    strategies = [
        CalinskiHarabaszScore(),
        DaviesBouldinIndex(),
        FukuyamaSugenoIndex(),
        PartitionCoefficient(),
        PartitionCoefficientDifference(),
        PartitionEntropyCoefficient(),
        SilhouetteScore(),
        SumOfSquaredErrors(),
        XieBeniIndex()
    ]

    figures = []

    for strategy in strategies:
        print(dataframe[['filename']].head(10))

        name = str(strategy)
        print(name)

        scorer = Scorer()
        scorer.strategy = strategy

        search = GridSearch()
        search.grid = grid
        search.scorer = scorer

        search.fit(x)

        filename = repr(scorer.strategy)
        search.export(filename)

        fcm = FCM(**search.best_parameters)
        fcm.fit(x)

        label = fcm.predict(x)
        dataframe['fcm_label_2d'] = label

        # Create a Voronoi plot for each strategy
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

        figures.append(figure)

        filename = filename + '.png'

        voronoi.save(
            figure=figure,
            filename=filename
        )



    # # Create an image of all strategies
    # plot = len(figures)
    # row = int(plot ** 0.5)
    # column = (plot + row - 1) // row

    # dpi = 1200
    # figsize = (18, 9)

    # fig, axes = plt.subplots(
    #     row,
    #     column,
    #     dpi=dpi,
    #     figsize=figsize
    # )

    # fig.subplots_adjust(hspace=0)

    # for index, figure in enumerate(figures, 0):
    #     figure.canvas.draw()
    #     width, height = figure.canvas.get_width_height()

    #     image = np.frombuffer(figure.canvas.tostring_rgb(), dtype=np.uint8)
    #     image = image.reshape(int(height), int(width), 3)

    #     ax = axes.flat[index]
    #     ax.axis('off')
    #     ax.imshow(image)

    # path = PROJECTION.joinpath('tuning.png')

    # fig.suptitle(
    #     'Clustering by Best Scoring Metrics',
    #     fontweight='bold',
    #     fontsize=18
    # )

    # fig.savefig(
    #     path,
    #     bbox_inches='tight',
    #     dpi=dpi,
    #     format='png'
    # )


if __name__ == '__main__':
    main()
