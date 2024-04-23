from __future__ import annotations

import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import numpy.typing as npt
import pandas as pd
import scienceplots

from fcmeans import FCM
from joblib import delayed, Parallel
from matplotlib.ticker import MaxNLocator
from sklearn.utils import resample
from tqdm import tqdm
from typing_extensions import Any
from warbler.constant import TUNING
from warbler.datatype.dataset import Dataset
from warbler.datatype.scorer import (
    CalinskiHarabaszScore,
    DaviesBouldinIndex,
    MetricStrategy,
    PartitionCoefficient,
    PartitionEntropyCoefficient,
    Scorer,
    SilhouetteScore,
    SumOfSquaredErrors,
    XieBeniIndex
)


def run(
    i: int, x: npt.NDArray,
    metrics: list[MetricStrategy]
) -> dict[str, Any]:
    if i == 42:
        return None

    sample = resample(x, replace=True)

    fcm = FCM(
        m=2.9,
        max_iter=200,
        n_clusters=14,
        random_state=i
    )

    fcm.fit(sample)

    scoring = {}

    for metric in metrics:
        metric = metric(estimator=fcm, x=x)
        scorer = Scorer(metric=metric)

        k = repr(scorer)
        v = scorer()

        scoring[k] = v

    return scoring


def main() -> None:
    plt.style.use('science')

    dataset = Dataset('segment')
    dataframe = dataset.load()

    dataframe = dataframe.drop('fcm_label_2d', axis=1)

    x = np.array(
        [
            dataframe.umap_x_2d,
            dataframe.umap_y_2d
        ]
    ).transpose()

    metrics = [
        CalinskiHarabaszScore,
        DaviesBouldinIndex,
        PartitionCoefficient,
        PartitionEntropyCoefficient,
        SilhouetteScore,
        SumOfSquaredErrors,
        XieBeniIndex
    ]

    scoring = {
        repr(metric()): []
        for metric in metrics
    }

    total = 1001
    n_jobs = int(multiprocessing.cpu_count() / 1.5)

    results = Parallel(n_jobs=n_jobs)(
        delayed(run)(i, x, metrics)
        for i in tqdm(range(total), desc='Processing')
    )

    for local in results:
        if local is not None:
            for k, v in local.items():
                s = v[k]
                scoring[k].append(s)

    score = pd.DataFrame.from_dict(scoring)

    path = TUNING.joinpath('bootstrap.csv')
    score.to_csv(path, index_label='id')

    figsize = (18, 9)

    for metric in metrics:
        instance = metric()

        column = repr(instance)
        title, ylabel = str(instance), str(instance)

        scores = score[column].tolist()

        plt.figure(figsize=figsize)

        plt.plot(scores, marker='o')

        plt.xlabel('Iteration')
        plt.ylabel(ylabel)

        ax = plt.gca()

        locator = MaxNLocator(integer=True)
        ax.xaxis.set_major_locator(locator)

        plt.title(title)

        plt.grid(visible=True)

        filename = f"bootstrap_{column}.png"
        path = TUNING.joinpath(filename)

        plt.savefig(
            path,
            bbox_inches='tight',
            dpi=300,
            format='png'
        )

        plt.close()


if __name__ == '__main__':
    main()
