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

    fcm = FCM(
        m=2.9,
        max_iter=200,
        n_clusters=14,
        random_state=i
    )

    fcm.fit(x)

    scoring = {}

    for metric in metrics:
        metric = metric(estimator=fcm, x=x)
        scorer = Scorer(metric=metric)

        k = repr(scorer)
        v = scorer()

        scoring[k] = v[k]

    return scoring


def resample(dataframe: pd.DataFrame) -> npt.NDArray:
    unique = dataframe['fcm_label_2d'].unique()

    samples = []

    for fcm_label_2d in unique:
        subset = dataframe[dataframe['fcm_label_2d'] == fcm_label_2d]
        sample = subset.sample(800, replace=True)
        samples.append(sample)

    sample = pd.concat(samples)
    sample = sample.sort_index().reset_index()

    sample = sample.drop('fcm_label_2d', axis=1)

    return np.array(
        [
            sample.umap_x_2d,
            sample.umap_y_2d
        ]
    ).transpose()


def main() -> None:
    plt.style.use('science')

    dataset = Dataset('segment')
    dataframe = dataset.load()

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

    iterable = [
        (i, resample(dataframe), metrics)
        for i in range(total)
    ]

    results = Parallel(n_jobs=n_jobs)(
        delayed(run)(*parameter)
        for parameter in tqdm(iterable, total=total)
    )

    for result in results:
        if result is not None:
            for k, v in result.items():
                scoring[k].append(v)

    score = pd.DataFrame.from_dict(scoring)

    path = TUNING.joinpath('subset.csv')
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

        filename = f"subset_{column}.png"
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
