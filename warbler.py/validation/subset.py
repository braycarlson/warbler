import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import numpy.typing as npt
import pandas as pd

from constant import TUNING
from datatype.dataset import Dataset
from datatype.scorer import (
    CalinskiHarabaszScore,
    DaviesBouldinIndex,
    PartitionCoefficient,
    PartitionEntropyCoefficient,
    Scorer,
    SilhouetteScore,
    Strategy,
    SumOfSquaredErrors,
    XieBeniIndex
)
from fcmeans import FCM
from joblib import delayed, Parallel
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm
from typing_extensions import Any


def metric(
    i: int, x: npt.NDArray,
    strategies: list[Strategy]
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

    label = np.argmax(fcm.u, axis=1)

    scoring = {}

    for strategy in strategies:
        scorer = Scorer()
        scorer.strategy = strategy
        scorer.estimator = fcm
        scorer.label = label
        scorer.x = x

        k = repr(scorer)
        v = scorer()

        scoring[k] = v[k]

    return scoring


def resample(dataframe: pd.DataFrame) -> npt.NDArray:
    unique = dataframe['fcm_label_2d'].unique()

    samples = []

    for fcm_2d_label in unique:
        subset = dataframe[dataframe['fcm_label_2d'] == fcm_2d_label]
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
    dataset = Dataset('segment')
    dataframe = dataset.load()

    strategies = [
        CalinskiHarabaszScore(),
        DaviesBouldinIndex(),
        PartitionCoefficient(),
        PartitionEntropyCoefficient(),
        SilhouetteScore(),
        SumOfSquaredErrors(),
        XieBeniIndex()
    ]

    scoring = {
        repr(strategy): []
        for strategy in strategies
    }

    total = 1001
    n_jobs = int(multiprocessing.cpu_count() / 1.5)

    iterable = [
        (i, resample(dataframe), strategies)
        for i in range(total)
    ]

    results = Parallel(n_jobs=n_jobs)(
        delayed(metric)(*parameter)
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

    for strategy in strategies:
        column = repr(strategy)
        title, ylabel = str(strategy), str(strategy)

        s = score[column].tolist()

        plt.figure(figsize=figsize)

        plt.plot(s, marker='o')

        plt.xlabel('Iteration')
        plt.ylabel(ylabel)

        ax = plt.gca()

        locator = MaxNLocator(integer=True)
        ax.xaxis.set_major_locator(locator)

        plt.title(title)

        plt.grid(True)

        filename = f"subset_{column}.png"
        path = TUNING.joinpath(filename)

        plt.savefig(
            path,
            bbox_inches='tight',
            dpi=300,
            format='png'
        )


if __name__ == '__main__':
    main()
