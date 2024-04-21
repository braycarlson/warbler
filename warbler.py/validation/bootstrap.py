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
from sklearn.utils import resample
from tqdm import tqdm
from typing_extensions import Any


def metric(
    i: int, x: npt.NDArray,
    strategies: list[Strategy]
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

    label = np.argmax(fcm.u, axis=1)

    scoring = {}

    for strategy in strategies:
        scorer = Scorer()
        scorer.strategy = strategy
        scorer.estimator = fcm
        scorer.label = label
        scorer.x = sample

        k = repr(scorer)
        v = scorer()

        scoring[k] = v

    return scoring


def main() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    dataframe = dataframe.drop('fcm_label_2d', axis=1)

    x = np.array(
        [
            dataframe.umap_x_2d,
            dataframe.umap_y_2d
        ]
    ).transpose()

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

    results = Parallel(n_jobs=n_jobs)(
        delayed(metric)(i, x, strategies)
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

        filename = f"bootstrap_{column}.png"
        path = TUNING.joinpath(filename)

        plt.savefig(
            path,
            bbox_inches='tight',
            dpi=300,
            format='png'
        )


if __name__ == '__main__':
    main()
