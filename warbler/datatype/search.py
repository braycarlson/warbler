from __future__ import annotations

import multiprocessing
import pandas as pd

from fcmeans import FCM
from itertools import product
from joblib import Parallel, delayed
from tqdm import tqdm
from typing_extensions import TYPE_CHECKING
from warbler.constant import TUNING

if TYPE_CHECKING:
    import numpy.typing as npt

    from typing_extensions import Any
    from warbler.scorer import Scorer, Strategy


class GridSearch:
    def __init__(
        self,
        grid: dict[Any, Any] | None = None,
        scorer: Scorer | None = None
    ):
        self._history = {}

        self.grid = grid
        self.scorer = scorer

    @property
    def dataframe(self) -> pd.DataFrame:
        dataframe = pd.DataFrame.from_dict(
            self._history,
            orient='index'
        )

        score = dataframe.score.apply(pd.Series)

        columns = ['score']
        dataframe = dataframe.drop(columns=columns)

        return pd.concat(
            [dataframe, score],
            axis=1
        )

    def _score(
        self,
        x: npt.NDArray,
        parameter: tuple[int | float, ...]
    ) -> tuple[dict[str: int | float], int | float]:
        grid = dict(
            zip(
                self.grid,
                parameter
            )
        )

        try:
            estimator = FCM(**grid)
            estimator.fit(x)

            self.scorer.estimator = estimator
            self.scorer.x = x
            score = self.scorer()
        except ZeroDivisionError:
            return None, None
        else:
            return parameter, score

    def best(self, metric: Strategy) -> dict[str, float]:
        ascending = not metric.maximize
        name = repr(metric)

        dataframe = self.dataframe.sort_values(
            name,
            ascending=ascending
        )

        column = ['parameter', name]
        return tuple(dataframe.iloc[0][column])

    def export(self, filename: str) -> None:
        path = TUNING.joinpath(filename + '.csv')
        self.dataframe.to_csv(path, index_label='id')

    def fit(self, x: npt.NDArray) -> None:
        v = self.grid.values()

        parameters = list(
            product(*v)
        )

        n_jobs = int(multiprocessing.cpu_count() / 1.5)

        result = Parallel(n_jobs=n_jobs)(
            delayed(self._score)(x, parameter)
            for parameter in tqdm(parameters)
        )

        for index, (parameter, score) in enumerate(result, 1):
            if parameter is None or score is None:
                continue

            parameter = dict(
                zip(
                    self.grid,
                    parameter
                )
            )

            self._history[index] = {
                'parameter': parameter,
                'score': score
            }
