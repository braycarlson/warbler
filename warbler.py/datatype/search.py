"""
Search
------

"""

from __future__ import annotations

import pandas as pd

from constant import TUNING
from fcmeans import FCM
from itertools import product
from joblib import Parallel, delayed
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt

    from typing_extensions import Any


class GridSearch:
    def __init__(self, grid: dict[Any, Any] = None):
        self._history = {}

        self.best_parameters = None
        self.best_score = None
        self.grid = grid
        self.scorer = None

    def _score(
        self,
        x: npt.NDArray,
        parameters: tuple[int | float, ...]
    ) -> tuple[dict[str: int | float], int | float]:
        grid = dict(
            zip(
                self.grid,
                parameters
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
            return parameters, score

    def export(self, filename: str) -> None:
        dataframe = self.history()

        path = TUNING.joinpath(filename + '.csv')
        dataframe.to_csv(path, index=False)

    def fit(self, x: npt.NDArray) -> None:
        best_score = (
            float('-inf')
            if self.scorer.maximize
            else float('inf')
        )

        best_parameters = None

        # Generate all combinations of parameter values
        v = self.grid.values()

        parameters = list(
            product(*v)
        )

        n_jobs = self.grid.get('n_jobs', 5)

        results = Parallel(n_jobs=n_jobs)(
            delayed(self._score)(x, parameter)
            for parameter in tqdm(parameters)
        )

        # Find the best parameters and score
        for index, (parameters, score) in enumerate(results, 1):
            if parameters is None or score is None:
                continue

            if self.scorer.maximize:
                if score > best_score:
                    best_score = score
                    best_parameters = parameters
            else:
                if score <= best_score:
                    best_score = score
                    best_parameters = parameters

            parameters = dict(
                zip(
                    self.grid,
                    parameters
                )
            )

            self._history[index] = {
                'parameters': parameters,
                'score': score
            }

        best_parameters = dict(
            zip(
                self.grid,
                best_parameters
            )
        )

        self.best_parameters = best_parameters
        self.best_score = best_score

    def history(self) -> pd.DataFrame:
        columns = {'index': 'id'}

        dataframe = pd.DataFrame.from_dict(
            self._history,
            orient='index'
        )

        dataframe = dataframe.reset_index()
        dataframe = dataframe.rename(columns=columns)

        ascending = not self.scorer.strategy.maximize

        return dataframe.sort_values(
            'score',
            ascending=ascending
        )

    def result(self) -> dict[str, Any]:
        return {
            'parameters': self.best_parameters,
            'score': self.best_score,
        }
