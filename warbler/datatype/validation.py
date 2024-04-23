from __future__ import annotations

import pandas as pd

from fcmeans import FCM
from joblib import delayed, Parallel
from sklearn.metrics import (
    adjusted_rand_score,
    jaccard_score,
    mutual_info_score,
    silhouette_score,
    rand_score
)
from tqdm import tqdm
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt


class Validate:
    def __init__(self, x: npt.NDArray = None, y: npt.NDArray = None):
        self.x = x
        self.y = y

    def _adjusted_rand_score(self) -> int | float:
        return adjusted_rand_score(self.x, self.y)

    def _jaccard_score(self) -> int | float:
        return jaccard_score(self.x, self.y)

    def _label(self, x: npt.NDArray) -> npt.NDArray:
        fcm = FCM(
            m=1.5,
            max_iter=150,
            n_clusters=19
        )

        fcm.fit(x)
        return fcm.u.argmax(axis=1)

    def _mutual_info_score(self) -> int | float:
        return mutual_info_score(self.x, self.y)

    def _score(
        self,
        i: int,
        j: int,
        x: npt.NDArray,
        y: npt.NDArray
    ) -> list[dict[str, float]]:
        score = silhouette_score(x, y)
        return {f"{i}-{j}": score}

    def _silhouette_score(self) -> int | float:
        return silhouette_score(self.x, self.y)

    def _rand_score(self) -> int | float:
        return rand_score(self.x, self.y)

    def bootstrap(self, x: npt.NDArray) -> pd.DataFrame:
        iteration = 10

        result = Parallel(n_jobs=-1)(
            delayed(self._label)(x)
            for _ in tqdm(range(iteration), desc='Clustering')
        )

        scoring = []

        scoring = Parallel(n_jobs=-1)(
            delayed(self._score)(i, j, result[i], result[j])
            for i in tqdm(range(iteration), desc='Scoring')
            for j in range(i + 1, iteration)
        )

        dataframe = pd.DataFrame(scoring)

        value_vars = dataframe.columns.tolist()

        return dataframe.melt(
            value_vars=value_vars,
            value_name='score',
            var_name='trial'
        ).dropna().reset_index(drop=True)


    def run(self) -> pd.DataFrame:
        scoring = {
            'adjusted_rand_score': self._adjusted_rand_score(),
            'jaccard_score': self._jaccard_score(),
            'mutual_info_score': self._mutual_info_score(),
            'silhouette_score': self._silhouette_score(),
            'rand_score': self._rand_score()
        }

        return pd.DataFrame.from_dict(scoring)
