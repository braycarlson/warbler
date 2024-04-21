"""
Scorer
------

"""

from __future__ import annotations

import numpy as np

from abc import abstractmethod, ABC
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt

    from fcmeans import FCM


class Strategy(ABC):
    """A base class for strategies on creating a scorer"""

    def __init__(self, estimator: FCM = None, x: npt.NDArray = None):
        self.estimator = estimator
        self.maximize = True
        self.x = x

    @abstractmethod
    def score(self) -> int | float:
        raise NotImplementedError


class CalinskiHarabaszScore(Strategy):
    def __init__(self):
        super().__init__()
        self.maximize = True

    def __repr__(self) -> str:
        return 'calinski_harabasz_score'

    def __str__(self) -> str:
        return 'Calinski-Harabasz Score'

    def score(self) -> int | float:
        label = np.argmax(self.estimator.u, axis=1)

        return calinski_harabasz_score(
            self.x,
            label
        )


class DaviesBouldinIndex(Strategy):
    def __init__(self):
        super().__init__()
        self.maximize = False

    def __repr__(self) -> str:
        return 'davies_bouldin_index'

    def __str__(self) -> str:
        return 'Davies-Bouldin Index'

    def score(self) -> int | float:
        label = np.argmax(self.estimator.u, axis=1)

        return davies_bouldin_score(
            self.x,
            label
        )


class FukuyamaSugenoIndex(Strategy):
    def __init__(self):
        super().__init__()
        self.maximize = False

    def __repr__(self) -> str:
        return 'fukuyama_sugeno_index'

    def __str__(self) -> str:
        return 'Fukuyama-Sugeno Index'

    def score(self) -> int | float:
        distances = np.sum(
            (self.x[:, np.newaxis] - self.estimator.centers) ** 2,
            axis=2
        )

        inertia = np.sum(distances * self.estimator.u)
        mean = np.mean(self.x, axis=0)

        distances = np.sum(
            (self.x - mean) ** 2,
            axis=1
        )

        return inertia - np.sum(distances)


class PartitionCoefficient(Strategy):
    def __init__(self):
        super().__init__()
        self.maximize = True

    def __repr__(self) -> str:
        return 'partition_coefficient'

    def __str__(self) -> str:
        return 'Partition Coefficient'

    def score(self) -> int | float:
        return self.estimator.partition_coefficient


class PartitionEntropyCoefficient(Strategy):
    def __init__(self):
        super().__init__()
        self.maximize = False

    def __repr__(self) -> str:
        return 'partition_entropy_coefficient'

    def __str__(self) -> str:
        return 'Partition Entropy Coefficient'

    def score(self) -> int | float:
        return self.estimator.partition_entropy_coefficient


class SilhouetteScore(Strategy):
    def __init__(self):
        super().__init__()
        self.maximize = True

    def __repr__(self) -> str:
        return 'silhouette_score'

    def __str__(self) -> str:
        return 'Silhouette Score'

    def score(self) -> int | float:
        label = np.argmax(self.estimator.u, axis=1)

        return silhouette_score(
            self.x,
            label,
            metric='euclidean'
        )


class SumOfSquaredErrors(Strategy):
    def __init__(self):
        super().__init__()
        self.maximize = False

    def __repr__(self) -> str:
        return 'sum_of_squared_errors'

    def __str__(self) -> str:
        return 'Sum Of Squared Errors'

    def score(self) -> int | float:
        distances = np.sum(
            (self.x[:, np.newaxis] - self.estimator.centers) ** 2,
            axis=2
        )

        return np.sum(
            distances *
            self.estimator.u**self.estimator.m
        )


class XieBeniIndex(Strategy):
    def __init__(self):
        super().__init__()
        self.maximize = False

    def __repr__(self) -> str:
        return 'xie_beni_index'

    def __str__(self) -> str:
        return 'Xie-Beni Index'

    def score(self) -> int | float:
        n_samples, _ = self.x.shape
        n_clusters = len(self.estimator.centers)

        label = np.argmax(self.estimator.u, axis=1)

        compactness = np.sum(
            [np.sum(self.estimator.u[label == i, i]**2 *
                np.linalg.norm(self.x[label == i]
                    - self.estimator.centers[i], axis=1)**2)
            for i in range(n_clusters)]
        )

        separation = np.min([
            np.linalg.norm(self.estimator.centers[i] - self.estimator.centers[j])**2
            for i in range(n_clusters) for j in range(i + 1, n_clusters)
        ])

        return compactness / (n_samples * separation)


class MultiScorer:
    def __init__(self, strategies: list[Strategy] = None):
        self.strategies = strategies

    def __call__(self):
        score = {}

        for strategy in self.strategies:
            name = repr(strategy)
            score[name] = strategy.score()

        return score

    @property
    def estimator(self) -> FCM:
        pass

    @estimator.setter
    def estimator(self, estimator: FCM) -> None:
        for strategy in self.strategies:
            strategy.estimator = estimator

    @property
    def maximize(self) -> bool:
        return self.strategy.maximize

    @property
    def x(self) -> npt.NDArray:
        pass

    @x.setter
    def x(self, x: npt.NDArray) -> None:
        for strategy in self.strategies:
            strategy.x = x


class Scorer:
    def __init__(self, strategy: Strategy = None):
        self.strategy = strategy

    def __call__(self):
        score = {}

        name = repr(self.strategy)
        score[name] = self.strategy.score()

        return score

    def __repr__(self) -> str:
        return repr(self.strategy)

    def __str__(self) -> str:
        return str(self.strategy)

    @property
    def estimator(self) -> FCM:
        return self._estimator

    @estimator.setter
    def estimator(self, estimator: FCM) -> None:
        self.strategy.estimator = estimator

    @property
    def maximize(self) -> bool:
        return self.strategy.maximize

    @property
    def x(self) -> npt.NDArray:
        return self._x

    @x.setter
    def x(self, x: npt.NDArray) -> None:
        self.strategy.x = x
