from __future__ import annotations

import numpy as np

from abc import abstractmethod, ABC
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score
)
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt

    from fcmeans import FCM


class MetricStrategy(ABC):
    def __init__(
        self,
        estimator: FCM = None,
        x: npt.NDArray = None
    ):
        self.estimator = estimator
        self.maximize = True
        self.x = x

    @abstractmethod
    def score(self) -> int | float:
        raise NotImplementedError


class CalinskiHarabaszScore(MetricStrategy):
    def __init__(
        self,
        estimator: FCM = None,
        x: npt.NDArray = None
    ):
        super().__init__(estimator, x)

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


class DaviesBouldinIndex(MetricStrategy):
    def __init__(
        self,
        estimator: FCM = None,
        x: npt.NDArray = None
    ):
        super().__init__(estimator, x)

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


class FukuyamaSugenoIndex(MetricStrategy):
    def __init__(
        self,
        estimator: FCM = None,
        x: npt.NDArray = None
    ):
        super().__init__(estimator, x)

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


class PartitionCoefficient(MetricStrategy):
    def __init__(
        self,
        estimator: FCM = None,
        x: npt.NDArray = None
    ):
        super().__init__(estimator, x)

        self.maximize = True

    def __repr__(self) -> str:
        return 'partition_coefficient'

    def __str__(self) -> str:
        return 'Partition Coefficient'

    def score(self) -> int | float:
        return self.estimator.partition_coefficient


class PartitionEntropyCoefficient(MetricStrategy):
    def __init__(
        self,
        estimator: FCM = None,
        x: npt.NDArray = None
    ):
        super().__init__(estimator, x)

        self.maximize = False

    def __repr__(self) -> str:
        return 'partition_entropy_coefficient'

    def __str__(self) -> str:
        return 'Partition Entropy Coefficient'

    def score(self) -> int | float:
        return self.estimator.partition_entropy_coefficient


class SilhouetteScore(MetricStrategy):
    def __init__(
        self,
        estimator: FCM = None,
        x: npt.NDArray = None
    ):
        super().__init__(estimator, x)

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


class SumOfSquaredErrors(MetricStrategy):
    def __init__(
        self,
        estimator: FCM = None,
        x: npt.NDArray = None
    ):
        super().__init__(estimator, x)

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


class XieBeniIndex(MetricStrategy):
    def __init__(
        self,
        estimator: FCM = None,
        x: npt.NDArray = None
    ):
        super().__init__(estimator, x)

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
    def __init__(self, metrics: list[MetricStrategy] | None = None):
        self.metrics = metrics

    def __call__(self):
        score = {}

        for metric in self.metrics:
            name = repr(metric)
            score[name] = metric.score()

        return score

    @property
    def estimator(self) -> FCM:
        raise NotImplementedError

    @estimator.setter
    def estimator(self, estimator: FCM) -> None:
        for metric in self.metrics:
            metric.estimator = estimator

    @property
    def maximize(self) -> bool:
        return self.metric.maximize

    @property
    def x(self) -> npt.NDArray:
        raise NotImplementedError

    @x.setter
    def x(self, x: npt.NDArray) -> None:
        for metric in self.metrics:
            metric.x = x


class Scorer:
    def __init__(self, metric: MetricStrategy = None):
        self.metric = metric

    def __call__(self):
        score = {}

        name = repr(self.metric)
        score[name] = self.metric.score()

        return score

    def __repr__(self) -> str:
        return repr(self.metric)

    def __str__(self) -> str:
        return str(self.metric)

    @property
    def estimator(self) -> FCM:
        return self.metric.estimator

    @estimator.setter
    def estimator(self, estimator: FCM) -> None:
        self.metric.estimator = estimator

    @property
    def maximize(self) -> bool:
        return self.metric.maximize

    @property
    def x(self) -> npt.NDArray:
        return self.metric.x

    @x.setter
    def x(self, x: npt.NDArray) -> None:
        self.metric.x = x
