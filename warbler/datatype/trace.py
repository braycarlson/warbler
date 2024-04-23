from __future__ import annotations

import numpy as np
import seaborn as sns

from abc import ABC, abstractmethod
from matplotlib.cm import viridis
from plotly import graph_objs as go
from typing import TYPE_CHECKING
from warbler.datatype.partition import Partition
from warbler.datatype.settings import Settings

if TYPE_CHECKING:
    import pandas as pd

    from typing_extensions import Any


class Visitor(ABC):
    def __init__(self, settings: Settings = None):
        if settings is None:
            settings = Settings.from_dict(
            {
                'marker': {
                    'opacity': 0.25,
                    'size': 3
                }
            }
        )

        self.settings = settings

    @abstractmethod
    def palette(self, labels: Any) -> Any:
        raise NotImplementedError

    @abstractmethod
    def two(self, dataframe: pd.DataFrame) -> list[go.Scattergl]:
        raise NotImplementedError

    @abstractmethod
    def three(self, dataframe: pd.DataFrame) -> list[go.Scatter3d]:
        raise NotImplementedError


class AcousticTrace(Visitor):
    def __init__(self, settings: Settings = None):
        super().__init__(settings)

        self.column = None

    def palette(self, labels: Any) -> Any:
        total = len(labels)

        palette = [
            viridis(i / (total - 1))
            for i in range(total)
        ]

        iterable = zip(labels, palette)
        return dict(iterable)

    def two(self, dataframe: pd.DataFrame) -> list[go.Scattergl]:
        traces = []

        partition = Partition()
        partition.dataframe = dataframe
        partition.column = self.column

        partitions, sizes = partition.divide()
        legend = self.palette(sizes)

        for name, partition in zip(sizes, partitions):
            color = legend[name]

            trace = go.Scattergl(
                x=partition.umap_x_2d,
                y=partition.umap_y_2d,
                customdata=[partition.index],
                mode='markers',
                marker=dict(
                    color=f"rgba{color}",
                    **self.settings.marker
                ),
                name=name
            )

            traces.append(trace)

        return traces

    def three(self, dataframe: pd.DataFrame) -> list[go.Scatter3d]:
        traces = []

        partition = Partition()
        partition.dataframe = dataframe
        partition.column = self.column

        partitions, sizes = partition.divide()
        legend = self.palette(sizes)

        for name, partition in zip(sizes, partitions):
            color = legend[name]

            trace = go.Scatter3d(
                x=partition.umap_x_3d,
                y=partition.umap_y_3d,
                z=partition.umap_z_3d,
                customdata=[partition.index],
                mode='markers',
                marker=dict(
                    color=f"rgba{color}",
                    **self.settings.marker
                ),
                name=name
            )

            traces.append(trace)

        return traces


class AnimationTrace(Visitor):
    def __init__(self, settings: Settings = None):
        super().__init__(settings)

        self._frames = []
        self.x_eye = -1.0
        self.y_eye = 2.0
        self.z_eye = 0.5

    @property
    def frames(self) -> list[go.Frame]:
        return self._frames

    @staticmethod
    def rotate_z(
        x: float,
        y: float,
        z: float,
        theta: float
    ) -> tuple[float, float, float]:
        w = x + 1j * y

        return (
            np.real(np.exp(1j * theta) * w),
            np.imag(np.exp(1j * theta) * w),
            z
        )

    def animate(self) -> None:
        for theta in np.arange(0, 10, 0.1):
            (xe, ye, ze) = self.rotate_z(
                self.x_eye,
                self.y_eye,
                self.z_eye,
                -theta
            )

            self._frames.append(
                go.Frame(
                    layout={
                        'scene_camera_eye': {
                            'x': xe,
                            'y': ye,
                            'z': ze
                        }
                    }
                )
            )

    def palette(self, labels: Any) -> Any:
        raise NotImplementedError

    def three(self, dataframe: pd.DataFrame) -> go.Scatter3d:
        return go.Scatter3d(
            x=dataframe.umap_x_3d,
            y=dataframe.umap_y_3d,
            z=dataframe.umap_z_3d,
            marker=dict(
                **self.settings.marker
            ),
            mode='markers',
            showlegend=False
        )

    def two(self, dataframe: pd.DataFrame) -> list[go.Scattergl]:
        raise NotImplementedError


class CoordinateTrace(Visitor):
    def __init__(self, settings: Settings = None):
        super().__init__(settings)

    def palette(self, labels: Any) -> Any:
        length = len(labels) * 2

        palette = (
            sns
            .color_palette('tab20', length)
            .as_hex()
        )

        palette = [
            palette[i:i + 2]
            for i in range(0, len(palette), 2)
        ]

        iterable = zip(labels, palette)
        return dict(iterable)

    def two(self, dataframe: pd.DataFrame) -> list[go.Scattergl]:
        columns = [
            'umap_x_2d',
            'umap_y_2d'
        ]

        columns = [
            column
            for pair in zip(columns, columns)
            for column in pair
        ]

        legend = self.palette(columns)

        traces = []

        for index, column in enumerate(columns, 0):
            if index % 2 == 0:
                coordinate = dataframe.nlargest(50, column)
                color, _ = legend[column]
                name = f"{column}_largest"
            else:
                coordinate = dataframe.nsmallest(50, column)
                _, color = legend[column]
                name = f"{column}_smallest"

            trace = go.Scattergl(
                x=coordinate.umap_x_2d,
                y=coordinate.umap_y_2d,
                customdata=[coordinate.index],
                mode='markers',
                marker=dict(
                    color=color,
                    **self.settings.marker
                ),
                name=name,
                showlegend=True
            )

            traces.append(trace)

        return traces

    def three(self, dataframe: pd.DataFrame) -> list[go.Scatter3d]:
        columns = [
            'umap_x_3d',
            'umap_y_3d',
            'umap_z_3d',
        ]

        columns = [
            column
            for pair in zip(columns, columns)
            for column in pair
        ]

        legend = self.palette(columns)

        traces = []

        for index, column in enumerate(columns, 0):
            if index % 2 == 0:
                coordinate = dataframe.nlargest(50, column)
                color, _ = legend[column]
                name = f"{column}_largest"
            else:
                coordinate = dataframe.nsmallest(50, column)
                _, color = legend[column]
                name = f"{column}_smallest"

            trace = go.Scatter3d(
                x=coordinate.umap_x_3d,
                y=coordinate.umap_y_3d,
                z=coordinate.umap_z_3d,
                customdata=[coordinate.index],
                mode='markers',
                marker=dict(
                    color=color,
                    **self.settings.marker
                ),
                name=name,
                showlegend=True
            )

            traces.append(trace)

        return traces


class DurationTrace(AcousticTrace):
    def __init__(self, settings: Settings = None):
        super().__init__(settings)

        self.column = 'duration'


class FuzzyClusterTrace(Visitor):
    def __init__(self, settings: Settings = None):
        super().__init__(settings)

    def palette(self, _: Any = None) -> Any:
        self.settings.fcm.sort()

        palette = (
            sns
            .color_palette(
                'tab20',
                self.settings.fcm.size
            )
            .as_hex()
        )

        iterable = zip(self.settings.fcm, palette)
        return dict(iterable)

    def two(self, dataframe: pd.DataFrame) -> list[go.Scattergl]:
        legend = self.palette()

        labels = dataframe.fcm_label_2d.unique()
        labels.sort()

        traces = []

        for label in labels:
            cluster = dataframe[dataframe.fcm_label_2d == label]
            color = legend[label]

            trace = go.Scattergl(
                x=cluster.umap_x_2d,
                y=cluster.umap_y_2d,
                customdata=[cluster.index],
                mode='markers',
                marker=dict(
                    color=color,
                    **self.settings.marker
                ),
                name=str(label),
                showlegend=True
            )

            traces.append(trace)

        return traces

    def three(self, dataframe: pd.DataFrame) -> list[go.Scatter3d]:
        legend = self.palette()

        labels = dataframe.fcm_label_3d.unique()
        labels.sort()

        traces = []

        for label in labels:
            cluster = dataframe[dataframe.fcm_label_3d == label]
            color = legend[label]

            trace = go.Scatter3d(
                x=cluster.umap_x_3d,
                y=cluster.umap_y_3d,
                z=cluster.umap_z_3d,
                customdata=[cluster.index],
                mode='markers',
                marker=dict(
                    color=color,
                    **self.settings.marker
                ),
                name=str(label),
                showlegend=True
            )

            traces.append(trace)

        return traces


class HDBScanTrace(Visitor):
    def __init__(self, settings: Settings = None):
        super().__init__(settings)

    def palette(self, _: Any = None) -> Any:
        self.settings.hdbscan.sort()

        palette = (
            sns
            .color_palette(
                'tab20',
                self.settings.hdbscan.size
            )
            .as_hex()
        )

        iterable = zip(self.settings.hdbscan, palette)
        return dict(iterable)

    def two(self, dataframe: pd.DataFrame) -> list[go.Scattergl]:
        legend = self.palette()

        labels = dataframe.hdbscan_label_2d.unique()
        labels.sort()

        traces = []

        for label in labels:
            cluster = dataframe[dataframe.hdbscan_label_2d == label]

            color = legend[label]

            trace = go.Scattergl(
                x=cluster.umap_x_2d,
                y=cluster.umap_y_2d,
                customdata=[cluster.index],
                mode='markers',
                marker=dict(
                    color=color,
                    **self.settings.marker
                ),
                name=str(label),
                showlegend=True
            )

            traces.append(trace)

        return traces

    def three(self, dataframe: pd.DataFrame) -> list[go.Scatter3d]:
        legend = self.palette()

        labels = dataframe.hdbscan_label_3d.unique()
        labels.sort()

        traces = []

        for label in labels:
            cluster = dataframe[dataframe.hdbscan_label_3d == label]

            color = legend[label]

            trace = go.Scatter3d(
                x=cluster.umap_x_3d,
                y=cluster.umap_y_3d,
                z=cluster.umap_z_3d,
                customdata=[cluster.index],
                mode='markers',
                marker=dict(
                    color=color,
                    **self.settings.marker
                ),
                name=str(label),
                showlegend=True
            )

            traces.append(trace)

        return traces


class IndividualTrace(Visitor):
    def __init__(self, settings: Settings = None):
        super().__init__(settings)

    def palette(self, labels: Any) -> Any:
        length = len(labels)

        palette = (
            sns
            .color_palette('tab20', length)
            .as_hex()
        )

        iterable = zip(labels, palette)
        return dict(iterable)

    def two(self, dataframe: pd.DataFrame) -> list[go.Scattergl]:
        labels = dataframe.folder.unique()
        labels.sort()

        legend = self.palette(labels)

        traces = []

        for label in labels:
            cluster = dataframe[dataframe.folder == label]

            color = legend[label]

            trace = go.Scattergl(
                x=cluster.umap_x_2d,
                y=cluster.umap_y_2d,
                customdata=[cluster.index],
                mode='markers',
                marker=dict(
                    color=color,
                    **self.settings.marker
                ),
                name=str(label),
                showlegend=True
            )

            traces.append(trace)

        return traces

    def three(self, dataframe: pd.DataFrame) -> list[go.Scatter3d]:
        labels = dataframe.folder.unique()
        labels.sort()

        legend = self.palette(labels)

        traces = []

        for label in labels:
            cluster = dataframe[dataframe.folder == label]

            color = legend[label]

            trace = go.Scatter3d(
                x=cluster.umap_x_3d,
                y=cluster.umap_y_3d,
                z=cluster.umap_z_3d,
                customdata=[cluster.index],
                mode='markers',
                marker=dict(
                    color=color,
                    **self.settings.marker
                ),
                name=str(label),
                showlegend=True
            )

            traces.append(trace)

        return traces


class MaximumFrequencyTrace(AcousticTrace):
    def __init__(self, settings: Settings = None):
        super().__init__(settings)

        self.column = 'maximum'


class MeanFrequencyTrace(AcousticTrace):
    def __init__(self, settings: Settings = None):
        super().__init__(settings)

        self.column = 'mean'


class MinimumFrequencyTrace(AcousticTrace):
    def __init__(self, settings: Settings = None):
        super().__init__(settings)

        self.column = 'minimum'


class SequenceTrace(Visitor):
    def __init__(self, settings: Settings = None):
        super().__init__(settings)

    def palette(self, labels: Any) -> Any:
        length = len(labels)

        palette = (
            sns
            .color_palette('tab20', length)
            .as_hex()
        )

        iterable = zip(labels, palette)
        return dict(iterable)

    def two(self, dataframe: pd.DataFrame) -> list[go.Scattergl]:
        sequences = dataframe.sequence.unique()
        sequences.sort()

        legend = self.palette(sequences)

        traces = []

        for sequence in sequences:
            cluster = dataframe[dataframe.sequence == sequence]
            color = legend[sequence]

            trace = go.Scattergl(
                x=cluster.umap_x_2d,
                y=cluster.umap_y_2d,
                customdata=[cluster.index],
                mode='markers',
                marker=dict(
                    color=color,
                    **self.settings.marker
                ),
                name=str(sequence),
                showlegend=True
            )

            traces.append(trace)

        return traces

    def three(self, dataframe: pd.DataFrame) -> list[go.Scatter3d]:
        sequences = dataframe.sequence.unique()
        sequences.sort()

        legend = self.palette(sequences)

        traces = []

        for sequence in sequences:
            cluster = dataframe[dataframe.sequence == sequence]
            color = legend[sequence]

            trace = go.Scatter3d(
                x=cluster.umap_x_3d,
                y=cluster.umap_y_3d,
                z=cluster.umap_z_3d,
                customdata=[cluster.index],
                mode='markers',
                marker=dict(
                    color=color,
                    **self.settings.marker
                ),
                name=str(sequence),
                showlegend=True
            )

            traces.append(trace)

        return traces


class SingleClusterTrace(Visitor):
    def __init__(self, settings: Settings = None):
        super().__init__(settings)

    def palette(self, labels: Any) -> None:
        raise NotImplementedError

    def two(self, dataframe: pd.DataFrame) -> list[go.Scattergl]:
        return go.Scattergl(
            x=dataframe.umap_x_2d,
            y=dataframe.umap_y_2d,
            customdata=[dataframe.index],
            marker=dict(
                **self.settings.marker
            ),
            mode='markers',
            name=str(dataframe.index),
            showlegend=False
        )

    def three(self, dataframe: pd.DataFrame) -> list[go.Scatter3d]:
        return go.Scatter3d(
            x=dataframe.umap_x_3d,
            y=dataframe.umap_y_3d,
            z=dataframe.umap_z_3d,
            customdata=[dataframe.index],
            marker=dict(
                **self.settings.marker
            ),
            mode='markers',
            name=str(dataframe.index),
            showlegend=False
        )
