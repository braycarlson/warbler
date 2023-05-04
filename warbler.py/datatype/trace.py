import numpy as np
import seaborn as sns

from abc import ABC, abstractmethod
from datatype.settings import Settings
from plotly import graph_objs as go


class Visitor(ABC):
    def __init__(self):
        self._settings = Settings.from_dict(
            {
                'marker': {
                    'opacity': 0.25,
                    'size': 3
                }
            }
        )

    @abstractmethod
    def palette(self, labels):
        pass

    @abstractmethod
    def two(self, dataframe):
        pass

    @abstractmethod
    def three(self, dataframe):
        pass

    @property
    def settings(self):
        return self._settings

    @settings.setter
    def settings(self, settings):
        self._settings = settings


class AnimationTrace(Visitor):
    def __init__(self):
        self._frames = []
        self.x_eye = -1
        self.y_eye = 2
        self.z_eye = 0.5

    @property
    def frames(self):
        return self._frames

    @staticmethod
    def rotate_z(x, y, z, theta):
        w = x + 1j * y

        return (
            np.real(np.exp(1j * theta) * w),
            np.imag(np.exp(1j * theta) * w),
            z
        )

    def animate(self):
        for theta in np.arange(0, 10, 0.1):
            (xe, ye, ze) = self.rotate_z(
                self.x_eye,
                self.y_eye,
                self.z_eye,
                -theta
            )

            self._frames.append(
                go.Frame(
                    layout=dict(
                        scene_camera_eye=dict(
                            x=xe,
                            y=ye,
                            z=ze
                        )
                    )
                )
            )

    def palette(self, labels):
        pass

    def three(self, dataframe):
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

    def two(self, dataframe):
        pass


class CoordinateTrace(Visitor):
    def palette(self, labels):
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

    def two(self, dataframe):
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

    def three(self, dataframe):
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


class DurationTrace(Visitor):
    def palette(self, labels):
        palette = [
            (212, 212, 212, 0.9),
            (154, 154, 154, 0.9),
            (112, 112, 112, 0.9),
            (69, 69, 69, 0.9)
        ]

        iterable = zip(labels, palette)
        return dict(iterable)

    def two(self, dataframe):
        labels = dataframe.hdbscan_label_2d.unique()
        labels = labels[labels != -1]

        sizes = [
            'Minumum to Minimum Median',
            'Minimum Median to Median',
            'Median to Median Maximum',
            'Median Maximum to Maximum',
        ]

        legend = self.palette(sizes)

        traces = []

        for label in labels:
            cluster = dataframe[dataframe.hdbscan_label_2d == label]

            minimum = cluster.duration.min()
            median = cluster.duration.median()
            maximum = cluster.duration.max()

            minimum_median = (median + minimum) / 2
            maximum_median = (maximum + median) / 2

            partitions = [
                cluster.loc[
                    (cluster.duration >= minimum) &
                    (cluster.duration < minimum_median)
                ],
                cluster.loc[
                    (cluster.duration >= minimum_median) &
                    (cluster.duration < median)
                ],
                cluster.loc[
                    (cluster.duration >= median) &
                    (cluster.duration < maximum_median)
                ],
                cluster.loc[
                    (cluster.duration >= maximum_median) &
                    (cluster.duration < maximum)
                ]
            ]

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
                    name=str(label)
                )

                traces.append(trace)

        return traces

    def three(self, dataframe):
        labels = dataframe.hdbscan_label_2d.unique()
        labels = labels[labels > -1]

        sizes = [
            'Minumum to Minimum Median',
            'Minimum Median to Median',
            'Median to Median Maximum',
            'Median Maximum to Maximum',
        ]

        legend = self.palette(sizes)

        traces = []

        for label in labels:
            cluster = dataframe[dataframe.hdbscan_label_2d == label]

            minimum = cluster.duration.min()
            median = cluster.duration.median()
            maximum = cluster.duration.max()

            minimum_median = (median + minimum) / 2
            maximum_median = (maximum + median) / 2

            partitions = [
                cluster.loc[
                    (cluster.duration >= minimum) &
                    (cluster.duration < minimum_median)
                ],
                cluster.loc[
                    (cluster.duration >= minimum_median) &
                    (cluster.duration < median)
                ],
                cluster.loc[
                    (cluster.duration >= median) &
                    (cluster.duration < maximum_median)
                ],
                cluster.loc[
                    (cluster.duration >= maximum_median) &
                    (cluster.duration < maximum)
                ]
            ]

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
                    name=str(label)
                )

                traces.append(trace)

        return traces


class FuzzyClusterTrace(Visitor):
    def palette(self):
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

    def two(self, dataframe):
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

    def three(self, dataframe):
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
    def palette(self):
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

    def two(self, dataframe):
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

    def three(self, dataframe):
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
    def palette(self, labels):
        length = len(labels)

        palette = (
            sns
            .color_palette('tab20', length)
            .as_hex()
        )

        iterable = zip(labels, palette)
        return dict(iterable)

    def two(self, dataframe):
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

    def three(self, dataframe):
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


class SequenceTrace(Visitor):
    def palette(self, labels):
        length = len(labels)

        palette = (
            sns
            .color_palette('tab20', length)
            .as_hex()
        )

        iterable = zip(labels, palette)
        return dict(iterable)

    def two(self, dataframe):
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

    def three(self, dataframe):
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
    def palette(self, labels):
        pass

    def two(self, dataframe):
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

    def three(self, dataframe):
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
