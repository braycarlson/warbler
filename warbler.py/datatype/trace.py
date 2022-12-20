import seaborn as sns

from abc import ABC, abstractmethod
from plotly import graph_objs as go


class Visitor(ABC):
    @abstractmethod
    def palette(self, labels):
        pass

    @abstractmethod
    def two(self, dataframe):
        pass

    @abstractmethod
    def three(self, dataframe):
        pass


class SingleClusterTrace(Visitor):
    def palette(self, labels):
        pass

    def two(self, dataframe):
        return go.Scattergl(
            x=dataframe.umap_x_2d,
            y=dataframe.umap_y_2d,
            customdata=[dataframe.index],
            marker=dict(
                size=3,
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
                size=3,
            ),
            mode='markers',
            name=str(dataframe.index),
            showlegend=False
        )


class CoordinateTrace(Visitor):
    def palette(self, labels):
        length = len(labels) * 2

        palette = (
            sns
            .color_palette('Paired', length)
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
                    size=3,
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
                    size=3,
                ),
                name=name,
                showlegend=True
            )

            traces.append(trace)

        return traces


class HDBScanTrace(Visitor):
    def palette(self, labels):
        length = len(labels)

        palette = (
            sns
            .color_palette('Paired', length)
            .as_hex()
        )

        iterable = zip(labels, palette)
        return dict(iterable)

    def two(self, dataframe):
        labels = dataframe.hdbscan_label_2d.unique()
        labels.sort()

        legend = self.palette(labels)

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
                    size=3,
                ),
                name=str(label),
                showlegend=True
            )

            traces.append(trace)

        return traces

    def three(self, dataframe):
        labels = dataframe.hdbscan_label_3d.unique()
        labels.sort()

        legend = self.palette(labels)

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
                    size=3,
                ),
                name=str(label),
                showlegend=True
            )

            traces.append(trace)

        return traces


class FuzzyClusterTrace(Visitor):
    def palette(self, labels):
        length = len(labels)

        palette = (
            sns
            .color_palette('Paired', length)
            .as_hex()
        )

        iterable = zip(labels, palette)
        return dict(iterable)

    def two(self, dataframe):
        labels = dataframe.fcm_label_2d.unique()
        labels.sort()

        legend = self.palette(labels)

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
                    size=3,
                ),
                name=str(label),
                showlegend=True
            )

            traces.append(trace)

        return traces

    def three(self, dataframe):
        labels = dataframe.fcm_label_3d.unique()
        labels.sort()

        legend = self.palette(labels)

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
                    size=3,
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
            .color_palette('Paired', length)
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
                    size=3,
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
                    size=3,
                ),
                name=str(sequence),
                showlegend=True
            )

            traces.append(trace)

        return traces
