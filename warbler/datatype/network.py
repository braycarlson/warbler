from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from matplotlib.lines import Line2D
from typing import TYPE_CHECKING
from warbler.datatype.builder import Base, Plot

if TYPE_CHECKING:
    import numpy.typing as npt

    from typing_extensions import Any, Self
    from warbler.datatype.settings import Settings


class Builder(Base):
    def __init__(
        self,
        component: dict[Any, Any] | None = None,
        embedding: npt.NDArray | None = None,
        label: npt.NDArray | None = None,
        sequence: npt.NDArray | None = None,
        settings: Settings | None = None,
        spectrogram: npt.NDArray | None = None,
        unique: npt.NDArray | None = None,
    ):
        super().__init__(
            component,
            embedding,
            label,
            sequence,
            settings,
            spectrogram,
            unique
        )
    def ax(self) -> Self:
        figsize = self.settings.figure.get('figsize')
        figure, ax = plt.subplots(figsize=figsize)

        ax.xaxis.set_visible(self.settings.is_axis)
        ax.yaxis.set_visible(self.settings.is_axis)

        self.component['ax'] = ax
        self.component['figure'] = figure

        return self

    def build(self) -> Self:
        sequences = self.component.get('sequences')

        unique = np.unique(
            np.concatenate(sequences)
        )

        length = len(unique)

        element_dict = {
            element: index
            for index, element in enumerate(unique)
        }

        matrix = np.zeros(
            (length, length)
        )

        for sequence in sequences:
            for index in range(len(sequence) - 1):
                source = sequence[index]
                destination = sequence[index + 1]

                matrix[
                    element_dict[source],
                    element_dict[destination]
                ] += 1

        remove: list[Any] = []

        min_cluster_sample = self.settings.min_cluster_sample

        for index, row in enumerate(matrix):
            if (np.sum(row) < min_cluster_sample) or (
                unique[index - len(remove)] in self.settings.ignore
            ):
                matrix = np.delete(
                    matrix, index - len(remove), 0
                )

                matrix = np.delete(
                    matrix, index - len(remove), 1
                )

                unique = np.delete(
                    unique,
                    index - len(remove)
                )

                remove.append(index)

        matrix = matrix / np.sum(matrix, axis=0)

        self.component['matrix'] = matrix
        self.component['element_dict'] = element_dict
        self.component['remove'] = remove

        return self

    def centers(self) -> Self:
        # Compute the center(s) of each label
        element_centers = {
            label: np.mean(
                self.embedding[self.label == label],
                axis=0
            )
            for label in np.unique(self.label)
        }

        self.component['element_centers'] = element_centers

        return self

    def centroid(self) -> Self:
        ax = self.component.get('ax')
        location = self.component.get('location')

        element_dict_r = self.component.get('element_dict_r')
        label = self.component.get('label')
        position = self.component.get('position')

        color = [
            label[element_dict_r[i]]
            for i in position
        ]

        # Centroids
        ax.scatter(
            location[:, 0],
            location[:, 1],
            color=np.array(color) * 0.5,
            s=250,
            zorder=100,
        )

        ax.scatter(
            location[:, 0],
            location[:, 1],
            color=color,
            s=150,
            zorder=100
        )

        return self

    def compute(self) -> Self:
        column_names = self.component.get('true')
        matrix = self.component.get('matrix')

        # Add all nodes to the list
        graph = nx.DiGraph()

        # For each item in the node list get outgoing connection of the
        # last item in the list from the dense array
        length = len(matrix)

        if column_names is None:
            column_names = np.arange(length)

        for out_node in np.arange(length):
            in_list = np.array(
                np.where(matrix[out_node] > self.settings.min_connection)[0]
            )

            for in_node in in_list:
                graph.add_edge(
                    column_names[out_node],
                    column_names[in_node]
                )

                graph[column_names[out_node]][column_names[in_node]]['weight'] = matrix[
                    out_node
                ][in_node]

        # Remove self-loop(s)
        graph.remove_edges_from(
            nx.selfloop_edges(graph)
        )

        self.component['graph'] = graph

        return self

    def drop(self) -> Self:
        matrix = self.component.get('matrix')
        remove = self.component.get('remove')

        true_label: list[int] = []
        used_drop_list: list[int] = []

        length = len(matrix)

        for i in range(length):
            for drop in remove:
                if i + len(used_drop_list) >= drop:
                    if drop not in used_drop_list:
                        used_drop_list.append(drop)

            true_label.append(i + len(used_drop_list))

        self.component['true'] = true_label

        return self

    def edges(self) -> Self:
        ax = self.component.get('ax')
        graph = self.component.get('graph')
        position = self.component.get('position')

        graph_weights = [
            graph[edge[0]][edge[1]]['weight']
            for edge in graph.edges()
        ]

        edge_color = [
            [0, 0, 0, i]
            for i in graph_weights
        ]

        nx.draw_networkx_edges(
            graph,
            position,
            ax=ax,
            edge_color=edge_color,
            width=2,
        )

        return self

    def element(self) -> Self:
        element_dict = self.component.get('element_dict')

        element_dict_r = {
            v: k
            for k, v in element_dict.items()
        }

        self.component['element_dict_r'] = element_dict_r

        return self

    def nodes(self) -> Self:
        graph = self.component.get('graph')
        element_centers = self.component.get('element_centers')
        element_dict_r = self.component.get('element_dict_r')
        position = self.component.get('position')

        # Graph positions
        position = nx.random_layout(graph)

        position = {
            i: element_centers[element_dict_r[i]]
            for i in position
        }

        # Get locations
        values = list(position.values())
        location = np.vstack(values)

        self.component['location'] = location
        self.component['position'] = position

        return self

    def legend(self) -> Self:
        element_dict_r = self.component.get('element_dict_r')
        label = self.component.get('label')
        position = self.component.get('position')

        test = {
            element_dict_r[i]: label[element_dict_r[i]]
            for i in list(position.keys())
        }

        test = dict(
            sorted(
                test.items()
            )
        )

        legend = {}

        for key, value in test.items():
            line = Line2D(
                [],
                [],
                color='white',
                linewidth=0,
                marker='o',
                markerfacecolor=value
            )

            legend[key] = line

        legend = dict(
            sorted(
                legend.items()
            )
        )

        plt.legend(
            legend.values(),
            legend.keys(),
            bbox_to_anchor=[1.1, 0.5],
            loc='center',
            ncol=1
        )

        return self

    def sequences(self) -> Self:
        sequences = [
            self.label[self.sequence == sequence]
            for sequence in np.unique(self.sequence)
        ]

        self.component['sequences'] = sequences

        return self

    def title(self) -> Self:
        ax = self.component.get('ax')

        title = f"Network for {self.settings.name} using {self.settings.cluster} Clustering"

        ax.set_title(
            title,
            fontsize=18,
            pad=25
        )

        return self


class NetworkFCM(Plot):
    def __init__(self, builder: Any = None):
        super().__init__(builder)

        self.builder = builder

    def build(self) -> dict[Any, Any]:
        cluster = {'cluster': 'Fuzzy C-Means'}
        self.builder.settings.update(cluster)

        return (
            self.builder
            .ax()
            .title()
            .palette()
            .sequences()
            .centers()
            .build()
            .element()
            .drop()
            .compute()
            .nodes()
            .palette()
            .edges()
            .centroid()
            .legend()
            .get()
        )


class NetworkHDBSCAN(Plot):
    def __init__(self, builder: Any = None):
        super().__init__(builder)

        self.builder = builder

    def build(self) -> dict[Any, Any]:
        cluster = {'cluster': 'HDBSCAN'}
        self.builder.settings.update(cluster)

        return (
            self.builder
            .ax()
            .title()
            .palette()
            .sequences()
            .centers()
            .build()
            .element()
            .drop()
            .compute()
            .nodes()
            .palette()
            .edges()
            .centroid()
            .legend()
            .get()
        )
