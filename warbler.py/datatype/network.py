from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns

from collections import defaultdict
from datatype.builder import Base, Component, Plot
from datatype.settings import Settings
from matplotlib.lines import Line2D


class Builder(Base):
    def ax(self) -> Self:  # noqa
        """Create a new figure and axis.

        Returns:
            The modified Builder instance.

        """

        figsize = self.settings.figure.get('figsize')
        fig, ax = plt.subplots(figsize=figsize)

        ax.xaxis.set_visible(self.settings.is_axis)
        ax.yaxis.set_visible(self.settings.is_axis)

        self.component.collection['ax'] = ax

        return self

    def build(self) -> Self:  # noqa
        """Build the transition matrix and other data structures.

        Returns:
            The modified Builder instance.

        """

        sequences = self.component.collection.get('sequences')

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

        # # breakme
        # remove = []

        # min_cluster_sample = self.settings.min_cluster_sample

        # for index, row in enumerate(matrix):
        #     if (np.sum(row) < min_cluster_sample) or (
        #         unique[index - len(remove)] in self.settings.ignore
        #     ):
        #         matrix = np.delete(
        #             matrix, index - len(remove), 0
        #         )

        #         matrix = np.delete(
        #             matrix, index - len(remove), 1
        #         )

        #         unique = np.delete(
        #             unique,
        #             index - len(remove)
        #         )

        #         remove.append(index)

        matrix = matrix / np.sum(matrix, axis=0)

        self.component.collection['matrix'] = matrix
        self.component.collection['element_dict'] = element_dict
        self.component.collection['remove'] = []

        return self

    def centroid(self) -> Self:  # noqa
        """Plot the centroids of each label.

        Returns:
            The modified Builder instance.

        """

        ax = self.component.collection.get('ax')
        location = self.component.collection.get('location')

        element_dict_r = self.component.collection.get('element_dict_r')
        label = self.component.collection.get('label')
        position = self.component.collection.get('position')

        color = [
            label[element_dict_r[i]]
            for i in position.keys()
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

    def compute(self) -> Self:  # noqa
        """Compute the transition graph from the transition matrix.

        Returns:
            The modified Builder instance.

        """

        column_names = self.component.collection.get('column_names')
        matrix = self.component.collection.get('matrix')

        # Add all nodes to the list
        graph = nx.DiGraph()
        # For each item in the node list get outgoing connection of the
        #   *last* item in the list from the dense array
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

        # Remove noise
        graph.remove_node(0)

        self.component.collection['graph'] = graph

        return self

    def decorate(self) -> Self:  # noqa
        """Set the default settings and update them with the custom settings.

        Returns:
            The modified Builder instance.

        """

        default = {
            'figure': {
                'figsize': (9, 8)
            },
            'ignore': [-1],
            'is_axis': False,
            'is_cluster': True,
            'is_color': True,
            'is_legend': True,
            'is_title': True,
            'min_cluster_sample': 0,
            'min_connection': 0.05,
            'palette': 'tab20'
        }

        if self.settings is not None:
            merge = defaultdict(dict)
            merge.update(default)

            for key, value in self.settings.items():
                if isinstance(value, dict):
                    merge[key].update(value)
                else:
                    merge[key] = value

            default = dict(merge)

        self.settings = Settings.from_dict(default)

        return self

    def drop(self) -> Self:  # noqa
        """Drop rows and columns from the transition matrix.

        Returns:
            The modified Builder instance.

        """
        matrix = self.component.collection.get('matrix')
        remove = self.component.collection.get('remove')

        true_label = []
        used_drop_list = []

        length = len(matrix)

        for i in range(length):
            for drop in remove:
                if i + len(used_drop_list) >= drop:
                    if drop not in used_drop_list:
                        used_drop_list.append(drop)

            true_label.append(i + len(used_drop_list))

        return self

    def edges(self) -> Self:  # noqa
        """Plot the edges of the transition graph.

        Returns:
            The modified Builder instance.

        """

        ax = self.component.collection.get('ax')
        graph = self.component.collection.get('graph')
        position = self.component.collection.get('position')

        graph_weights = [
            graph[edge[0]][edge[1]]['weight']
            for edge in graph.edges()
        ]

        edge_color = [
            [0, 0, 0] + [i]
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

    def element(self) -> Self:  # noqa
        """Create a dictionary mapping element index to label.

        Returns:
            The modified Builder instance.

        """

        element_dict = self.component.collection.get('element_dict')

        element_dict_r = {
            v: k
            for k, v in element_dict.items()
        }

        self.component.collection['element_dict_r'] = element_dict_r

        return self

    def nodes(self) -> Self:  # noqa
        """Compute the positions of the nodes in the transition graph.

        Returns:
            The modified Builder instance.

        """

        graph = self.component.collection.get('graph')
        element_centers = self.component.collection.get('element_centers')
        element_dict_r = self.component.collection.get('element_dict_r')
        position = self.component.collection.get('position')

        # Graph positions
        position = nx.random_layout(graph)

        position = {
            i: element_centers[element_dict_r[i]]
            for i in position.keys()
        }

        # Get locations
        values = list(position.values())
        location = np.vstack(values)

        self.component.collection['location'] = location
        self.component.collection['position'] = position

        return self

    def legend(self) -> Self:  # noqa
        """Plot the legend for the labels.

        Returns:
            The modified Builder instance.

        """

        element_dict_r = self.component.collection.get('element_dict_r')
        label = self.component.collection.get('label')
        position = self.component.collection.get('position')

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

        # if -1 in legend:
        #     del legend[-1]

        plt.legend(
            legend.values(),
            legend.keys(),
            bbox_to_anchor=[1.1, 0.5],
            loc='center',
            ncol=1
        )

        return self

    def sequences(self) -> Self:  # noqa
        """Create a list of sequences based on the labels.

        Returns:
            The modified Builder instance.

        """

        sequences = [
            self.label[self.sequence == sequence]
            for sequence in np.unique(self.sequence)
        ]

        self.component.collection['sequences'] = sequences

        return self

    def centers(self) -> Self:  # noqa
        """Compute the center(s) of each label.

        Returns:
            The modified Builder instance.

        """

        # Compute the center(s) of each label
        element_centers = {
            label: np.mean(
                self.embedding[self.label == label],
                axis=0
            )
            for label in np.unique(self.label)
        }

        self.component.collection['element_centers'] = element_centers

        return self


class Network(Plot):
    def construct(self) -> Component:
        """Construct the network graph.

        Returns:
            Component: The constructed network graph component.

        """

        return (
            self.builder
            .decorate()
            .ax()
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


def plot_network_graph(
    labels,
    embedding,
    sequences,
    n_colors=None,
    original=None,
    color_palette='tab20',
    ax=None,
    min_cluster_samples=0,
    pal_dict=None,
):
    sequences = [
        labels[sequences == sequence]
        for sequence in np.unique(sequences)
    ]

    # Compute the center(s) of each label
    element_centers = {
        label: np.mean(
            embedding[labels == label],
            axis=0
        )
        for label in np.unique(labels)
    }

    # Create a transition matrix of elements
    transition_matrix, element_dict, drop_list = build_transition_matrix(
        sequences,
        min_cluster_samples=min_cluster_samples
    )

    element_dict_r = {
        v: k
        for k, v in element_dict.items()
    }

    true_label = []
    used_drop_list = []

    length = len(transition_matrix)

    for i in range(length):
        for drop in drop_list:
            if i + len(used_drop_list) >= drop:
                if drop not in used_drop_list:
                    used_drop_list.append(drop)

        true_label.append(i + len(used_drop_list))

    # Generate graph
    graph = compute_graph(
        transition_matrix,
        min_connections=0.05,
        column_names=true_label
    )

    # Remove self-loop(s)
    graph.remove_edges_from(
        nx.selfloop_edges(graph)
    )

    # graph.remove_node(0)

    # Graph positions
    position = nx.random_layout(graph)

    position = {
        i: element_centers[element_dict_r[i]]
        for i in position.keys()
    }

    # Get the palette
    if pal_dict is None:
        palette = sns.color_palette(
            color_palette,
            n_colors=n_colors
        )

        master = dict(
            zip(np.unique(original), palette)
        )

        label = {
            label: master[label]
            for label in labels[labels > -1]
        }

        label = dict(
            sorted(
                label.items()
            )
        )

        pos_colors = [
            label[element_dict_r[i]]
            for i in position.keys()
        ]

        test = {
            element_dict_r[i]: label[element_dict_r[i]]
            for i in list(position.keys())
        }

        test = dict(
            sorted(
                test.items()
            )
        )
    else:
        pos_colors = [
            pal_dict[element_dict_r[i]]
            for i in position.keys()
        ]

    # Get locations
    values = list(position.values())
    location = np.vstack(values)

    # Plot
    if ax is None:
        figsize = (9, 8)
        fig, ax = plt.subplots(figsize=figsize)

    graph_weights = [
        graph[edge[0]][edge[1]]['weight']
        for edge in graph.edges()
    ]

    edge_color = [
        [0, 0, 0] + [i]
        for i in graph_weights
    ]

    nx.draw_networkx_edges(
        graph,
        position,
        ax=ax,
        edge_color=edge_color,
        width=2,
    )

    # Centroids
    ax.scatter(
        location[:, 0],
        location[:, 1],
        color=np.array(pos_colors) * 0.5,
        s=250,
        zorder=100,
    )

    ax.scatter(
        location[:, 0],
        location[:, 1],
        color=pos_colors,
        s=150,
        zorder=100
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

    return ax


def build_transition_matrix(sequences, min_cluster_samples=0, ignore=[-1]):
    """
    Builds a transition matrix from a set of sequences of discrete categories
    If there are fewer than min_cluster_samples samples for a given cluster,
    ignore it in the transition matrix
    """
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

    # breakme
    remove = []

    for index, row in enumerate(matrix):
        if (np.sum(row) < min_cluster_samples) or (
            unique[index - len(remove)] in ignore
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
    return matrix, element_dict, remove


def compute_graph(dense_matrix, min_connections=0.05, column_names=None):
    # Add all nodes to the list
    graph = nx.DiGraph()
    # For each item in the node list get outgoing connection of the
    #   *last* item in the list from the dense array
    length = len(dense_matrix)

    if column_names is None:
        column_names = np.arange(length)

    for out_node in np.arange(length):
        in_list = np.array(
            np.where(dense_matrix[out_node] > min_connections)[0]
        )

        for in_node in in_list:
            graph.add_edge(
                column_names[out_node],
                column_names[in_node]
            )

            graph[column_names[out_node]][column_names[in_node]]['weight'] = dense_matrix[
                out_node
            ][in_node]

    return graph
