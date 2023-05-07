from __future__ import annotations

import matplotlib.pyplot as plt

from collections import defaultdict
from datatype.builder import Base, Component, Plot
from datatype.settings import Settings
from matplotlib.lines import Line2D


class Builder(Base):
    """A class for building plots with customizable settings.

    Args:
        None.

    Returns:
        An instance of the Builder class.

    """

    def ax(self) -> Self:  # noqa
        """Creates a new figure and axes for the plot.

        Args:
            None.

        Returns:
            The modified Builder instance.

        """

        figsize = self.settings.figure.get('figsize')
        fig, ax = plt.subplots(figsize=figsize)

        ax.xaxis.set_visible(self.settings.is_axis)
        ax.yaxis.set_visible(self.settings.is_axis)

        self.component.collection['ax'] = ax

        return self

    def decorate(self) -> Self:  # noqa
        """Sets default settings and updates them with custom settings.

        Args:
            None.

        Returns:
            The modified Builder instance.

        """

        default = {
            'figure': {
                'figsize': (9, 8)
            },
            'legend': {
                'borderaxespad': 0,
                'bbox_to_anchor': (1.15, 1.00),
            },
            'line': {
                'marker': 'o',
                'rasterized': False
            },
            'scatter': {
                'alpha': 0.50,
                'color': 'black',
                'label': None,
                'rasterized': False,
                's': 10
            },
            'cluster': 'HDBSCAN',
            'is_axis': False,
            'is_cluster': True,
            'is_color': True,
            'is_legend': True,
            'is_title': True,
            'name': 'Adelaide\'s warbler',
            'palette': 'tab20',
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

    def legend(self) -> Self:  # noqa
        """Adds a legend to the plot if settings allow.

        Args:
            None.

        Returns:
            The modified Builder instance.

        """

        condition = (
            not self.settings.is_legend or
            not self.settings.is_color
        )

        if condition:
            return self

        ax = self.component.collection.get('ax')
        handles = self.component.collection.get('handles')

        ax.legend(
            handles=handles,
            **self.settings.legend
        )

        return self

    def line(self) -> Self:  # noqa
        """Adds line handles to the plot for legend and color settings.

        Args:
            None.

        Returns:
            The modified Builder instance.

        """

        condition = (
            not self.settings.is_legend or
            not self.settings.is_color
        )

        if condition:
            return self

        label = self.component.collection.get('label')

        handles = [
            Line2D(
                [0],
                [0],
                color=color,
                label=label,
                **self.settings.line
            )
            for label, color in label.items()
        ]

        self.component.collection['handles'] = handles

        return self

    def scatter(self) -> Self:  # noqa
        """Adds scatter points to the plot.

        Args:
            None.

        Returns:
            The modified Builder instance.

        """

        ax = self.component.collection.get('ax')

        if self.settings.is_color:
            color = self.component.collection.get('color')
            self.settings.scatter['color'] = color

        if self.settings.is_legend:
            label = self.component.collection.get('label')
            self.settings.scatter['label'] = label

        ax.scatter(
            self.embedding[:, 0],
            self.embedding[:, 1],
            **self.settings.scatter
        )

        return self

    def title(self) -> Self:  # noqa
        """Sets the title of the plot based on settings.

        Args:
            None.

        Returns:
            The modified Builder instance.

        """

        ax = self.component.collection.get('ax')

        name = self.settings.name
        cluster = self.settings.cluster

        title = f"{cluster} Clustering of {name}"
        ax.set_title(title)

        return self


class ScatterHDBSCAN(Plot):
    """A class for constructing an HDBSCAN scatter plot.

    Args:
        None.

    Returns:
        Component: The constructed scatter plot component.

    """

    def construct(self) -> Component:
        """Constructs the scatter plot component with HDBSCAN settings.

        Args:
            None.

        Returns:
            Component: The constructed scatter plot component.

        """

        self.builder.settings['cluster'] = 'HDBSCAN'

        return (
            self.builder
            .decorate()
            .ax()
            .title()
            .palette()
            .line()
            .scatter()
            .legend()
            .get()
        )


class ScatterFCM(Plot):
    """A class for constructing a Fuzzy C-Means scatter plot.

    Args:
        None.

    Returns:
        Component: The constructed scatter plot component.

    """

    def construct(self) -> Component:
        """Constructs the scatter plot component with Fuzzy C-Means settings.

        Args:
            None.

        Returns:
            Component: The constructed scatter plot component.

        """

        self.builder.settings['cluster'] = 'Fuzzy C-Means'

        return (
            self.builder
            .decorate()
            .ax()
            .title()
            .palette()
            .line()
            .scatter()
            .legend()
            .get()
        )
