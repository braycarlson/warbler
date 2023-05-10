"""
Scatter
-------

"""

from __future__ import annotations

import matplotlib.pyplot as plt

from collections import defaultdict
from datatype.builder import Base, Plot
from datatype.settings import Settings
from matplotlib.lines import Line2D
from typing import Any, Self


class Builder(Base):
    """A class for building plots with customizable settings.

    Args:
        None.

    Returns:
        An instance of the Builder class.

    """

    def ax(self) -> Self:
        """Creates a new figure and axes for the plot.

        Args:
            None.

        Returns:
            The modified Builder instance.

        """

        figsize = self.settings.figure.get('figsize')
        figure, ax = plt.subplots(figsize=figsize)

        ax.xaxis.set_visible(self.settings.is_axis)
        ax.yaxis.set_visible(self.settings.is_axis)

        self.component['ax'] = ax
        self.component['figure'] = figure

        return self

    def initialize(self) -> Self:
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
            merge: defaultdict = defaultdict(dict)
            merge.update(default)

            for key, value in self.settings.items():
                if isinstance(value, dict):
                    merge[key].update(value)
                else:
                    merge[key] = value

            default = dict(merge)

        self.settings = Settings.from_dict(default)

        return self

    def legend(self) -> Self:
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

        ax = self.component.get('ax')
        handles = self.component.get('handles')

        ax.legend(
            handles=handles,
            **self.settings.legend
        )

        return self

    def line(self) -> Self:
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

        label = self.component.get('label')

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

        self.component['handles'] = handles

        return self

    def scatter(self) -> Self:
        """Adds scatter points to the plot.

        Args:
            None.

        Returns:
            The modified Builder instance.

        """

        ax = self.component.get('ax')

        if self.settings.is_color:
            color = self.component.get('color')
            self.settings.scatter['color'] = color

        if self.settings.is_legend:
            label = self.component.get('label')
            self.settings.scatter['label'] = label

        ax.scatter(
            self.embedding[:, 0],
            self.embedding[:, 1],
            **self.settings.scatter
        )

        return self

    def title(self) -> Self:
        """Sets the title of the plot based on settings.

        Args:
            None.

        Returns:
            The modified Builder instance.

        """

        ax = self.component.get('ax')

        name = self.settings.name
        cluster = self.settings.cluster

        title = f"{cluster} Clustering for {name}"

        ax.set_title(
            title,
            fontsize=18,
            pad=25
        )

        return self


class ScatterFCM(Plot):
    """A class for building a Fuzzy C-Means scatter plot.

    Args:
        None.

    Returns:
        Component: The constructed scatter plot component.

    """

    def build(self) -> dict[Any, Any]:
        """Builds the scatter plot component with Fuzzy C-Means settings.

        Args:
            None.

        Returns:
            Component: The constructed scatter plot component.

        """

        self.builder.settings['cluster'] = 'Fuzzy C-Means'

        return (
            self.builder
            .initialize()
            .ax()
            .title()
            .palette()
            .line()
            .scatter()
            .legend()
            .get()
        )


class ScatterHDBSCAN(Plot):
    """A class for building an HDBSCAN scatter plot.

    Args:
        None.

    Returns:
        Component: The constructed scatter plot component.

    """

    def build(self) -> dict[Any, Any]:
        """Builds the scatter plot component with HDBSCAN settings.

        Args:
            None.

        Returns:
            Component: The constructed scatter plot component.

        """

        self.builder.settings['cluster'] = 'HDBSCAN'

        return (
            self.builder
            .initialize()
            .ax()
            .title()
            .palette()
            .line()
            .scatter()
            .legend()
            .get()
        )
