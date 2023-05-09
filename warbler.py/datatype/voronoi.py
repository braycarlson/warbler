"""
Voronoi
-------

"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from collections import defaultdict
from datatype.builder import Base, Component, Plot
from datatype.settings import Settings
from matplotlib.lines import Line2D
from matplotlib import gridspec
from scipy.spatial import cKDTree


class Builder(Base):
    """A class for constructing plots.

    Args:
        None.

    Returns:
        None.

    """

    def axis(self) -> Self:  # noqa
        """Create an axis for the plot.

        Args:
            None.

        Returns:
            The modified Builder instance.

        """

        figure = self.component.collection.get('figure')
        grid = self.component.collection.get('grid')

        axis = figure.add_subplot(
            grid[
                1:self.settings.size - 1,
                1:self.settings.size - 1
            ]
        )

        self.component.collection['axis'] = axis

        return self

    def axes(self) -> Self:  # noqa
        """Create multiple axes for the plot.

        Args:
            None.

        Returns:
            The modified Builder instance.

        """

        axs = {}

        figure = self.component.collection.get('figure')
        grid = self.component.collection.get('grid')

        n_column = self.component.collection.get('n_column')

        xmin = self.component.collection.get('xmin')
        ymax = self.component.collection.get('ymax')
        x_block = self.component.collection.get('x_block')
        y_block = self.component.collection.get('y_block')

        size = self.settings.size

        for column in range(n_column):
            if column < size:
                row = 0
                col = column
            elif size <= column < (size * 2 - 1):
                row = column - size + 1
                col = size - 1
            elif (size * 2 - 1) <= column < (size * 3 - 2):
                row = size - 1
                col = size - 3 - (column - (size * 2))
            else:
                row = n_column - column
                col = 0

            axs[column] = {
                'ax': figure.add_subplot(grid[row, col]),
                'col': col,
                'row': row
            }

            # Sample a point in z based upon the row and column
            xpos = xmin + x_block * col + x_block / 2
            ypos = ymax - y_block * row - y_block / 2

            axs[column]['xpos'] = xpos
            axs[column]['ypos'] = ypos

        self.component.collection['axs'] = axs

        return self

    def grid(self) -> Self:  # noqa
        """Create a grid for the plot.

        Args:
            None.

        Returns:
            The modified Builder instance.

        """

        n_column = self.settings.size * 4 - 4

        figure = plt.figure(**self.settings.figure)

        grid = gridspec.GridSpec(
            self.settings.size,
            self.settings.size
        )

        self.component.collection['figure'] = figure
        self.component.collection['grid'] = grid
        self.component.collection['n_column'] = n_column

        return self

    def initialize(self) -> Self:  # noqa
        """Set the default settings for the plot.

        Args:
            None.

        Returns:
            The modified Builder instance.

        """

        default = {
            'figure': {
                'figsize': (10, 10)
            },
            'grid': {
                'wspace': 0,
                'hspace': 0
            },
            'legend': {
                'borderaxespad': 0,
                'bbox_to_anchor': (1.20, 1.0772)
            },
            'line': {
                'color': 'white',
                'marker': 'o',
                'rasterized': False
            },
            'scatter': {
                'alpha': 0.50,
                'color': 'black',
                'rasterized': False,
                's': 10
            },
            'voronoi': {
                'line': {
                    'alpha': 1.00,
                    'color': 'black',
                    'ls': 'solid',
                    'lw': 1,
                    'rasterized': False
                },
                'matshow': {
                    'origin': 'upper',
                    'interpolation': 'bicubic',
                    'aspect': 'auto',
                    'cmap': plt.cm.Greys
                },
                'scatter': {
                    'alpha': 0.75,
                    'color': 'black',
                    'rasterized': False,
                    's': 10
                },
                'linewidth': 1,
                'n_subset': -1,
            },
            'color': 'k',
            'is_color': True,
            'is_legend': True,
            'is_line': True,
            'name': 'Adelaide\'s warbler',
            'padding': 0.1,
            'palette': 'tab20',
            'size': 15
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

    def line(self) -> Self:  # noqa
        """Create legend lines for the plot.

        Args:
            None.

        Returns:
            The modified Builder instance.

        """

        if not self.settings.is_legend:
            return self

        label = self.component.collection.get('label')

        handles = [
            Line2D(
                [0],
                [0],
                label=label,
                markerfacecolor=markerfacecolor,
                **self.settings.line
            )
            for label, markerfacecolor in label.items()
        ]

        self.component.collection['handles'] = handles

        return self

    def legend(self) -> Self:  # noqa
        """Add a legend to the plot.

        Args:
            None.

        Returns:
            The modified Builder instance.

        """

        if not self.settings.is_legend:
            return self

        axis = self.component.collection.get('axis')
        handles = self.component.collection.get('handles')

        axis.legend(
            handles=handles,
            **self.settings.legend
        )

        return self

    def limit(self) -> Self:  # noqa
        """Set the limits of the plot.

        Args:
            None.

        Returns:
            The modified Builder instance.

        """

        axis = self.component.collection.get('axis')

        xmin = self.component.collection.get('xmin')
        xmax = self.component.collection.get('xmax')
        ymin = self.component.collection.get('ymin')
        ymax = self.component.collection.get('ymax')

        axis.set_xlim(
            [
                xmin,
                xmax
            ]
        )

        axis.set_ylim(
            [
                ymin,
                ymax
            ]
        )

        return self

    def mask(self) -> Self:  # noqa
        """Determine the range of the plot.

        Args:
            None.

        Returns:
            The modified Builder instance.

        """

        xmin = self.component.collection.get('xmin')
        xmax = self.component.collection.get('xmax')
        ymin = self.component.collection.get('ymin')
        ymax = self.component.collection.get('ymax')

        xmin_mask = self.embedding[:, 0] > xmin
        ymin_mask = self.embedding[:, 1] > ymin
        xmax_mask = self.embedding[:, 0] < xmax
        ymax_mask = self.embedding[:, 1] < ymax

        mask = np.logical_and.reduce(
            [
                xmin_mask,
                ymin_mask,
                xmax_mask,
                ymax_mask
            ]
        )

        if self.label.size:
            self.label = np.array(self.label)[mask]

        self.spectrogram = np.array(self.spectrogram)[mask]
        self.embedding = self.embedding[mask]

        return self

    def range(self) -> Self:  # noqa
        """Determine the range of the plot.

        Args:
            None.

        Returns:
            The modified Builder instance.

        """

        embedding = np.array(self.embedding)
        n = len(embedding)

        x = embedding[:, 0]
        x = np.sort(x)
        xmin = x[int(n * 0.01)]
        xmax = x[int(n * 0.99)]

        y = embedding[:, 1]
        y = np.sort(y)
        ymin = y[int(n * 0.01)]
        ymax = y[int(n * 0.99)]

        xmin = xmin - (xmax - xmin) * self.settings.padding
        xmax = xmax + (xmax - xmin) * self.settings.padding
        ymin = ymin - (ymax - ymin) * self.settings.padding
        ymax = ymax + (ymax - ymin) * self.settings.padding

        x_block = (xmax - xmin) / self.settings.size
        y_block = (ymax - ymin) / self.settings.size

        self.component.collection['xmin'] = xmin
        self.component.collection['xmax'] = xmax
        self.component.collection['ymin'] = ymin
        self.component.collection['ymax'] = ymax
        self.component.collection['x_block'] = x_block
        self.component.collection['y_block'] = y_block

        return self

    def scatter(self) -> Self:  # noqa
        """Add scatter points to the plot.

        Args:
            None.

        Returns:
            The modified Builder instance.

        """

        axis = self.component.collection.get('axis')

        if self.settings.is_color:
            color = self.component.collection.get('color')
            self.settings.scatter['color'] = color

        axis.scatter(
            self.embedding[:, 0],
            self.embedding[:, 1],
            **self.settings.scatter
        )

        return self

    def spacing(self) -> Self:  # noqa
        """Set the spacing of the grid.

        Args:
            None.

        Returns:
            The modified Builder instance.

        """

        grid = self.component.collection.get('grid')
        grid.update(**self.settings.grid)

        return self

    def title(self) -> Self:  # noqa
        """Sets the title of the plot based on settings.

        Args:
            None.

        Returns:
            The modified Builder instance.

        """

        axis = self.component.collection.get('axis')

        title = f"Voronoi of {self.settings.name}"

        axis.set_title(
            title,
            fontsize=18,
            pad=50
        )

        return self

    def voronoi(self) -> Self:  # noqa
        """Create Voronoi regions on the plot.

        Args:
            None.

        Returns:
            The modified Builder instance.

        """

        line = self.settings.voronoi.get('line')
        matshow = self.settings.voronoi.get('matshow')
        scatter = self.settings.voronoi.get('scatter')
        linewidth = self.settings.voronoi.get('linewidth')
        n_subset = self.settings.voronoi.get('n_subset')

        s = scatter.get('s')

        axis = self.component.collection.get('axis')
        axs = self.component.collection.get('axs')
        figure = self.component.collection.get('figure')

        # Create a voronoi diagram over the x and y pos points
        points = [
            [
                axs[i]['xpos'],
                axs[i]['ypos']
            ]
            for i in axs
        ]

        voronoi = cKDTree(points)

        # Find where each point lies in the voronoi diagram
        embedding = self.embedding[:n_subset]

        _, region = voronoi.query(
            list(embedding)
        )

        lines = []

        # Loop through regions and select a point
        for key in axs:
            # Sample a point in (or near) Voronoi region
            nearest = np.argsort(np.abs(region - key))
            possible = np.where(region == region[nearest][0])[0]
            selection = np.random.choice(possible, size=1)[0]
            region[selection] = 1e4

            # Plot point
            if s > 0:
                x = embedding[selection, 0]
                y = embedding[selection, 1]

                axis.scatter(
                    [x],
                    [y],
                    **scatter,
                )

            # Draw spectrogram
            axs[key]['ax'].matshow(
                self.spectrogram[selection],
                **matshow
            )

            axs[key]['ax'].set_xticks([])
            axs[key]['ax'].set_yticks([])

            # plt.setp(
            #     axs[key]['ax'].spines.values(),
            #     color=color[key]
            # )

            for i in axs[key]['ax'].spines.values():
                i.set_linewidth(linewidth)

            # Draw a line between point and image
            if self.settings.is_line:
                mytrans = (
                    axs[key]['ax'].transAxes +
                    axs[key]['ax'].figure.transFigure.inverted()
                )

                position = [0.5, 0.5]

                if axs[key]['row'] == 0:
                    position[1] = 0

                if axs[key]['row'] == self.settings.size - 1:
                    position[1] = 1

                if axs[key]['col'] == 0:
                    position[0] = 1

                if axs[key]['col'] == self.settings.size - 1:
                    position[0] = 0

                infig_position = mytrans.transform(position)

                xpos, ypos = axis.transLimits.transform(
                    (
                        embedding[selection, 0],
                        embedding[selection, 1]
                    )
                )

                transform = (
                    axis.transAxes +
                    axis.figure.transFigure.inverted()
                )

                infig_position_start = transform.transform(
                    [
                        xpos,
                        ypos
                    ]
                )

                lines.append(
                    Line2D(
                        [infig_position_start[0], infig_position[0]],
                        [infig_position_start[1], infig_position[1]],
                        transform=figure.transFigure,
                        **self.settings.voronoi.get('line'),
                    )
                )

        if self.settings.is_line:
            for line in lines:
                figure.lines.append(line)

        self.component.collection['lines'] = lines

        return self


class Voronoi(Plot):
    """A class for building Voronoi plots.

    Args:
        None.

    Returns:
        None.

    """

    def build(self) -> Component:
        """Build the Voronoi plot.

        Args:
            None.

        Returns:
            The components.

        """

        return (
            self.builder
            .initialize()
            .grid()
            .range()
            .mask()
            .axis()
            .title()
            .palette()
            .scatter()
            .line()
            .legend()
            .axes()
            .limit()
            .voronoi()
            .spacing()
            .get()
        )
