from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from matplotlib.lines import Line2D
from matplotlib import gridspec
from scipy.spatial import cKDTree
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

    def axis(self) -> Self:
        figure = self.component.get('figure')
        grid = self.component.get('grid')

        axis = figure.add_subplot(
            grid[
                1:self.settings.size - 1,
                1:self.settings.size - 1
            ]
        )

        if self.settings.is_label:
            axis.set_xlabel(
                self.settings.x,
                fontsize=20,
                labelpad=30
            )

            axis.set_ylabel(
                self.settings.y,
                fontsize=20,
                labelpad=30
            )

        axis.set_xticks([])
        axis.set_yticks([])

        self.component['axis'] = axis

        return self

    def axes(self) -> Self:
        axs = {}

        figure = self.component.get('figure')
        grid = self.component.get('grid')

        n_column = self.component.get('n_column')

        xmin = self.component.get('xmin')
        ymax = self.component.get('ymax')
        x_block = self.component.get('x_block')
        y_block = self.component.get('y_block')

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

        self.component['axs'] = axs

        return self

    def grid(self) -> Self:
        n_column = self.settings.size * 4 - 4

        figure = plt.figure(**self.settings.figure)

        grid = gridspec.GridSpec(
            self.settings.size,
            self.settings.size
        )

        self.component['figure'] = figure
        self.component['grid'] = grid
        self.component['n_column'] = n_column

        return self

    def line(self) -> Self:
        if not self.settings.is_legend or self.label is None:
            return self

        label = self.component.get('label')

        if not self.settings.is_numerical:
            label = {
                chr(k + 65): v
                for k, v in label.items()
            }

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

        self.component['handles'] = handles

        return self

    def legend(self) -> Self:
        if not self.settings.is_legend:
            return self

        axis = self.component.get('axis')
        handles = self.component.get('handles')

        axis.legend(
            handles=handles,
            **self.settings.legend
        )

        return self

    def limit(self) -> Self:
        axis = self.component.get('axis')

        xmin = self.component.get('xmin')
        xmax = self.component.get('xmax')
        ymin = self.component.get('ymin')
        ymax = self.component.get('ymax')

        axis.set_xlim(
            [
                xmin + 0.1,
                xmax + 0.1
            ]
        )

        axis.set_ylim(
            [
                ymin + 0.1,
                ymax + 0.1
            ]
        )

        return self

    def mask(self) -> Self:
        xmin = self.component.get('xmin')
        xmax = self.component.get('xmax')
        ymin = self.component.get('ymin')
        ymax = self.component.get('ymax')

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

        if self.label is not None and self.label.size:
            self.label = np.array(self.label)[mask]

            self.spectrogram = np.array(self.spectrogram)[mask]
            self.embedding = self.embedding[mask]

        return self

    def range(self) -> Self:
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

        self.component['xmin'] = xmin
        self.component['xmax'] = xmax
        self.component['ymin'] = ymin
        self.component['ymax'] = ymax
        self.component['x_block'] = x_block
        self.component['y_block'] = y_block

        return self

    def scatter(self) -> Self:
        axis = self.component.get('axis')

        self.settings.scatter['color'] = 'cornflowerblue'

        if self.settings.is_color and self.label is not None:
            color = self.component.get('color')
            self.settings.scatter['color'] = color

        axis.scatter(
            self.embedding[:, 0],
            self.embedding[:, 1],
            **self.settings.scatter
        )

        return self

    def spacing(self) -> Self:
        grid = self.component.get('grid')
        grid.update(**self.settings.grid)

        return self

    def title(self) -> Self:
        axis = self.component.get('axis')

        title = f"{self.settings.cluster} for {self.settings.name}"

        axis.set_title(
            title,
            fontsize=18,
            pad=50
        )

        return self

    def voronoi(self) -> Self:
        mapping = self.component.get('label')
        line = self.settings.voronoi.get('line')
        matshow = self.settings.voronoi.get('matshow')
        scatter = self.settings.voronoi.get('scatter')
        linewidth = self.settings.voronoi.get('linewidth')
        n_subset = self.settings.voronoi.get('n_subset')

        s = scatter.get('s')

        axis = self.component.get('axis')
        axs = self.component.get('axs')
        figure = self.component.get('figure')

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

            if self.settings.is_cluster_color:
                label = self.label[selection]
                color = mapping[label]

                line['color'] = color

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

            if self.settings.is_cluster_color:
                for spine in axs[key]['ax'].spines.values():
                    spine.set_color(color)

            # Draw a line between point and image
            if self.settings.is_line:
                for spine in axs[key]['ax'].spines.values():
                    spine.set_linewidth(linewidth)

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

                line['color'] = '#e7e6e6'

                lines.append(
                    Line2D(
                        [infig_position_start[0], infig_position[0]],
                        [infig_position_start[1], infig_position[1]],
                        transform=figure.transFigure,
                        **line
                    )
                )

        if self.settings.is_line:
            for line in lines:
                figure.lines.append(line)

        self.component['lines'] = lines

        return self


class VoronoiFCM(Plot):
    def __init__(self, builder: Any = None):
        super().__init__(builder)

        self.builder = builder

    def build(self) -> dict[Any, Any]:
        cluster = self.builder.settings.cluster
        self.builder.settings['cluster'] = cluster

        return (
            self.builder
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


class VoronoiHDBSCAN(Plot):
    def __init__(self, builder: Any = None):
        super().__init__(builder)

        self.builder = builder

    def build(self) -> dict[Any, Any]:
        cluster = self.builder.settings.cluster
        self.builder.settings['cluster'] = cluster

        return (
            self.builder
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
