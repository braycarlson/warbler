"""
Gradient
--------

"""

from __future__ import annotations

import matplotlib.pyplot as plt

from datatype.builder import Plot
from datatype.voronoi import Builder
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Any, Self


class GradientBuilder(Builder):
    def __init__(self):
        super().__init__()

    """A class for constructing plots.

    Args:
        None.

    Returns:
        None.

    """

    def colorbar(self) -> Self:
        axis = self.component.get('axis')
        scatter = self.component.get('scatter')

        label = self.settings.colorbar.get('label')

        cax = inset_axes(
            axis,
            width='100%',
            height='100%',
            loc=5,
            bbox_to_anchor=[1.16, 0.2, 0.05, 0.6],
            bbox_transform=axis.transAxes
        )

        colorbar = plt.colorbar(scatter, cax=cax)

        colorbar.set_label(label, labelpad=20)

        self.component['colorbar'] = colorbar

        return self

    def diagonal(self) -> Self:
        if not self.settings.is_diagonal:
            return self

        axis = self.component.get('axis')

        line = Line2D(
            [0, 1],
            [0, 1],
            lw=3,
            color='red',
            transform=axis.transAxes
        )

        axis.add_line(line)

        return self

    def scatter(self) -> Self:
        """Add scatter points to the plot.

        Args:
            None.

        Returns:
            The modified Builder instance.

        """

        axis = self.component.get('axis')

        self.settings.scatter.pop('color')

        if self.settings.is_color and self.label is not None:
            self.settings.scatter['c'] = self.label

            self.settings.scatter['norm'] = Normalize(
                vmin=min(self.label),
                vmax=max(self.label)
            )

        scatter = axis.scatter(
            self.embedding[:, 0],
            self.embedding[:, 1],
            cmap='viridis',

            **self.settings.scatter
        )

        self.component['scatter'] = scatter

        return self


class VoronoiGradient(Plot):
    """A class for building Voronoi plots.

    Args:
        None.

    Returns:
        None.

    """

    def build(self) -> dict[Any, Any]:
        """Build the Voronoi plot.

        Args:
            None.

        Returns:
            The components.

        """

        cluster = self.builder.settings.cluster
        self.builder.settings['cluster'] = cluster

        return (
            self.builder
            .grid()
            .range()
            .mask()
            .axis()
            .title()
            .scatter()
            .colorbar()
            .axes()
            .limit()
            .voronoi()
            .spacing()
            .diagonal()
            .get()
        )
