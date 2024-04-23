from __future__ import annotations

import matplotlib.pyplot as plt

from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from typing import TYPE_CHECKING
from warbler.datatype.builder import Plot
from warbler.datatype.voronoi import Builder

if TYPE_CHECKING:
    import numpy.typing as npt

    from typing_extensions import Any, Self
    from warbler.datatype.settings import Settings


class GradientBuilder(Builder):
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
            .scatter()
            .colorbar()
            .axes()
            .limit()
            .voronoi()
            .spacing()
            .diagonal()
            .get()
        )
