from __future__ import annotations

import matplotlib.pyplot as plt

from matplotlib.lines import Line2D
from typing_extensions import TYPE_CHECKING
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

    def legend(self) -> Self:
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
            .line()
            .scatter()
            .legend()
            .get()
        )


class ScatterHDBSCAN(Plot):
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
            .line()
            .scatter()
            .legend()
            .get()
        )
