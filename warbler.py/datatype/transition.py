"""
Transition
----------

"""

from __future__ import annotations

import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np
from datatype.builder import Base, Plot
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt

    from typing_extensions import Any, Self


class Builder(Base):
    def ax(self) -> Self:
        figsize = self.settings.figure.get('figsize')
        figure, ax = plt.subplots(figsize=figsize)

        ax.axis('off')

        self.component['ax'] = ax
        self.component['figure'] = figure

        return self

    def _collection(
        self,
        x: npt.NDArray,
        y: npt.NDArray,
        array: npt.NDArray = None
    ) -> Self:
        ax = self.component.get('ax')

        alpha = self.settings.colorline.get('alpha')
        cmap = self.settings.colorline.get('cmap')
        linewidth = self.settings.colorline.get('linewidth')
        norm = self.settings.colorline.get('norm')

        # Default colors equally spaced on [0, 1]:
        if array is None:
            array = np.linspace(
                0.0,
                1.0,
                len(x)
            )

        if not hasattr(array, '__iter__'):
            array = np.array([array])

        array = np.asarray(array)

        line = [x, y]
        points = np.array(line).T.reshape(-1, 1, 2)

        segment = np.concatenate(
            [
                points[:-1],
                points[1:]
            ],
            axis=1
        )

        collection = mcoll.LineCollection(
            segment,
            alpha=alpha,
            array=array,
            cmap=cmap,
            linewidth=linewidth,
            norm=norm
        )

        ax.add_collection(collection)
        self.component['collection'] = collection

        return self

    def colorline(self) -> Self:
        for sequence in np.unique(self.sequence):
            mask = self.sequence == sequence
            embedding = self.embedding[mask]

            length = len(embedding)

            self._collection(
                embedding[:, 0],
                embedding[:, 1],
                np.linspace(0, 1, length),
            )

        return self

    def range(self) -> Self:
        ax = self.component.get('ax')

        xmin, xmax = np.sort(self.embedding[:, 0])[
            np.array(
                [
                    int(len(self.embedding) * 0.01),
                    int(len(self.embedding) * 0.99)
                ]
            )
        ]

        ymin, ymax = np.sort(self.embedding[:, 1])[
            np.array(
                [
                    int(len(self.embedding) * 0.01),
                    int(len(self.embedding) * 0.99)
                ]
            )
        ]

        xmin = xmin - (xmax - xmin) * self.settings.padding
        xmax = xmax + (xmax - xmin) * self.settings.padding
        ymin = ymin - (ymax - ymin) * self.settings.padding
        ymax = ymax + (ymax - ymin) * self.settings.padding

        xlimit = (xmin, xmax)
        ylimit = (ymin, ymax)

        ax.set_xlim(xlimit)
        ax.set_ylim(ylimit)

        return self

    def title(self) -> Self:
        """Sets the title of the plot based on settings.

        Args:
            None.

        Returns:
            The modified Builder instance.

        """

        ax = self.component.get('ax')

        title = f"Transition for {self.settings.name}"

        ax.set_title(
            title,
            fontsize=18,
            pad=25
        )

        return self


class Transition(Plot):
    def build(self) -> dict[Any, Any]:
        return (
            self.builder
            .ax()
            .title()
            .colorline()
            .range()
            .get()
        )
