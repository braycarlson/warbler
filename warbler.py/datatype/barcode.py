"""
Barcode
-------

"""


from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from datatype.builder import Base, Plot
from joblib import Parallel, delayed
from matplotlib import gridspec
from matplotlib import ticker
from nltk.metrics.distance import edit_distance
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt

    from matplotlib.axes import Axes
    from typing import Any, Self


def palplot(
    palette: list[tuple[float, ...]],
    size: int = 1,
    ax: Axes | None = None
) -> None:
    length = len(palette)

    if ax is None:
        _, ax = plt.subplots(
            1,
            1,
            figsize=(length * size, size)
        )

    ax.imshow(
        np.arange(length).reshape(1, length),
        cmap=mpl.colors.ListedColormap(list(palette)),
        interpolation='nearest',
        aspect='auto'
    )

    ax.set_xticks(np.arange(length) - 0.5)
    ax.set_yticks([-0.5, 0.5])

    # Ensure nice border between colors
    ax.set_xticklabels(
        ['' for _ in range(length)]
    )

    ax.xaxis.set_major_locator(
        ticker.NullLocator()
    )

    ax.yaxis.set_major_locator(
        ticker.NullLocator()
    )


class SongBarcode():
    def __init__(
        self,
        onset: list[float] | None = None,
        offset: list[float] | None = None,
        label: list[int] | None = None,
        mapping: dict[int, str] | None = None,
        palette: dict[str, tuple[float, ...]] | None = None
    ):
        self._onset = onset
        self._offset = offset
        self._label = label
        self._mapping = mapping
        self._palette = palette

        self.color = None
        self.transition = None

    @property
    def onset(self) -> list[float] | None:
        return self._onset

    @onset.setter
    def onset(self, onset: list[float]) -> None:
        self._onset = onset

    @property
    def offset(self) -> list[float] | None:
        return self._offset

    @offset.setter
    def offset(self, offset: list[float]) -> None:
        self._offset = offset

    @property
    def label(self) -> list[int] | None:
        return self._label

    @label.setter
    def label(self, label: list[int]) -> None:
        self._label = label

    @property
    def mapping(self) -> dict[int, str] | None:
        return self._mapping

    @mapping.setter
    def mapping(self, mapping: dict[int, str]) -> None:
        self._mapping = mapping

    @property
    def palette(self) -> dict[str, tuple[float, ...]] | None:
        return self._palette

    @palette.setter
    def palette(self, palette: dict[str, tuple[float, ...]]) -> None:
        self._palette = palette

    def build(self) -> tuple[npt.NDArray, npt.NDArray]:
        self.transitions()
        self.colors()

        return (self.transition, self.color)

    def colors(self) -> None:
        color = [
            self.palette[i]
            if i in self.palette else [1, 1, 1]
            for i in self.transition
        ]

        self.color = np.expand_dims(color, 1)

    def transitions(self) -> None:
        begin = np.min(self.onset)
        end = np.max(self.offset)
        resolution = 0.01

        transition = (
            np.zeros(
                int(
                    (end - begin) / resolution
                )
            )
            .astype('str')
            .astype('object')
        )

        for start, stop, label in zip(self.onset, self.offset, self.label, strict=True):
            transition[
                int((start - begin) / resolution):
                int((stop - begin) / resolution)
            ] = self.mapping[label]

        self.transition = transition


class Builder(Base):
    def __init__(
        self,
        ax: Axes | None = None,
        color: list[npt.NDArray] | None = None,
        transition: list[npt.NDArray] | None = None
    ):
        super().__init__()

        self._ax = ax
        self._color = color
        self._transition = transition

    @property
    def ax(self) -> Axes | None:
        return self._ax

    @ax.setter
    def ax(self, ax: Axes) -> None:
        self._ax = ax

    @property
    def color(self) -> list[npt.NDArray] | None:
        return self._color

    @color.setter
    def color(self, color: list[npt.NDArray]) -> None:
        self._color = color

    @property
    def transition(self) -> list[npt.NDArray] | None:
        return self._transition

    @transition.setter
    def transition(self, transition: list[npt.NDArray]) -> None:
        self._transition = transition

    def cluster(self) -> Self:
        # Hierarchical clustering
        distance = self.component.get('distance')

        distance = squareform(distance)
        linkage_matrix = linkage(distance, 'single')

        self.component['linkage'] = linkage_matrix

        return self

    def cut(self) -> Self:
        # Make a list of symbols padded to equal length
        self.transition = np.array(self.transition, dtype='object')

        cut = [
            list(i[:self.settings.length].astype('str'))
            if len(i) >= self.settings.length
            else list(i) + list(
                np.zeros(self.settings.length - len(i)).astype('str')
            )
            for i in self.transition
        ]

        cut = [
            ''.join(np.array(i).astype('str'))
            for i in cut
        ]

        self.component['cut'] = cut

        return self

    def dendrogram(self) -> Self:
        linkage = self.component.get('linkage')

        if self.ax is None:
            self._grid()

            ax = self.component.get('axx')

            dn = dendrogram(
                linkage,
                ax=ax,
                get_leaves=True,
                link_color_func=lambda k: 'k',
                no_labels=True,
                orientation='left',
                p=6,
                show_contracted=False,
                truncate_mode='none',
            )

            ax.axis('off')
        else:
            dn = dendrogram(
                linkage,
                get_leaves=True,
                link_color_func=lambda k: 'k',
                no_labels=True,
                no_plot=True,
                orientation='left',
                p=6,
                show_contracted=False,
                truncate_mode='none'
            )

        self.component['dendrogram'] = dn

        return self

    def distance(self) -> Self:
        cut = self.component.get('cut')
        length = len(cut)

        # Create a distance matrix
        matrix = np.zeros(
            (length, length)
        )

        items = [
            (i, j) for i in range(1, length)
            for j in range(0, i)
        ]

        distances = Parallel(n_jobs=4)(
            delayed(edit_distance)(cut[i], cut[j])
            for i, j in items
        )

        for distance, (i, j) in zip(distances, items, strict=True):
            matrix[i, j] = distance
            matrix[j, i] = distance

        self.component['distance'] = matrix

        return self

    def _grid(self) -> Self:
        figsize = self.settings.figure.get('figsize')
        figure = plt.figure(figsize=figsize)

        grid = gridspec.GridSpec(
            1,
            2,
            width_ratios=[1, 10],
            wspace=0,
            hspace=0
        )

        x, y = grid

        axx = plt.subplot(x)
        axy = plt.subplot(y)

        self.component['figure'] = figure
        self.component['grid'] = grid
        self.component['axx'] = axx
        self.component['axy'] = axy

        return self

    def image(self) -> Self:
        colors = self.component.get('colors')
        dendrogram = self.component.get('dendrogram')

        leaves = dendrogram.get('leaves')
        leaves = np.array(leaves)

        image = self.ax.imshow(
            colors.swapaxes(0, 1)[leaves],
            aspect='auto',
            interpolation=None,
            origin='lower',
        )

        self.ax.axis('off')

        self.component['image'] = image

        return self

    def matrix(self) -> Self:
        # Subset dataset
        self.color = self.color[:self.settings.nex]
        self.transition = self.transition[:self.settings.nex]

        # Get length of lists
        length = [len(i) for i in self.transition]

        # Set maximum length
        if self.settings.maximum is None:
            self.settings.maximum = np.max(length)

        # Make a matrix for color representations of syllables
        colors = np.ones(
            (self.settings.maximum, len(length), 3)
        )

        for li, _list in enumerate(self.color):
            colors[: len(_list), li, :] = np.squeeze(
                _list[:self.settings.maximum]
            )

        colors.swapaxes(0, 1)

        self.component['colors'] = colors

        return self

    def title(self) -> Self:
        """Sets the title of the plot based on settings.

        Args:
            None.

        Returns:
            The modified Builder instance.

        """

        title = f"Barcode for {self.settings.name} using {self.settings.cluster} Clustering"

        self.ax.set_title(
            title,
            fontsize=18,
            pad=125
        )

        return self


class BarcodeFCM(Plot):
    def build(self) -> dict[Any, Any]:
        cluster = {'cluster': 'Fuzzy C-Means'}
        self.builder.settings.update(cluster)

        return (
            self.builder
            .matrix()
            .cut()
            .distance()
            .cluster()
            .dendrogram()
            .image()
            .title()
            .get()
        )


class BarcodeHDBSCAN(Plot):
    def build(self) -> dict[Any, Any]:
        cluster = {'cluster': 'HDBSCAN'}
        self.builder.settings.update(cluster)

        return (
            self.builder
            .matrix()
            .cut()
            .distance()
            .cluster()
            .dendrogram()
            .image()
            .title()
            .get()
        )
