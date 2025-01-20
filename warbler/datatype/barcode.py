from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from joblib import Parallel, delayed
from matplotlib import gridspec
from matplotlib import ticker
from nltk.metrics.distance import edit_distance
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from typing_extensions import TYPE_CHECKING
from warbler.datatype.builder import Base, Plot

if TYPE_CHECKING:
    import numpy.typing as npt

    from matplotlib.axes import Axes
    from typing_extensions  import Any, Self


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


class SongBarcode:
    def __init__(
        self,
        onset: list[float] | None = None,
        offset: list[float] | None = None,
        label: list[int] | None = None,
        mapping: dict[int, str] | None = None,
        palette: dict[str, tuple[float, ...]] | None = None
    ):
        self.onset = onset
        self.offset = offset
        self.label = label
        self.mapping = mapping
        self.palette = palette

        self.color = None
        self.transition = None

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

        for start, stop, label in zip(self.onset, self.offset, self.label):
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

        self.ax = ax
        self.color = color
        self.transition = transition

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

        for distance, (i, j) in zip(distances, items):
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
        title = f"Barcode for {self.settings.name} using {self.settings.cluster} Clustering"

        self.ax.set_title(
            title,
            fontsize=18,
            pad=125
        )

        return self


class BarcodeFCM(Plot):
    def __init__(self, builder: Any = None):
        super().__init__(builder)

        self.builder = builder

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
    def __init__(self, builder: Any = None):
        super().__init__(builder)

        self.builder = builder

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
