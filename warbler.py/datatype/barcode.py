"""
Barcode
-------

"""


from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from collections import defaultdict
from datatype.builder import Base, Component, Plot
from datatype.settings import Settings
from joblib import Parallel, delayed
from matplotlib import gridspec
from matplotlib.axes import Axes
from nltk.metrics.distance import edit_distance
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from typing import Any, Dict, List, Tuple


def palplot(
    palette: List[Tuple[float, float, float]],
    size: int = 1,
    ax: Axes = None
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
        cmap=matplotlib.colors.ListedColormap(list(palette)),
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
        onset=None,
        offset=None,
        label=None,
        mapping=None,
        palette=None
    ):
        self._onset = onset
        self._offset = offset
        self._label = label
        self._mapping = mapping
        self._palette = palette

        self.color = None
        self.transition = None

    @property
    def onset(self) -> List[float]:
        return self._onset

    @onset.setter
    def onset(self, onset: List[float]) -> None:
        self._onset = onset

    @property
    def offset(self) -> List[float]:
        return self._offset

    @offset.setter
    def offset(self, offset: List[float]) -> None:
        self._offset = offset

    @property
    def label(self) -> List[int]:
        return self._label

    @label.setter
    def label(self, label: List[int]) -> None:
        self._label = label

    @property
    def mapping(self) -> Dict[int, str]:
        return self._mapping

    @mapping.setter
    def mapping(self, mapping: Dict[int, str]) -> None:
        self._mapping = mapping

    @property
    def palette(self) -> Dict[str, Tuple[float, float, float]]:
        return self._palette

    @palette.setter
    def palette(self, palette: Dict[str, Tuple[float, float, float]]) -> None:
        self._palette = palette

    def build(self) -> Tuple[np.ndarray, np.ndarray]:
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
    def __init__(self, ax: Axes = None, color=None, transition=None):
        super().__init__()

        self._ax = ax
        self._color = color
        self._transition = transition

    @property
    def ax(self) -> Axes:
        return self._ax

    @ax.setter
    def ax(self, ax: Axes) -> None:
        self._ax = ax

    @property
    def color(self) -> List[np.ndarray]:
        return self._color

    @color.setter
    def color(self, color: List[np.ndarray]) -> None:
        self._color = color

    @property
    def transition(self) -> List[np.ndarray]:
        return self._transition

    @transition.setter
    def transition(self, transition: List[np.ndarray]) -> None:
        self._transition = transition

    def cluster(self) -> Self:  # noqa
        # Hierarchical clustering
        distance = self.component.collection.get('distance')

        distance = squareform(distance)
        linkage_matrix = linkage(distance, 'single')

        self.component.collection['linkage'] = linkage_matrix

        return self

    def cut(self) -> Self:  # noqa
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

        self.component.collection['cut'] = cut

        return self

    def dendrogram(self) -> Self:  # noqa
        linkage = self.component.collection.get('linkage')

        if self.ax is None:
            self._grid()

            ax = self.component.collection.get('axx')

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

        self.component.collection['dendrogram'] = dn

        return self

    def distance(self) -> Self:  # noqa
        cut = self.component.collection.get('cut')
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

        self.component.collection['distance'] = matrix

        return self

    def _grid(self) -> Self:  # noqa
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

        self.component.collection['figure'] = figure
        self.component.collection['grid'] = grid
        self.component.collection['axx'] = axx
        self.component.collection['axy'] = axy

        return self

    def image(self) -> Self:  # noqa
        colors = self.component.collection.get('colors')
        dendrogram = self.component.collection.get('dendrogram')

        leaves = dendrogram.get('leaves')
        leaves = np.array(leaves)

        image = self.ax.imshow(
            colors.swapaxes(0, 1)[leaves],
            aspect='auto',
            interpolation=None,
            origin='lower',
        )

        self.ax.axis('off')

        self.component.collection['image'] = image

        return self

    def initialize(self) -> Self:  # noqa
        default: Dict[Any, Any] = {}

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

    def matrix(self) -> Self:  # noqa
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

        self.component.collection['colors'] = colors

        return self

    def title(self) -> Self:  # noqa
        """Sets the title of the plot based on settings.

        Args:
            None.

        Returns:
            The modified Builder instance.

        """

        ax = self.component.collection.get('axx')

        title = f"Barcode for {self.settings.name}"

        ax.set_title(
            title,
            fontsize=18,
            pad=25
        )

        return self


class Barcode(Plot):
    def build(self) -> Component:
        return (
            self.builder
            .initialize()
            .matrix()
            .cut()
            .distance()
            .cluster()
            .dendrogram()
            .image()
            .title()
            .get()
        )
