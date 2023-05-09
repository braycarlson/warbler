"""
Builder
-------

"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings

from abc import abstractmethod
from constant import PROJECTION
from datatype.settings import Settings
from matplotlib.backends._backend_tk import FigureManagerTk
from matplotlib.figure import Figure
from typing import Any, Dict


warnings.simplefilter('ignore', UserWarning)


class Component():
    """A class representing a component."""

    def __init__(self):
        self.collection = {}


class Base:
    """Base class for other builder classes."""

    def __init__(self, settings=None):
        self._component = Component()
        self._embedding = None
        self._label = None
        self._sequence = None
        self._settings = settings
        self._spectrogram = None
        self._unique = None

    @property
    def component(self) -> Component:
        """Get the components."""

        return self._component

    @property
    def embedding(self) -> np.ndarray:
        """Get the embeddings."""

        return self._embedding

    @embedding.setter
    def embedding(self, embedding: np.ndarray) -> None:
        self._embedding = embedding

    @property
    def label(self) -> np.ndarray:
        """Get the cluster labels."""

        return self._label

    @label.setter
    def label(self, label: np.ndarray) -> None:
        self._label = label

    @property
    def sequence(self) -> np.ndarray:
        """Get the sequences."""

        return self._sequence

    @sequence.setter
    def sequence(self, sequence: np.ndarray) -> None:
        self._sequence = sequence

    @property
    def settings(self) -> Settings:
        """Get the settings."""

        return self._settings

    @settings.setter
    def settings(self, settings: Dict[Any, Any]) -> None:
        self._settings = settings

    @property
    def spectrogram(self) -> np.ndarray:
        """Get the spectrograms."""

        return self._spectrogram

    @spectrogram.setter
    def spectrogram(self, spectrogram: np.ndarray) -> None:
        self._spectrogram = spectrogram

    @property
    def unique(self) -> np.ndarray:
        """Get the unique cluster labels."""

        return self._unique

    @unique.setter
    def unique(self, unique: np.ndarray) -> None:
        self._unique = unique

    def get(self) -> Component:
        """Get the component.

        Returns:
            The components.

        """

        return self.component

    def palette(self) -> Self:  # noqa
        """Create a color palette.

        Returns:
            The modified Base instance.

        """

        if self.unique is None:
            self.unique = np.unique(self.label)

        self.unique.sort()
        n_colors = len(self.unique)

        palette = sns.color_palette(
            self.settings.palette,
            n_colors=n_colors
        )

        master = dict(
            zip(self.unique, palette)
        )

        label = {
            label: master[label]
            for label in self.label
        }

        label = dict(
            sorted(
                label.items()
            )
        )

        if -1 in label:
            label[-1] = (0.85, 0.85, 0.85)

        color = np.array(
            [label[i] for i in self.label],
            dtype='float'
        )

        self.component.collection['color'] = color
        self.component.collection['label'] = label
        self.component.collection['palette'] = palette

        return self


class Plot:
    """A class for plotting."""

    def __init__(self):
        self._builder = None
        self._embedding = None
        self._label = None
        self._sequence = None
        self._spectrogram = None
        self._settings = None

    @property
    def builder(self) -> Any:
        """Get the builder."""

        return self._builder

    @builder.setter
    def builder(self, builder: Any) -> None:
        self._builder = builder

    @property
    def settings(self) -> Settings:
        """Get the settings from the builder."""

        return self.builder.settings

    @settings.setter
    def settings(self, settings: Settings) -> None:
        self.builder.settings = settings

    @property
    def embedding(self):
        return self.builder.embedding

    @embedding.setter
    def embedding(self, embedding: np.ndarray) -> None:
        self.builder.embedding = embedding

    @property
    def label(self):
        return self.builder.label

    @label.setter
    def label(self, label: np.ndarray) -> None:
        self.builder.label = label

    @property
    def sequence(self):
        return self.builder.sequence

    @sequence.setter
    def sequence(self, sequence: np.ndarray) -> None:
        self.builder.sequence = sequence

    @property
    def spectrogram(self):
        return self.builder.spectrogram

    @spectrogram.setter
    def spectrogram(self, spectrogram: np.ndarray) -> None:
        self.builder.spectrogram = spectrogram

    @property
    def unique(self):
        return self.builder.unique

    @unique.setter
    def unique(self, unique: np.ndarray) -> None:
        self.builder.unique = unique

    @abstractmethod
    def build(self) -> Component:
        """Build the plot. This must be implemented in subclasses."""

        raise NotImplementedError

    def save(self, figure: Figure = None, filename: str = None) -> None:
        """Save the plot to an image.

        Args:
            filename: The name of the file to save.
            fig: The figure to save.

        """

        PROJECTION.mkdir(
            parents=True,
            exist_ok=True
        )

        path = PROJECTION.joinpath(filename)

        if figure is None:
            plt.savefig(
                path,
                bbox_inches='tight',
                dpi=300,
                format='png'
            )
        else:
            figure.savefig(
                path,
                bbox_inches='tight',
                dpi=300,
                format='png'
            )

        plt.close()

    def show(self) -> None:
        """Show the plot.

        Adjust the figure size and display the plot.

        """

        manager = plt.get_current_fig_manager()

        if isinstance(manager, FigureManagerTk):
            w, h = self.builder.settings.figure.get('figsize')
            w = w * 96
            h = h * 96

            width = manager.window.winfo_screenwidth()
            height = manager.window.winfo_screenheight()

            x = (width / 2) - (w / 2)
            x = int(x)

            y = (height / 2) - (h / 2)
            y = int(y)

            geometry = f"{w}x{h}+{x}+{y}"
            manager.window.wm_geometry(geometry)

        plt.tight_layout()
        plt.show()
        plt.close()
