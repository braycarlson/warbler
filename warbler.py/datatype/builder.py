from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings

from abc import abstractmethod
from constant import PROJECTION
from datatype.settings import Settings
from matplotlib.backends._backend_tk import FigureManagerTk
from typing import NoReturn


warnings.simplefilter('ignore', UserWarning)


class Component():
    """A class representing a component."""

    def __init__(self):
        self.collection = {}


class Base:
    """Base class for other builder classes."""

    def __init__(self, settings: Settings = None):
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

    @property
    def label(self) -> np.ndarray:
        """Get the cluster labels."""

        return self._label

    @property
    def sequence(self) -> np.ndarray:
        """Get the sequences."""

        return self._sequence

    @property
    def settings(self) -> Settings:
        """Get the settings."""

        return self._settings

    @property
    def spectrogram(self) -> np.ndarray:
        """Get the spectrograms."""

        return self._spectrogram

    @property
    def unique(self) -> np.ndarray:
        """Get the unique cluster labels."""

        return self._unique

    @embedding.setter
    def embedding(self, embedding: np.ndarray) -> None:
        self._embedding = embedding

    @label.setter
    def label(self, label: np.ndarray) -> None:
        self._label = label

    @sequence.setter
    def sequence(self, sequence: np.ndarray) -> None:
        self._sequence = sequence

    @settings.setter
    def settings(self, settings: Settings) -> None:
        self._settings = settings

    @spectrogram.setter
    def spectrogram(self, spectrogram: np.ndarray) -> None:
        self._spectrogram = spectrogram

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
    def builder(self) -> Base:
        """Get the builder."""

        return self._builder

    @property
    def settings(self) -> Settings:
        """Get the settings from the builder."""

        return self.builder.settings

    @builder.setter
    def builder(self, builder: Base) -> None:
        """Set the builder.

        Args:
            builder: An instance of the Base class.

        """

        self._builder = builder

    @Base.embedding.setter
    def embedding(self, embedding: np.ndarray) -> None:
        """Set the embeddings in the builder.

        Args:
            embedding: The embeddings array.

        """

        self.builder.embedding = embedding

    @Base.label.setter
    def label(self, label: np.ndarray) -> None:
        """Set the cluster label in the builder.

        Args:
            label: The cluster labels array.

        """

        self.builder.label = label

    @Base.sequence.setter
    def sequence(self, sequence: np.ndarray) -> None:
        """Set the sequences in the builder.

        Args:
            sequence: The sequences array.

        """

        self.builder.sequence = sequence

    @settings.setter
    def settings(self, settings: Settings) -> None:
        """Set the settings in the builder.

        Args:
            settings: An instance of the Settings class.

        """

        self.builder.settings = settings

    @Base.spectrogram.setter
    def spectrogram(self, spectrogram: np.ndarray) -> None:
        """Set the spectrograms in the builder.

        Args:
            spectrogram: The spectrograms array.

        """

        self.builder.spectrogram = spectrogram

    @Base.unique.setter
    def unique(self, unique: np.ndarray) -> None:
        """Set the unique cluster labels in the builder.

        Args:
            unique: The unique cluster labels array.

        """

        self.builder.unique = unique

    @abstractmethod
    def construct(self) -> NoReturn:
        """Construct the plot. This must be implemented in subclasses."""

        raise NotImplementedError

    def save(self, filename: str) -> None:
        """Save the plot to an image.

        Args:
            filename: The name of the file to save.

        """

        PROJECTION.mkdir(
            parents=True,
            exist_ok=True
        )

        path = PROJECTION.joinpath(filename)

        plt.savefig(
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
