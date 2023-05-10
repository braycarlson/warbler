"""
Plot
----

"""

from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import PIL

from abc import ABC, abstractmethod
from datatype.axes import LinearAxes, SpectrogramAxes
from matplotlib import gridspec
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
# from matplotlib.widgets import Slider
# from PIL import ImageOps
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from datatype.segmentation import DynamicThresholdSegmentation
    from datatype.settings import Settings
    from datatype.signal import Signal
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage


class Plot(ABC):
    """An abstract base class for creating plots.

    Args:
        scale: The type of scale to be used.
        settings: The settings for the plot.
        signal: The signal object to be plotted.
        spectrogram: The spectrogram data to be plotted.

    Attributes:
        _scale: The type of scale to be used.
        _settings: The settings for the plot.
        _signal: The signal object to be plotted.
        _spectrogram: The spectrogram data to be plotted.

    """

    def __init__(
        self,
        settings: Settings | None = None,
        signal: Signal | None = None,
        spectrogram : np.array = None
    ):
        self._scale = LinearAxes
        self._settings = settings
        self._signal = signal
        self._spectrogram = spectrogram

    @property
    def settings(self) -> Settings:
        """Get the settings for the plot."""

        return self._settings

    @settings.setter
    def settings(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def signal(self) -> Signal:
        """Get the signal object to be plotted."""

        return self._signal

    @signal.setter
    def signal(self, signal: Signal) -> None:
        self._signal = signal

    @property
    def spectrogram(self) -> npt.NDArray:
        """Get the spectrogram data to be plotted."""

        return self._spectrogram

    @spectrogram.setter
    def spectrogram(self, spectrogram: npt.NDArray) -> None:
        self._spectrogram = spectrogram

    @property
    def scale(self) -> SpectrogramAxes:
        """Get the type of scale to be used."""

        return self._scale

    @scale.setter
    def scale(self, scale: SpectrogramAxes) -> None:
        self._scale = scale

    @abstractmethod
    def create(self) -> Figure | AxesImage:
        """Create the plot."""

        raise NotImplementedError

    def plot(self, **kwargs) -> AxesImage:
        """Plot the data.

        Args:
            **kwargs: Additional keyword arguments.

        Returns:
            The plotted image.

        """

        x_minimum = 0
        x_maximum = self.signal.duration
        y_minimum = 0
        y_maximum = self.signal.rate / 2

        ax = kwargs.pop('ax')

        if 'spectrogram' in kwargs:
            self.spectrogram = kwargs.pop('spectrogram')

        extent = [
            x_minimum,
            x_maximum,
            y_minimum,
            y_maximum
        ]

        origin = (
            'upper'
            if isinstance(self.spectrogram, PIL.Image.Image)
            else 'lower'
        )

        image = ax.matshow(
            self.spectrogram,
            aspect='auto',
            extent=extent,
            interpolation=None,
            origin=origin,
            **kwargs
        )

        # if self.signal.duration > 10:
        #     window = 10

        #     axes = plt.axes(
        #         [0.0, 0.0, 0.65, 0.03],
        #     )

        #     slider = Slider(
        #         axes,
        #         'Position',
        #         0.0,
        #         self.signal.duration - window
        #     )

        #     def update(val):
        #         ax.axis(
        #             [
        #                 slider.val,
        #                 slider.val + window,
        #                 y_minimum,
        #                 y_maximum
        #             ]
        #         )

        #         fig = plt.gcf()
        #         fig.canvas.draw()

        #     slider.on_changed(update)

        ax.initialize()
        ax._x_lim(x_maximum)
        ax._x_step(x_maximum)
        ax._y_lim(y_maximum)

        return image


class StandardSpectrogram(Plot):
    """A class for creating standard spectrogram plots.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create(self) -> Figure:
        """Create the standard spectrogram plot.

        Returns:
            The created figure.

        """

        fig = plt.figure(
            constrained_layout=True,
            figsize=(20, 4)
        )

        projection = self.scale(
            fig,
            [0, 0, 0, 0]
        )

        ax = fig.add_subplot(projection=projection)

        # cmap = plt.cm.afmhot
        cmap = plt.cm.Greys
        self.plot(ax=ax, cmap=cmap)

        return fig


class BandwidthSpectrogram(Plot):
    """A class for creating bandwidth spectrogram plots.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def create(self) -> Figure:
        """Create the bandwidth spectrogram plot.

        Returns:
            The created figure.

        """

        fig = plt.figure(
            constrained_layout=True,
            figsize=(20, 4)
        )

        projection = self.scale(
            fig,
            [0, 0, 0, 0]
        )

        ax = fig.add_subplot(projection=projection)

        cmap = plt.cm.Greys
        self.plot(ax=ax, cmap=cmap)

        fig.patch.set_facecolor('#ffffff')
        ax.patch.set_facecolor('#ffffff')

        color = mcolors.to_rgba('#d1193e', alpha=0.75)

        ax.axhline(
            self.settings.butter_lowcut,
            color=color,
            ls='dashed',
            lw=1,
            alpha=1
        )

        ax.axhline(
            self.settings.butter_highcut,
            color=color,
            ls='dashed',
            lw=1,
            alpha=1
        )

        return fig


class SegmentationSpectrogram(Plot):
    """A class for creating segmentation spectrogram plots.

    Args:
        *args: Variable length argument list.
        algorithm: The algorithm containing onset(s) and offset(s).
        **kwargs: Arbitrary keyword arguments.

    """

    def __init__(
        self,
        *args,
        algorithm: DynamicThresholdSegmentation = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.algorithm = algorithm

    def create(self) -> AxesImage:
        """Create a segmentation spectrogram plot.

        Returns:
            The created AxesImage object.

        """

        spectrogram = self.algorithm.component.get('spectrogram')
        onsets = self.algorithm.component.get('onset')
        offsets = self.algorithm.component.get('offset')

        fig = plt.figure(
            constrained_layout=True,
            figsize=(20, 4)
        )

        projection = self.scale(
            fig,
            [0, 0, 0, 0]
        )

        ax = fig.add_subplot(projection=projection)

        cmap = plt.cm.Greys

        image = self.plot(
            ax=ax,
            cmap=cmap,
            spectrogram=spectrogram
        )

        _, ylmax = ax.get_ylim()

        blue = mcolors.to_rgba('#0079d3', alpha=0.75)
        red = mcolors.to_rgba('#d1193e', alpha=0.75)

        for index, (onset, offset) in enumerate(zip(onsets, offsets, strict=True), 0):
            color = blue

            if self.settings is not None:
                color = red if index in self.settings.exclude else blue

            ax.axvline(
                onset,
                color=color,
                ls='dashed',
                lw=1,
                alpha=0.75
            )

            ax.axvline(
                offset,
                color=color,
                ls='dashed',
                lw=1,
                alpha=0.75
            )

            height = 2000

            rectangle = Rectangle(
                xy=(onset, ylmax - height),
                width=offset - onset,
                height=height,
                alpha=0.75,
                color=color,
                label=str(index)
            )

            rx, ry = rectangle.get_xy()
            cx = rx + rectangle.get_width() / 2.0
            cy = ry + rectangle.get_height() / 2.0

            ax.annotate(
                index,
                (cx, cy),
                color='white',
                weight=600,
                fontfamily='Arial',
                fontsize=8,
                ha='center',
                va='center'
            )

            ax.add_patch(rectangle)

        return image


class VocalEnvelopeSpectrogram(Plot):
    """A class for creating vocal envelope spectrogram plots.

    Args:
        *args: Variable length argument list.
        algorithm: The algorithm containing onset(s) and offset(s).
        **kwargs: Arbitrary keyword arguments.

    """

    def __init__(
        self,
        *args,
        algorithm: DynamicThresholdSegmentation = None,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.algorithm = algorithm

    def create(self) -> AxesImage:
        """Create a vocal envelope spectrogram plot.

        Returns:
            The created AxesImage object.

        """

        spectrogram = self.algorithm.component.get('spectrogram')
        onsets = self.algorithm.component.get('onset')
        offsets = self.algorithm.component.get('offset')
        vocal_envelope = self.algorithm.component.get('vocal_envelope')

        fig = plt.figure(
            figsize=(20, 4)
        )

        gs = gridspec.GridSpec(
            2,
            1,
            height_ratios=[1, 3]
        )

        projection = self.scale(
            fig,
            [0, 0, 0, 0]
        )

        gs.update(hspace=0.0)
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1], projection=projection)

        image = self.plot(
            ax=ax1,
            cmap=plt.cm.Greys,
            spectrogram=spectrogram
        )

        ax0.plot(vocal_envelope, color='k')

        ax0.set_xlim(
            [0, len(vocal_envelope)]
        )

        _, ylmax = ax1.get_ylim()

        patches = []

        for index, (onset, offset) in enumerate(zip(onsets, offsets, strict=True), 0):
            ax1.axvline(
                onset,
                color='dodgerblue',
                ls='dashed',
                lw=1,
                alpha=0.75
            )

            ax1.axvline(
                offset,
                color='dodgerblue',
                ls='dashed',
                lw=1,
                alpha=0.75
            )

            height = 2000

            rectangle = Rectangle(
                xy=(onset, ylmax - height),
                width=offset - onset,
                height=height,
            )

            rx, ry = rectangle.get_xy()
            cx = rx + rectangle.get_width() / 2.0
            cy = ry + rectangle.get_height() / 2.0

            ax1.annotate(
                index,
                (cx, cy),
                color='white',
                weight=600,
                fontfamily='Arial',
                fontsize=8,
                ha='center',
                va='center'
            )

            patches.append(rectangle)

        collection = PatchCollection(
            patches,
            color='dodgerblue',
            alpha=0.75
        )

        ax1.add_collection(collection)
        ax0.axis('off')

        fig.tight_layout()

        return image
