from __future__ import annotations

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import PIL

from abc import ABC, abstractmethod
from matplotlib import gridspec
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
# from matplotlib.widgets import Slider
# from PIL import ImageOps
from typing_extensions import TYPE_CHECKING
from warbler.datatype.axes import LinearAxes, SpectrogramAxes

if TYPE_CHECKING:
    import numpy as np
    import numpy.typing as npt

    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage
    from warbler.datatype.segmentation import DynamicThresholdSegmentation
    from warbler.datatype.settings import Settings
    from warbler.datatype.signal import Signal


class Plot(ABC):
    def __init__(
        self,
        scale: LinearAxes | SpectrogramAxes | None = LinearAxes,
        settings: Settings | None = None,
        signal: Signal | None = None,
        spectrogram : np.array = None
    ):
        self.scale = scale
        self.settings = settings
        self.signal = signal
        self.spectrogram = spectrogram

    @abstractmethod
    def create(self) -> Figure | AxesImage:
        raise NotImplementedError

    def plot(self, **kwargs) -> AxesImage:
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

        ax.initialize()
        ax._x_lim(x_maximum)
        ax._x_step(x_maximum)
        ax._y_lim(y_maximum)

        return image


class StandardSpectrogram(Plot):
    def __init__(
        self,
        scale: LinearAxes | SpectrogramAxes | None = LinearAxes,
        settings: Settings | None = None,
        signal: Signal | None = None,
        spectrogram : np.array = None
    ):
        super().__init__(
            scale=scale,
            settings=settings,
            signal=signal,
            spectrogram=spectrogram
        )

    def create(self) -> Figure:
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
    def __init__(
        self,
        scale: LinearAxes | SpectrogramAxes | None = LinearAxes,
        settings: Settings | None = None,
        signal: Signal | None = None,
        spectrogram : np.array = None
    ):
        super().__init__(
            scale=scale,
            settings=settings,
            signal=signal,
            spectrogram=spectrogram
        )

    def create(self) -> Figure:
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
    def __init__(
        self,
        algorithm: DynamicThresholdSegmentation = None,
        scale: LinearAxes | SpectrogramAxes | None = LinearAxes,
        settings: Settings | None = None,
        signal: Signal | None = None,
        spectrogram : np.array = None
    ):
        super().__init__(
            scale=scale,
            settings=settings,
            signal=signal,
            spectrogram=spectrogram
        )

        self.algorithm = algorithm

    def create(self) -> AxesImage:
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

        for index, (onset, offset) in enumerate(zip(onsets, offsets), 0):
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
    def __init__(
        self,
        algorithm: DynamicThresholdSegmentation = None,
        scale: LinearAxes | SpectrogramAxes | None = LinearAxes,
        settings: Settings | None = None,
        signal: Signal | None = None,
        spectrogram : np.array = None
    ):
        super().__init__(
            scale=scale,
            settings=settings,
            signal=signal,
            spectrogram=spectrogram
        )

        self.algorithm = algorithm

    def create(self) -> AxesImage:
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

        for index, (onset, offset) in enumerate(zip(onsets, offsets), 0):
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
