import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

from abc import ABC, abstractmethod
from matplotlib import gridspec
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


class Plot(ABC):
    def __init__(self, signal):
        self.signal = signal

    @abstractmethod
    def create(self):
        pass

    def plot(self, **kwargs):
        ax = kwargs.pop('ax')
        spectrogram = kwargs.pop('spectrogram')

        x_minimum = 0
        x_maximum = self.signal.duration
        y_minimum = 0
        y_maximum = self.signal.rate / 2

        extent = [
            x_minimum,
            x_maximum,
            y_minimum,
            y_maximum
        ]

        image = ax.matshow(
            spectrogram,
            aspect='auto',
            extent=extent,
            interpolation=None,
            origin='lower',
            **kwargs
        )

        ax.initialize()
        ax._x_lim(x_maximum)
        ax._x_step(x_maximum)

        return image


class GenericSpectrogram(Plot):
    def __init__(self, signal, spectrogram):
        self.signal = signal
        self.spectrogram = spectrogram

    def create(self):
        fig, ax = plt.subplots(
            constrained_layout=True,
            figsize=(20, 4),
            subplot_kw={'projection': 'spectrogram'}
        )

        cmap = plt.cm.afmhot

        self.plot(
            ax=ax,
            cmap=cmap,
            spectrogram=self.spectrogram
        )

        return fig


class LusciniaSpectrogram(Plot):
    def __init__(self, signal, spectrogram):
        self.signal = signal
        self.spectrogram = spectrogram

    def create(self):
        fig, ax = plt.subplots(
            constrained_layout=True,
            figsize=(20, 4),
            subplot_kw={'projection': 'luscinia'}
        )

        plt.xticks(
            fontfamily='Arial',
            fontsize=14,
            fontweight=600
        )

        plt.yticks(
            fontfamily='Arial',
            fontsize=14,
            fontweight=600
        )

        cmap = plt.cm.Greys

        self.plot(
            ax=ax,
            cmap=cmap,
            spectrogram=self.spectrogram
        )

        return fig


class CutoffSpectrogram(Plot):
    def __init__(self, signal, spectrogram, parameters):
        self.parameters = parameters
        self.signal = signal
        self.spectrogram = spectrogram

    def create(self):
        fig, ax = plt.subplots(
            figsize=(20, 4),
            subplot_kw={'projection': 'luscinia'}
        )

        fig.patch.set_facecolor('#ffffff')
        ax.patch.set_facecolor('#ffffff')

        cmap = plt.cm.Greys

        self.plot(
            ax=ax,
            cmap=cmap,
            spectrogram=self.spectrogram.data
        )

        color = mcolors.to_rgba('#d1193e', alpha=0.75)

        ax.axhline(
            self.parameters.butter_lowcut,
            color=color,
            ls='dashed',
            lw=1,
            alpha=1
        )

        ax.axhline(
            self.parameters.butter_highcut,
            color=color,
            ls='dashed',
            lw=1,
            alpha=1
        )

        return fig


class SegmentationSpectrogram(Plot):
    def __init__(self, signal, dts):
        self.dts = dts
        self.signal = signal

    def create(self):
        spectrogram = self.dts.get('spec')
        onsets = self.dts.get('onsets')
        offsets = self.dts.get('offsets')

        fig, ax = plt.subplots(
            figsize=(20, 4),
            subplot_kw={'projection': 'luscinia'}
        )

        plt.xticks(
            fontfamily='Arial',
            fontsize=14,
            fontweight=600
        )

        plt.yticks(
            fontfamily='Arial',
            fontsize=14,
            fontweight=600
        )

        cmap = plt.cm.Greys

        image = self.plot(
            ax=ax,
            cmap=cmap,
            spectrogram=spectrogram
        )

        ylmin, ylmax = ax.get_ylim()
        ysize = (ylmax - ylmin) * 0.1
        ymin = ylmax - ysize

        patches = []

        for index, (onset, offset) in enumerate(zip(onsets, offsets), 0):
            ax.axvline(
                onset,
                color='dodgerblue',
                ls='dashed',
                lw=1,
                alpha=0.75
            )

            ax.axvline(
                offset,
                color='dodgerblue',
                ls='dashed',
                lw=1,
                alpha=0.75
            )

            rectangle = Rectangle(
                xy=(onset, ymin),
                width=offset - onset,
                height=1000,
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

            patches.append(rectangle)

        collection = PatchCollection(
            patches,
            color='dodgerblue',
            alpha=0.75
        )

        ax.add_collection(collection)

        plt.tight_layout()

        return image


class VocalEnvelopeSpectrogram(Plot):
    def __init__(self, signal, dts):
        self.dts = dts
        self.signal = signal

    def create(self):
        spectrogram = self.dts.get('spec')
        vocal_envelope = self.dts.get('vocal_envelope')
        onsets = self.dts.get('onsets')
        offsets = self.dts.get('offsets')

        plt.figure(
            figsize=(20, 4)
        )

        gs = gridspec.GridSpec(
            2,
            1,
            height_ratios=[1, 3]
        )

        gs.update(hspace=0.0)
        ax0 = plt.subplot(gs[0])
        ax1 = plt.subplot(gs[1], projection='luscinia')

        image = self.plot(
            ax=ax1,
            cmap=plt.cm.Greys,
            spectrogram=spectrogram
        )

        ax0.plot(vocal_envelope, color='k')

        ax0.set_xlim(
            [0, len(vocal_envelope)]
        )

        ylmin, ylmax = ax1.get_ylim()
        ysize = (ylmax - ylmin) * 0.1
        ymin = ylmax - ysize

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

            rectangle = Rectangle(
                xy=(onset, ymin),
                width=offset - onset,
                height=1000,
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

        plt.tight_layout()

        return image
