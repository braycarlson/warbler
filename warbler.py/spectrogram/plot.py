import matplotlib.pyplot as plt
import numpy as np

from dataclass.signal import Signal
from dataclass.spectrogram import Spectrogram
from matplotlib import gridspec
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


def plot_spectrogram(spectrogram, **kwargs):
    ax = kwargs.pop('ax')
    signal = kwargs.pop('signal')

    x_minimum = 0
    x_maximum = signal.duration
    y_minimum = 0
    y_maximum = signal.rate / 2

    extent = [
        x_minimum,
        x_maximum,
        y_minimum,
        y_maximum
    ]

    image = ax.matshow(
        spectrogram,
        interpolation=None,
        aspect='auto',
        origin='lower',
        extent=extent,
        **kwargs
    )

    ax.initialize()
    ax._x_lim(x_maximum)
    ax._x_step(x_maximum)

    return image


def plot_segmentation(signal, dts):
    spectrogram = dts.get('spec')
    onsets = dts.get('onsets')
    offsets = dts.get('offsets')

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

    image = plot_spectrogram(
        spectrogram,
        ax=ax,
        signal=signal,
        cmap=plt.cm.Greys,
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
    return image


def plot_segmentation_with_vocal_envelope(signal, dts):
    spectrogram = dts.get('spec')
    vocal_envelope = dts.get('vocal_envelope')
    onsets = dts.get('onsets')
    offsets = dts.get('offsets')

    plt.figure(
        figsize=(15, 5)
    )

    gs = gridspec.GridSpec(
        2,
        1,
        height_ratios=[1, 3]
    )

    gs.update(hspace=0.0)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1], projection='luscinia')

    image = plot_spectrogram(
        spectrogram,
        ax=ax1,
        signal=signal,
        cmap=plt.cm.Greys,
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
    return image


def create_luscinia_spectrogram(path, parameters):
    signal = Signal(path)

    signal.filter(
        parameters.butter_lowcut,
        parameters.butter_highcut
    )

    spectrogram = Spectrogram(signal, parameters)

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

    plot_spectrogram(
        spectrogram.data,
        ax=ax,
        signal=signal,
        cmap=plt.cm.Greys,
    )

    return fig


def create_spectrogram(path, parameters):
    signal = Signal(path)

    signal.filter(
        parameters.butter_lowcut,
        parameters.butter_highcut
    )

    spectrogram = Spectrogram(signal, parameters)

    fig, ax = plt.subplots(
        figsize=(20, 3),
        subplot_kw={'projection': 'spectrogram'}
    )

    plot_spectrogram(
        spectrogram.data,
        ax=ax,
        signal=signal,
        cmap=plt.cm.afmhot,
    )

    plt.tight_layout()

    return fig
