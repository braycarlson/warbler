import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from dataclass.signal import Signal
from dataclass.spectrogram import Spectrogram
from matplotlib import gridspec
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle


def plot_spectrogram(
    spectrogram,
    fig=None,
    ax=None,
    signal=None,
    hop_len_ms=None,
    cmap=plt.cm.afmhot,
    show_cbar=False,
    spectral_range=None,
    time_range=None,
    figsize=(20, 6),
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    extent = [
        0,
        np.shape(spectrogram)[1],
        0,
        np.shape(spectrogram)[0]
    ]

    if signal is not None:
        extent[3] = 10000

    if hop_len_ms is not None:
        hop_len_ms_int_adj = int(hop_len_ms / 1000 * signal.rate) / (signal.rate / 1000)
        extent[1] = (np.shape(spectrogram)[1] * hop_len_ms_int_adj) / 1000

    if spectral_range is not None:
        extent[2] = spectral_range[0]
        extent[3] = spectral_range[1]

    if time_range is not None:
        extent[0] = time_range[0]
        extent[1] = time_range[1]

    spec_ax = ax.matshow(
        spectrogram,
        interpolation=None,
        aspect='auto',
        cmap=cmap,
        origin='lower',
        extent=extent,
    )

    if show_cbar:
        cbar = fig.colorbar(spec_ax, ax=ax)
        return spec_ax, cbar
    else:
        return spec_ax


def plot_segmentations(
    spectrogram, vocal_envelope, onsets, offsets, hop_length_ms, signal, figsize=(30, 5)
):
    fig = plt.figure(figsize=figsize)

    gs = gridspec.GridSpec(
        2,
        1,
        height_ratios=[1, 3]
    )

    # Set the spacing between axes.
    gs.update(hspace=0.0)
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    plot_spectrogram(
        spectrogram,
        fig,
        ax1,
        signal=signal,
        hop_len_ms=hop_length_ms,
        show_cbar=False
    )

    ax0.plot(vocal_envelope, color='k')

    ax0.set_xlim(
        [0, len(vocal_envelope)]
    )

    ylmin, ylmax = ax1.get_ylim()
    ysize = (ylmax - ylmin) * 0.1
    ymin = ylmax - ysize

    patches = []

    for onset, offset in zip(onsets, offsets):
        ax1.axvline(
            onset,
            color='#FFFFFF',
            ls='dashed',
            lw=0.75
        )

        ax1.axvline(
            offset,
            color='#FFFFFF',
            ls='dashed',
            lw=0.75
        )

        patches.append(
            Rectangle(
                xy=(onset, ymin),
                width=offset - onset,
                height=ysize
            )
        )

    collection = PatchCollection(
        patches,
        color='white',
        alpha=0.5
    )

    ax1.set_xlim([0, signal.duration])
    ax1.set_ylim([0, 10000])

    # Time
    ticks_x = ticker.FuncFormatter(
        lambda x, pos: '{0:,.1f}'.format(x)
    )

    # Frequency
    ticks_y = ticker.FuncFormatter(
        lambda y, pos: '{0:g}'.format(y / 1e3)
    )

    ax1.xaxis.set_major_formatter(ticks_x)
    ax1.yaxis.set_major_formatter(ticks_y)

    ax1.set_xlabel(
        'Time (s)',
        fontfamily='Arial',
        fontsize=16,
        fontweight=600
    )

    ax1.set_ylabel(
        'Frequency (kHz)',
        fontfamily='Arial',
        fontsize=16,
        fontweight=600
    )

    ax1.xaxis.tick_bottom()

    ax1.xaxis.set_minor_locator(
        ticker.MaxNLocator(30)
    )

    ax1.yaxis.set_ticks(
        [0, 5000, 10000]
    )

    ax1.xaxis.set_ticks(
        np.arange(0, signal.duration, 0.5)
    )

    ax1.yaxis.set_minor_locator(
        ticker.MultipleLocator(1000)
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

    ax1.add_collection(collection)
    ax0.axis('off')
    return fig


def create_luscinia_spectrogram(path, parameters):
    signal = Signal(path)
    signal.filter(parameters.butter_lowcut, parameters.butter_highcut)

    spectrogram = Spectrogram(signal, parameters)

    fig, ax = plt.subplots(
        constrained_layout=True,
        figsize=(20, 4)
    )

    plot_spectrogram(
        spectrogram.data,
        fig,
        ax,
        cmap=plt.cm.Greys,
        show_cbar=False
    )

    plt.ylim([0, 1000])
    plt.xlim([0, signal.duration * 1000])

    # Time
    ticks_x = ticker.FuncFormatter(
        lambda x, pos: '{0:,.1f}'.format(x / 1e3)
    )

    # Frequency
    ticks_y = ticker.FuncFormatter(
        lambda y, pos: '{0:g}'.format(y / 1e2)
    )

    ax.xaxis.set_major_formatter(ticks_x)
    ax.yaxis.set_major_formatter(ticks_y)

    ax.set_xlabel(
        'Time (s)',
        fontfamily='Arial',
        fontsize=16,
        fontweight=600
    )

    ax.set_ylabel(
        'Frequency (kHz)',
        fontfamily='Arial',
        fontsize=16,
        fontweight=600
    )

    ax.xaxis.tick_bottom()

    ax.xaxis.set_minor_locator(
        ticker.AutoMinorLocator()
    )

    ax.yaxis.set_ticks(
        [0, 500, 1000]
    )

    ax.yaxis.set_minor_locator(
        ticker.MultipleLocator(100)
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

    return plt


def create_spectrogram(path, parameters):
    signal = Signal(path)
    signal.filter(parameters.butter_lowcut, parameters.butter_highcut)

    spectrogram = Spectrogram(signal, parameters)

    fig, ax = plt.subplots(
        figsize=(20, 3)
    )

    plot_spectrogram(
        spectrogram.data,
        fig,
        ax,
        cmap=plt.cm.afmhot,
        show_cbar=False
    )

    plt.ylim([0, 1000])

    # Time
    ticks_x = ticker.FuncFormatter(
        lambda x, pos: '{0:,.1f}'.format(x / 1e3)
    )

    # Frequency
    ticks_y = ticker.FuncFormatter(
        lambda y, pos: '{0:g}'.format(y / 1e2)
    )

    ax.xaxis.set_major_formatter(ticks_x)
    ax.yaxis.set_major_formatter(ticks_y)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (kHz)',)

    ax.xaxis.tick_bottom()

    ax.xaxis.set_minor_locator(
        ticker.AutoMinorLocator()
    )

    ax.yaxis.set_ticks(
        [0, 500, 1000]
    )

    ax.yaxis.set_minor_locator(
        ticker.MultipleLocator(100)
    )

    plt.tight_layout()

    return plt
