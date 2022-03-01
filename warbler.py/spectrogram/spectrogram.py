import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from scipy.io import wavfile
from vocalseg.utils import (
    butter_bandpass_filter,
    spectrogram,
    int16tofloat32,
    plot_spec
)


def create_spectrogram(path, parameters):
    rate, data = wavfile.read(path)

    data = butter_bandpass_filter(
        int16tofloat32(data),
        parameters.butter_lowcut,
        parameters.butter_highcut,
        rate
    )

    spec = spectrogram(
        data,
        rate,
        n_fft=parameters.n_fft,
        hop_length_ms=parameters.hop_length_ms,
        win_length_ms=parameters.win_length_ms,
        ref_level_db=parameters.ref_level_db,
        pre=parameters.preemphasis,
        min_level_db=parameters.min_level_db,
    )

    np.shape(spec)

    figsize = (20, 3)
    fig, ax = plt.subplots(figsize=figsize)
    plot_spec(spec, fig, ax)

    plt.ylim([0, 1000])

    ticks_y = ticker.FuncFormatter(
        lambda x, pos: '{0:g}'.format(x / 1e2)
    )

    ax.yaxis.set_major_formatter(ticks_y)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel('Frequency (kHz)')

    return plt
