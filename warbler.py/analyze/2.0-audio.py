import librosa
import numpy as np
import shutil

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from parameters import Parameters
from path import DATA, SEGMENT
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import butter, lfilter


# Print without truncation
np.set_printoptions(threshold=np.inf)

# Parameters
file = SEGMENT.joinpath('parameters.json')
parameters = Parameters(file)


def int16tofloat32(data):
    return np.array(data / 32768).astype("float32")


def norm(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))


def preemphasis(x, pre):
    return lfilter([1, -pre], [1], x)


def _stft(y, fs, n_fft, hop_length_ms, win_length_ms):
    return librosa.stft(
        y=y,
        n_fft=n_fft,
        hop_length=int(hop_length_ms / 1000 * fs),
        win_length=int(win_length_ms / 1000 * fs),
    )


def _amp_to_db(x):
    return 20 * np.log10(
        np.maximum(1e-5, x)
    )


def _normalize(S, min_level_db):
    return np.clip(
        (S - min_level_db) / -min_level_db, 0, 1
    )


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    if highcut > int(fs / 2):
        highcut = int(fs / 2)

    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)

    return y


def spectrogram(
    y,
    fs,
    n_fft=1024,
    hop_length_ms=1,
    win_length_ms=5,
    ref_level_db=20,
    pre=0.97,
    min_level_db=-50,
):
    return _normalize(
        spectrogram_nn(
            y,
            fs,
            n_fft=n_fft,
            hop_length_ms=hop_length_ms,
            win_length_ms=win_length_ms,
            ref_level_db=ref_level_db,
            pre=pre,
        ),
        min_level_db=min_level_db,
    )


def spectrogram_nn(
    y,
    fs, n_fft,
    hop_length_ms,
    win_length_ms,
    ref_level_db,
    pre
):
    D = _stft(
        preemphasis(y, pre),
        fs,
        n_fft,
        hop_length_ms,
        win_length_ms
    )

    D = np.abs(D)
    S = _amp_to_db(D) - ref_level_db

    return S


def plot_spec(
    spec,
    fig=None,
    ax=None,
    rate=None,
    hop_len_ms=None,
    cmap=plt.cm.afmhot,
    show_cbar=True,
    spectral_range=None,
    time_range=None,
    figsize=(20, 6),
):
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    extent = [0, np.shape(spec)[1], 0, np.shape(spec)[0]]

    if rate is not None:
        extent[3] = rate / 2

    if hop_len_ms is not None:
        hop_len_ms_int_adj = int(hop_len_ms / 1000 * rate) / (rate / 1000)
        extent[1] = (np.shape(spec)[1] * hop_len_ms_int_adj) / 1000

    if spectral_range is not None:
        extent[2] = spectral_range[0]
        extent[3] = spectral_range[1]

    if time_range is not None:
        extent[0] = time_range[0]
        extent[1] = time_range[1]

    spec_ax = ax.matshow(
        spec,
        interpolation=None,
        aspect="auto",
        cmap=cmap,
        origin="lower",
        extent=extent,
    )

    if show_cbar:
        cbar = fig.colorbar(spec_ax, ax=ax)
        return spec_ax, cbar
    else:
        return spec_ax


def create_spectrogram(rate, data):
    spec = spectrogram(
        data,
        rate,
        n_fft=parameters.n_fft,
        hop_length_ms=parameters.hop_length_ms,
        win_length_ms=parameters.win_length_ms,
        ref_level_db=parameters.ref_level_db,
        pre=parameters.pre,
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


def main():
    parameters.n_fft = 4096
    parameters.hop_length_ms = 1
    parameters.win_length_ms = 5
    parameters.ref_level_db = 50
    parameters.pre = 0.97
    parameters.min_level_db = -50
    parameters.butter_lowcut = 1500
    parameters.butter_highcut = 10000

    # path = DATA.joinpath('LLbLg_STE2017/wav/STE01.1_LLbLg2017.wav')
    path = DATA.joinpath('RbRY_STE2017/wav/STE01_RbRY2017.wav')
    rate, data = wavfile.read(path)

    data = butter_bandpass_filter(
        int16tofloat32(data),
        parameters.butter_lowcut,
        parameters.butter_highcut,
        rate
    )

    original = 'original' + '_' + path.name
    filtered = 'filtered' + '_' + path.name

    Path(original).unlink(missing_ok=True)
    Path(filtered).unlink(missing_ok=True)

    # Copy the original .wav
    shutil.copy(path, original)

    # Save the filtered .wav
    wavfile.write(filtered, rate, data)

    create_spectrogram(rate, data)

    plt.figtext(
        0.03,
        -0.09,
        f"{path.stem}: Low Amplitude",
        color='black',
        fontsize=12,
        fontweight=600,
        fontfamily='monospace'
    )

    plt.tight_layout()

    plt.savefig(
        'test',
        bbox_inches='tight',
        pad_inches=0.5
    )

    plt.show()


if __name__ == '__main__':
    main()
