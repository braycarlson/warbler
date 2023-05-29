"""
Spectrogram
-----------

"""

from __future__ import annotations

import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from constant import DATASET, PICKLE, SETTINGS
from datatype.axes import LinearAxes
from datatype.plot import (
    BandwidthSpectrogram,
    SegmentationSpectrogram,
    StandardSpectrogram,
    VocalEnvelopeSpectrogram
)
from datatype.segmentation import DynamicThresholdSegmentation
from datatype.settings import Settings
from datatype.signal import Signal
from datatype.spectrogram import (
    Linear,
    Mel,
    mel_matrix,
    Spectrogram
)


def main() -> None:
    # Signal
    path = DATASET.joinpath('DbWY_STE2017/recordings/STE05_DbWY2017.wav')
    signal = Signal(path)

    # Settings
    path = DATASET.joinpath('DbWY_STE2017/segmentation/STE05_DbWY2017.json')
    settings = Settings.from_file(path)

    # Dereverberate
    path = SETTINGS.joinpath('dereverberate.json')
    dereverberate = Settings.from_file(path)

    # Filter(s)
    if settings.bandpass_filter:
        signal.filter(
            settings.butter_lowcut,
            settings.butter_highcut
        )

    if settings.normalize:
        signal.normalize()

    if settings.dereverberate:
        signal.dereverberate(dereverberate)

    if settings.reduce_noise:
        signal.reduce()

    # Mel filterbank
    path = PICKLE.joinpath('matrix.npy')

    if path.is_file():
        matrix = np.load(path, allow_pickle=True)
    else:
        matrix = mel_matrix(settings)
        np.save(path, matrix, allow_pickle=True)

    # Default settings
    path = SETTINGS.joinpath('spectrogram.json')
    default = Settings.from_file(path)

    strategy = Linear(signal, default)

    spectrogram = Spectrogram()
    spectrogram.strategy = strategy
    spectrogram = spectrogram.generate()

    # SegmentationSpectrogram or VocalEnvelopeSpectrogram
    algorithm = DynamicThresholdSegmentation()
    algorithm.signal = signal
    algorithm.settings = settings
    algorithm.start()

    algorithm.component['spectrogram'] = spectrogram

    scale = LinearAxes

    plot = VocalEnvelopeSpectrogram()
    plot.algorithm = algorithm
    plot.scale = scale
    plot.settings = settings
    plot.signal = signal
    plot.create()

    plot = BandwidthSpectrogram()
    plot.scale = scale
    plot.settings = settings
    plot.signal = signal
    plot.spectrogram = spectrogram
    plot.create()

    plot = StandardSpectrogram()
    plot.scale = scale
    plot.settings = settings
    plot.signal = signal
    plot.spectrogram = spectrogram
    plot.create()

    plot = SegmentationSpectrogram()
    plot.algorithm = algorithm
    plot.scale = scale
    plot.settings = settings
    plot.signal = signal
    plot.create()

    plt.show()
    plt.close()

    # Mel
    sr = settings.sample_rate / 2

    strategy = Mel(signal, default)

    spectrogram = Spectrogram()
    spectrogram.strategy = strategy
    spectrogram = spectrogram.generate()

    plt.figure(
        figsize=(20, 4)
    )

    librosa.display.specshow(
        spectrogram,
        x_axis='time',
        y_axis='mel',
        sr=sr,
    )

    ax = plt.gca()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')

    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')
    plt.tight_layout()

    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
