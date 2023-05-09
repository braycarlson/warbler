"""
Segmentation
------------

"""

import matplotlib.pyplot as plt

from constant import DATASET, SETTINGS
from datatype.plot import VocalEnvelopeSpectrogram
from datatype.segmentation import DynamicThresholdSegmentation
from datatype.settings import Settings
from datatype.signal import Signal


def main():
    path = SETTINGS.joinpath('spectrogram.json')
    settings = Settings.from_file(path)

    path = SETTINGS.joinpath('dereverberate.json')
    dereverberate = Settings.from_file(path)

    path = DATASET.joinpath('DbWY_STE2017/recordings/STE05_DbWY2017.wav')
    signal = Signal(path)

    if settings.bandpass_filter:
        signal.filter(
            settings.butter_lowcut,
            settings.butter_highcut
        )

    if settings.normalize:
        signal.normalize()

    if settings.reduce_noise:
        signal.reduce()

    if settings.dereverberate:
        signal.dereverberate(dereverberate)

    algorithm = DynamicThresholdSegmentation()
    algorithm.signal = signal
    algorithm.settings = settings
    algorithm.start()

    plot = VocalEnvelopeSpectrogram()
    plot.algorithm = algorithm
    plot.signal = signal
    plot.create()

    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
