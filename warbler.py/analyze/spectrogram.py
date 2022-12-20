import matplotlib.pyplot as plt

from constant import DATASET, SETTINGS
from datatype.axes import SpectrogramAxes
from datatype.settings import Settings
from datatype.signal import Signal
from datatype.spectrogram import Linear, Mel, Spectrogram
from plot import (
    CutoffSpectrogram,
    GenericSpectrogram,
    LusciniaSpectrogram
)


def main():
    path = SETTINGS.joinpath('spectrogram.json')
    settings = Settings.from_file(path)

    # path = DATASET.joinpath('DbWY_STE2017/segmentation/STE05_DbWY2017.json')
    # settings = Settings.from_file(path)

    path = SETTINGS.joinpath('dereverberate.json')
    dereverberate = Settings.from_file(path)

    path = DATASET.joinpath('DbWY_STE2017/recordings/STE05_DbWY2017.wav')
    signal = Signal(path)

    if settings.bandpass_filter:
        signal.filter(
            settings.butter_lowcut,
            settings.butter_highcut
        )

    if settings.reduce_noise:
        signal.reduce()

    signal.dereverberate(dereverberate)

    spectrogram = Spectrogram()
    strategy = Linear(signal, settings)
    spectrogram.strategy = strategy

    spectrogram = spectrogram.generate()

    plot = LusciniaSpectrogram(signal, spectrogram)
    plot.create()

    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
