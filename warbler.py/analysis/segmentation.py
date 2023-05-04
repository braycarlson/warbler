import matplotlib.pyplot as plt

from constant import DATASET, SETTINGS
from datatype.axes import SpectrogramAxes
from datatype.segmentation import dynamic_threshold_segmentation
from datatype.settings import Settings
from datatype.signal import Signal
from plot import VocalEnvelopeSpectrogram


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

    signal.normalize()

    if settings.reduce_noise:
        signal.reduce()

    signal.dereverberate(dereverberate)

    dts = dynamic_threshold_segmentation(
        signal,
        settings,
        full=True
    )

    plot = VocalEnvelopeSpectrogram(signal, dts)
    plot.create()

    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
