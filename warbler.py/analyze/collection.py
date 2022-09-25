import matplotlib.pyplot as plt

from constant import CWD
from datatype.axes import SpectrogramAxes
from datatype.file import File
from datatype.settings import Settings
from datatype.spectrogram import Spectrogram
from plot import LusciniaSpectrogram


def main():
    file = File('good.xz')
    collection = file.load()

    for recording in collection[:5]:
        signal = recording.get('signal')

        path = CWD.joinpath(
            recording.get('segmentation')
        )

        settings = Settings.from_file(path)

        if settings.bandpass_filter:
            signal.filter(
                settings.butter_lowcut,
                settings.butter_highcut
            )

        if settings.reduce_noise:
            signal.reduce()

        spectrogram = Spectrogram(signal, settings)
        spectrogram = spectrogram.generate()

        plot = LusciniaSpectrogram(signal, spectrogram)
        plot.create()

        plt.show()
        plt.close()


if __name__ == '__main__':
    main()
