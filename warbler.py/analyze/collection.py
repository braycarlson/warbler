import matplotlib.pyplot as plt

from constant import SETTINGS
from datatype.axes import SpectrogramAxes
from datatype.dataset import Dataset
from datatype.settings import Settings
from datatype.spectrogram import Linear, Spectrogram
from plot import LusciniaSpectrogram


def plot(signal, settings):
    spectrogram = Spectrogram()
    strategy = Linear(signal, settings)
    spectrogram.strategy = strategy

    spectrogram = spectrogram.generate()

    plot = LusciniaSpectrogram(signal, spectrogram)
    plot.create()

    plt.show()
    plt.close()


def main():
    dataset = Dataset('signal')
    dataframe = dataset.load()

    subset = dataframe.iloc[0:5]

    path = SETTINGS.joinpath('spectrogram.json')
    settings = Settings.from_file(path)

    # Bandpass filter
    column = ['signal', 'settings']

    subset[column].apply(
        lambda x: x.signal.filter(
            x.settings.butter_lowcut,
            x.settings.butter_highcut
        ),
        axis=1
    )

    # Reduce noise
    subset['signal'].apply(
        lambda x: x.reduce()
    )

    subset['signal'].apply(
        lambda x: plot(x, settings)
    )


if __name__ == '__main__':
    main()
