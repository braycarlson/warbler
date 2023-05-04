import matplotlib.pyplot as plt

from constant import SETTINGS
from datatype.axes import SpectrogramAxes
from datatype.dataset import Dataset
from datatype.imaging import (
    create_grid,
    create_image,
    filter_image,
    to_bytes
)
from datatype.settings import Settings
from datatype.spectrogram import create_spectrogram
from plot import LusciniaSpectrogram


def main():
    dataset = Dataset('signal')
    dataframe = dataset.load()

    signal = dataframe.at[10, 'signal']

    path = SETTINGS.joinpath('dereverberate.json')
    dereverberate = Settings.from_file(path)

    path = SETTINGS.joinpath('spectrogram.json')
    settings = Settings.from_file(path)

    signal.normalize()
    signal.dereverberate(dereverberate)

    spectrogram = create_spectrogram(signal, settings)

    # Original
    image = create_image(spectrogram)

    plot = LusciniaSpectrogram(signal, image)
    plot.create()

    plt.show()
    plt.close()

    original = to_bytes(image)

    # Filtered
    image = create_image(spectrogram)
    filtered = filter_image(image)

    plot = LusciniaSpectrogram(signal, filtered)
    plot.create()

    plt.show()
    plt.close()

    filtered = to_bytes(filtered)

    collection = [original, filtered]

    grid = create_grid(collection)

    grid.show()


if __name__ == '__main__':
    main()
