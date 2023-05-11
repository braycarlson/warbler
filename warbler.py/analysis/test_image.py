"""
Spectrogram
-----------

"""

import matplotlib.pyplot as plt

from constant import DATASET, SETTINGS
from datatype.axes import LinearAxes
from datatype.dataset import Dataset
from datatype.imaging import (
    create_grid,
    create_image,
    filter_image,
    to_bytes,
)
from datatype.plot import StandardSpectrogram
from datatype.settings import Settings
from datatype.signal import Signal
from datatype.spectrogram import (
    create_spectrogram,
    Linear,
    mel_matrix,
    pad,
    resize,
    Spectrogram
)
from PIL import ImageOps


def main() -> None:
    filename = 'STE06_1_DbWY2017'

    # Signal
    path = DATASET.joinpath('DbWY_STE2017/recordings/STE06_1_DbWY2017.wav')
    signal = Signal(path)

    # Settings
    path = DATASET.joinpath('DbWY_STE2017/segmentation/STE06_1_DbWY2017.json')
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

    # Default settings
    path = SETTINGS.joinpath('spectrogram.json')
    default = Settings.from_file(path)

    strategy = Linear(signal, default)

    spectrogram = Spectrogram()
    spectrogram.strategy = strategy
    spectrogram = spectrogram.generate()

    scale = LinearAxes

    plot = StandardSpectrogram()
    plot.scale = scale
    plot.settings = settings
    plot.signal = signal
    plot.spectrogram = spectrogram
    plot.create()

    plt.show()
    plt.close()

    # Segment
    dataset = Dataset('segment')
    dataframe = dataset.load()

    dataframe = (
        dataframe[dataframe['filename'] == filename]
        .reset_index(drop=True)
    )

    path = SETTINGS.joinpath('spectrogram.json')
    settings = Settings.from_file(path)

    matrix = mel_matrix(settings)

    dataframe['spectrogram'] = (
        dataframe['segment']
        .apply(
            lambda x: create_spectrogram(x, settings, matrix)
        )
    )

    dataframe['scale'] = (
        dataframe['spectrogram']
        .apply(
            lambda x: resize(x)
        )
    )

    padding = dataframe['scale'].apply(
        lambda x: len(x.T)
    ).max()

    dataframe['scale'] = (
        dataframe['scale']
        .apply(
            lambda x: pad(x, padding)
        )
    )

    dataframe['original'] = (
        dataframe['scale']
        .apply(
            lambda x: create_image(x)
        )
    )

    dataframe['original'] = (
        dataframe['original']
        .apply(
            lambda x: ImageOps.invert(x)
        )
    )

    dataframe['original'] = (
        dataframe['original']
        .apply(
            lambda x: ImageOps.flip(x)
        )
    )

    dataframe['filter_array'] = (
        dataframe['original']
        .apply(
            lambda x: filter_image(x)
        )
    )

    dataframe['filter_bytes'] = (
        dataframe['filter_array']
        .apply(
            lambda x: to_bytes(x)
        )
    )

    # images = dataframe.original.tolist()
    # [image.show() for image in images]

    images = dataframe.filter_bytes.tolist()

    grid = create_grid(images)
    grid.show()


if __name__ == '__main__':
    main()
