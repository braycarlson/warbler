"""
Image
-----

"""

from constant import SETTINGS
from datatype.dataset import Dataset
from datatype.imaging import (
    create_grid,
    create_image,
    filter_image,
    to_bytes
)
from datatype.settings import Settings
from datatype.spectrogram import create_spectrogram


def main():
    dataset = Dataset('signal')
    dataframe = dataset.load()

    minimum = 20
    maximum = minimum + 5

    subset = dataframe.iloc[minimum:maximum]
    subset.reset_index(inplace=True)
    subset = subset.copy()

    path = SETTINGS.joinpath('dereverberate.json')
    dereverberate = Settings.from_file(path)

    path = SETTINGS.joinpath('spectrogram.json')
    settings = Settings.from_file(path)

    subset['signal'].apply(
        lambda x: x.normalize()
    )

    subset['signal'].apply(
        lambda x: x.dereverberate(dereverberate)
    )

    subset['spectrogram'] = (
        subset['signal']
        .apply(
            lambda x: create_spectrogram(x, settings)
        )
    )

    subset['image'] = (
        subset['spectrogram']
        .apply(
            lambda x: create_image(x)
        )
    )

    subset['original'] = (
        subset['image']
        .apply(
            lambda x: to_bytes(x)
        )
    )

    subset['preprocess'] = (
        subset['image']
        .apply(
            lambda x: filter_image(x)
        )
    )

    subset['filter'] = (
        subset['preprocess']
        .apply(
            lambda x: to_bytes(x)
        )
    )

    subset['grid'] = subset.apply(
        lambda x: create_grid(
            [
                x['original'],
                x['filter']
            ]
        ),
        axis=1
    )

    for grid in subset.grid.tolist():
        grid.show()


if __name__ == '__main__':
    main()
