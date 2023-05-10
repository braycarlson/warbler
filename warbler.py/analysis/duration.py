"""
Duration
--------

"""

import logging
import numpy as np
import warnings

from constant import PICKLE, SETTINGS
from datatype.dataset import Dataset
from datatype.imaging import (
    create_grid,
    create_image,
    filter_image
)
from datatype.settings import Settings
from datatype.spectrogram import (
    create_spectrogram,
    mel_matrix,
    pad,
    resize
)
from logger import logger


log = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)


def main() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    path = SETTINGS.joinpath('spectrogram.json')
    settings = Settings.from_file(path)

    path = PICKLE.joinpath('matrix.npy')

    if path.is_file():
        matrix = np.load(path, allow_pickle=True)
    else:
        matrix = mel_matrix(settings)

    dataframe['duration'] = dataframe['duration'].astype(float)
    dataframe = dataframe[~dataframe.isna().any(axis=1)]

    dataframe = dataframe.reset_index(drop=True)
    dataframe = dataframe.copy()

    padding = dataframe['scale'].apply(
        lambda x: len(x.T)
    ).max()

    subset = dataframe.nlargest(100, 'duration')
    # subset = dataframe.nsmallest(100, 'duration')

    subset['spectrogram'] = (
        subset['segment']
        .apply(
            lambda x: create_spectrogram(x, settings, matrix)
        )
    )

    subset['scale'] = (
        subset['spectrogram']
        .apply(
            lambda x: resize(x)
        )
    )

    subset['scale'] = (
        subset['scale']
        .apply(
            lambda x: pad(x, padding)
        )
    )

    subset['image'] = (
        subset['scale']
        .apply(
            lambda x: create_image(x)
        )
    )

    subset['filter_array'] = (
        subset['image']
        .apply(
            lambda x: filter_image(x)
        )
    )

    image = subset.filter_array.tolist()

    grid = create_grid(image)
    grid.show()


if __name__ == '__main__':
    with logger():
        main()
