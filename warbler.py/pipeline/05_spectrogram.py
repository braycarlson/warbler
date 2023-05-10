"""
Spectrogram
-----------

"""

import logging
import numpy as np
import warnings

from bootstrap import bootstrap
from constant import PICKLE, SETTINGS
from datatype.dataset import Dataset
from datatype.imaging import (
    create_image,
    filter_image,
    to_bytes,
    to_numpy
)
from datatype.settings import Settings
from datatype.spectrogram import (
    create_spectrogram,
    mel_matrix,
    pad,
    resize
)
from logger import logger
from tqdm import tqdm


log = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning)


@bootstrap
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
        np.save(path, matrix, allow_pickle=True)

    tqdm.pandas(desc='Spectrogram')

    dataframe['spectrogram'] = (
        dataframe['segment']
        .progress_apply(
            create_spectrogram,
            args=(settings, matrix)
        )
    )

    tqdm.pandas(desc='Resize')

    dataframe['scale'] = (
        dataframe['spectrogram']
        .progress_apply(resize)
    )

    # Original spectrogram
    padding = dataframe['spectrogram'].progress_apply(
        lambda x: len(x.T)
    ).max()

    tqdm.pandas(desc='Padding')

    dataframe['spectrogram'] = (
        dataframe['spectrogram']
        .apply(
            pad,
            args=(padding, )
        )
    )

    # Log-scaled spectrogram
    padding = dataframe['scale'].progress_apply(
        lambda x: len(x.T)
    ).max()

    tqdm.pandas(desc='Padding')

    dataframe['scale'] = (
        dataframe['scale']
        .progress_apply(
            pad,
            args=(padding, )
        )
    )

    tqdm.pandas(desc='Image')

    # Use 'spectrogram' or 'scale'
    dataframe['original'] = (
        dataframe['scale']
        .progress_apply(create_image)
    )

    tqdm.pandas(desc='Array')

    dataframe['original_array'] = (
        dataframe['original']
        .progress_apply(to_numpy)
    )

    tqdm.pandas(desc='Bytes')

    dataframe['original_bytes'] = (
        dataframe['original']
        .progress_apply(to_bytes)
    )

    tqdm.pandas(desc='Array')

    dataframe['filter_array'] = (
        dataframe['original']
        .progress_apply(filter_image)
    )

    tqdm.pandas(desc='Bytes')

    dataframe['filter_bytes'] = (
        dataframe['filter_array']
        .progress_apply(to_bytes)
    )

    dataset.save(dataframe)


if __name__ == '__main__':
    with logger():
        main()
