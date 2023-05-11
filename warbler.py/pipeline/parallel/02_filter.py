"""
Filter
------

"""

import logging

from bootstrap import bootstrap
from constant import SETTINGS
from datatype.dataset import Dataset
from datatype.settings import resolve, Settings
from datatype.signal  import Signal
from joblib import delayed, Parallel
from logger import logger
from tqdm import tqdm


log = logging.getLogger(__name__)


@bootstrap
def main() -> None:
    dataset = Dataset('signal')
    dataframe = dataset.load()

    path = SETTINGS.joinpath('spectrogram.json')
    settings = Settings.from_file(path)

    path = SETTINGS.joinpath('dereverberate.json')
    dereverberate = Settings.from_file(path)

    tqdm.pandas(desc='Settings')

    dataframe['settings'] = (
        dataframe['segmentation']
        .progress_apply(resolve)
    )

    # Bandpass filter
    if settings.bandpass_filter:
        tqdm.pandas(desc='Bandpass Filter')

        try:
            column = ['signal', 'settings']

            dataframe[column].progress_apply(
                lambda x: x.signal.filter(
                    x.settings.butter_lowcut,
                    x.settings.butter_highcut
                ),
                axis=1
            )
        except Exception as exception:
            log.warning(exception)

    # Normalize
    if settings.normalize:
        tqdm.pandas(desc='Normalize')

        try:
            dataframe['signal'].progress_apply(
                lambda x: x.normalize()
            )
        except Exception as exception:
            log.warning(exception)

    # Dereverberate
    if settings.dereverberate:
        def func(signal: Signal) -> Signal:
            signal.dereverberate(dereverberate)
            return signal

        try:
            dataframe['signal'] = Parallel(n_jobs=4)(
                delayed(func)(x)
                for x in tqdm(dataframe['signal'], desc='Dereverberate')
            )
        except Exception as exception:
            log.warning(exception)

    # Reduce noise
    if settings.reduce_noise:
        tqdm.pandas(desc='Reduce Noise')

        try:
            dataframe['signal'].progress_apply(
                lambda x: x.reduce()
            )
        except Exception as exception:
            log.warning(exception)

    dataset.save(dataframe)


if __name__ == '__main__':
    with logger():
        main()
