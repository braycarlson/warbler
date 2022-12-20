import logging
import numpy as np
import swifter

from bootstrap import bootstrap
from constant import SETTINGS
from datatype.dataset import Dataset
from datatype.segmentation import dynamic_threshold_segmentation
from datatype.settings import resolve, Settings
from logger import logger


log = logging.getLogger(__name__)


def get_onset_offset(series):
    signal, settings = series
    print(f"Processing: {signal.path.name}")

    result = dynamic_threshold_segmentation(signal, settings)

    if result is None:
        log.warning(f"Envelope: {signal.path.name}")
        return (np.nan, np.nan)

    onset = result.get('onset')
    offset = result.get('offset')

    if len(onset) == len(offset):
        return (onset, offset)

    log.warning(f"Mismatch: {signal.path.name}")
    return (np.nan, np.nan)


@bootstrap
def main():
    swifter.set_defaults(
        npartitions=None,
        dask_threshold=1,
        scheduler='processes',
        progress_bar=False,
        progress_bar_desc=False,
        allow_dask_on_strings=True,
        force_parallel=True,
    )

    dataset = Dataset('signal')
    dataframe = dataset.load()

    path = SETTINGS.joinpath('spectrogram.json')
    settings = Settings.from_file(path)

    dataframe['rate'] = (
        dataframe['signal']
        .swifter
        .apply(
            lambda x: x.rate
        )
    )

    dataframe['duration'] = (
        dataframe['signal']
        .swifter
        .apply(
            lambda x: x.duration
        )
    )

    dataframe['settings'] = (
        dataframe['segmentation']
        .swifter
        .apply(
            lambda x: resolve(x)
        )
    )

    dataframe['exclude'] = [settings.exclude] * len(dataframe)

    source = ['signal', 'settings']
    destination = ['onset', 'offset']

    dataframe[destination] = np.nan

    dataframe[destination] = (
        dataframe[source]
        .swifter
        .apply(
            lambda x: get_onset_offset(x),
            axis=1,
            result_type='expand'
        )
    )

    dataframe['ignore'] = False

    mask = (
        dataframe.onset.isnull() |
        dataframe.offset.isnull()
    )

    dataframe.loc[mask, 'ignore'] = True

    dataset.save(dataframe)


if __name__ == '__main__':
    with logger():
        main()
