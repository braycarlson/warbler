from __future__ import annotations

import logging
import numpy as np
import pandas as pd

from librosa.util.exceptions import ParameterError
from logger import logger
from operator import attrgetter
from tqdm import tqdm
from warbler.bootstrap import bootstrap
from warbler.constant import SETTINGS
from warbler.datatype.dataset import Dataset
from warbler.datatype.segmentation import DynamicThresholdSegmentation
from warbler.datatype.settings import resolve, Settings


log = logging.getLogger(__name__)


def get_onset_offset(
    series: pd.Series
) -> tuple[float, float] | tuple[int, int]:
    signal, settings = series

    try:
        algorithm = DynamicThresholdSegmentation(
            signal=signal,
            settings=settings
        )

        algorithm.start()
    except ValueError:
        log.warning(f"Shape: {signal.path.name}")
        return (np.nan, np.nan)
    except ParameterError:
        log.warning(f"Parameter(s): {signal.path.name}")
        return (np.nan, np.nan)

    if algorithm.component is None:
        log.warning(f"Envelope: {signal.path.name}")
        return (np.nan, np.nan)

    onset = algorithm.component.get('onset')
    offset = algorithm.component.get('offset')

    if len(onset) == len(offset):
        return (onset, offset)

    log.warning(f"Mismatch: {signal.path.name}")
    return (np.nan, np.nan)


@bootstrap
def main() -> None:
    dataset = Dataset('signal')
    dataframe = dataset.load()

    path = SETTINGS.joinpath('spectrogram.json')
    settings = Settings.from_file(path)

    tqdm.pandas(desc='Rate')

    dataframe['rate'] = (
        dataframe['signal']
        .progress_apply(
            attrgetter('rate')
        )
    )

    tqdm.pandas(desc='Duration')

    dataframe['duration'] = (
        dataframe['signal']
        .progress_apply(
            attrgetter('duration')
        )
    )

    tqdm.pandas(desc='Settings')

    dataframe['settings'] = (
        dataframe['segmentation']
        .progress_apply(resolve)
    )

    dataframe['exclude'] = [settings.exclude] * len(dataframe)

    source = ['signal', 'settings']
    destination = ['onset', 'offset']

    dataframe[destination] = np.nan

    tqdm.pandas(desc='Dynamic Threshold Segmentation')

    dataframe[destination] = (
        dataframe[source]
        .progress_apply(
            lambda x: get_onset_offset(x),
            axis=1,
            result_type='expand'
        )
    )

    mask = (
        dataframe.onset.isna() |
        dataframe.offset.isna()
    )

    # Create a new dataframe for erroneous recordings
    reject = dataframe.loc[mask]
    reject = reject.reset_index(drop=True)
    reject = reject.copy()

    drop = [
        'duration',
        'exclude',
        'onset',
        'offset',
        'rate',
        'settings',
        'signal'
    ]

    reject = reject.drop(drop, axis=1)

    # Mask the erroneous recordings from the dataframe
    dataframe = dataframe[~mask]
    dataframe = dataframe.reset_index(drop=True)
    dataframe = dataframe.copy()

    dataset.save(dataframe)

    if not reject.empty:
        dataset = Dataset('ignore')
        dataframe = dataset.load()

        if dataframe is None:
            dataframe = reject.copy()
        else:
            dataframe = dataframe.merge(reject,  how='outer')

        # Merge the erroneous and non-viable dataframe
        dataframe = dataframe.reset_index(drop=True)
        dataframe = dataframe.copy()

        dataframe = dataframe.sort_values(
            by=['folder', 'filename']
        )

        dataset.save(dataframe)


if __name__ == '__main__':
    with logger():
        main()
