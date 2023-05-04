import logging
import numpy as np
import pandas as pd

from bootstrap import bootstrap
from constant import SETTINGS
from datatype.dataset import Dataset
from datatype.segmentation import dynamic_threshold_segmentation
from datatype.settings import resolve, Settings
from librosa.util.exceptions import ParameterError
from logger import logger
from operator import attrgetter
from tqdm import tqdm


log = logging.getLogger(__name__)


def get_onset_offset(series):
    signal, settings = series

    try:
        result = dynamic_threshold_segmentation(signal, settings)
    except ValueError:
        log.warning(f"Shape: {signal.path.name}")
        return (np.nan, np.nan)
    except ParameterError:
        log.warning(f"Parameter(s): {signal.path.name}")
        return (np.nan, np.nan)

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
        dataframe.onset.isnull() |
        dataframe.offset.isnull()
    )

    # Create a new dataframe for erroneous recordings
    reject = dataframe.loc[mask]
    reject.reset_index(drop=True, inplace=True)
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

    reject.drop(drop, axis=1, inplace=True)

    # Mask the erroneous recordings from the dataframe
    dataframe = dataframe[~mask]
    dataframe.reset_index(drop=True, inplace=True)
    dataframe = dataframe.copy()

    dataset.save(dataframe)

    if not reject.empty:
        dataset = Dataset('ignore')
        dataframe = dataset.load()

        if dataframe is None:
            dataframe = reject.copy()
        else:
            dataframe = pd.merge(dataframe, reject,  how='outer')

        # Merge the erroneous and non-viable dataframe
        dataframe.reset_index(drop=True, inplace=True)
        dataframe = dataframe.copy()

        dataframe.sort_values(
            by=['folder', 'filename'],
            inplace=True
        )

        dataset.save(dataframe)


if __name__ == '__main__':
    with logger():
        main()
