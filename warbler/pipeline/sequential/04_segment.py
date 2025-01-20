from __future__ import annotations

import logging
import pandas as pd

from operator import attrgetter
from tqdm import tqdm
from typing_extensions import TYPE_CHECKING
from warbler.bootstrap import bootstrap
from warbler.datatype.dataset import Dataset
from warbler.datatype.settings import Settings
from warbler.logger import logger

if TYPE_CHECKING:
    from typing_extensions import Any


log = logging.getLogger(__name__)


def get_segmentation(row: pd.Series) -> list[dict[str, Any]]:
    recording = []

    iterable = zip(row.onset, row.offset)

    folder = row.folder
    filename = row.filename
    settings = row.settings
    signal = row.signal
    exclude = row.settings.exclude

    for sequence, (onset, offset) in enumerate(iterable, 0):
        segment = signal.segment(onset, offset)
        minimum, maximum = segment.frequencies()
        mean = segment.mean()

        metadata = {
            'folder': folder,
            'filename': filename,
            'settings': settings,
            'segment': segment,
            'onset': onset,
            'offset': offset,
            'sequence': sequence,
            'minimum': minimum,
            'mean': mean,
            'maximum': maximum,
            'exclude': False
        }

        if sequence in exclude:
            metadata['exclude'] = True

        recording.append(metadata)

    return recording


@bootstrap
def main() -> None:
    dataset = Dataset('signal')
    dataframe = dataset.load()

    tqdm.pandas(desc='Settings')

    # Update the settings for each segmentation
    dataframe['settings'] = (
        dataframe['segmentation']
        .progress_apply(Settings.resolve)
    )

    tqdm.pandas(desc='Segmentation')

    recordings = (
        dataframe
        .progress_apply(
            get_segmentation,
            axis=1
        )
    )

    collection = [
        segment
        for recording in recordings
        for segment in recording
    ]

    dataset = Dataset('segment')
    dataframe = pd.DataFrame(collection)

    tqdm.pandas(desc='Duration')

    dataframe['duration'] = (
        dataframe['segment']
        .progress_apply(
            attrgetter('duration')
        )
    )

    # Create a new dataframe for excluded notes
    mask = (
        (dataframe.exclude) |
        (dataframe.duration < 0.0250) |
        (dataframe.duration > 0.1700)
    )

    reject = dataframe.loc[mask]
    reject = reject.reset_index(drop=True)
    reject = reject.copy()

    drop = [
        'exclude',
        'segment',
        'settings'
    ]

    reject = reject.drop(drop, axis=1)

    # Mask the excluded notes from the dataframe
    dataframe = dataframe.loc[~mask]
    dataframe = dataframe.reset_index(drop=True)
    dataframe = dataframe.copy()

    drop = ['exclude', 'settings']
    dataframe = dataframe.drop(drop, axis=1)

    dataset.save(dataframe)

    if not reject.empty:
        dataset = Dataset('exclude')
        dataset.save(reject)


if __name__ == '__main__':
    with logger():
        main()
