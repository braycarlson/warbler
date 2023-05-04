import logging
import pandas as pd

from bootstrap import bootstrap
from datatype.dataset import Dataset
from datatype.settings import resolve
from logger import logger
from operator import attrgetter
from tqdm import tqdm


log = logging.getLogger(__name__)


def get_segmentation(row):
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

        template = {
            'folder': folder,
            'filename': filename,
            'settings': settings,
            'segment': segment,
            'onset': onset,
            'offset': offset,
            'sequence': sequence,
            'minimum': minimum,
            'maximum': maximum,
            'exclude': False
        }

        if sequence in exclude:
            template['exclude'] = True

        recording.append(template)

    return recording


@bootstrap
def main():
    dataset = Dataset('signal')
    dataframe = dataset.load()

    tqdm.pandas(desc='Settings')

    # Update the settings for each segmentation
    dataframe['settings'] = (
        dataframe['segmentation']
        .progress_apply(resolve)
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
    reject.reset_index(drop=True, inplace=True)
    reject = reject.copy()

    drop = [
        'exclude',
        'segment',
        'settings'
    ]

    reject.drop(drop, axis=1, inplace=True)

    # Mask the excluded notes from the dataframe
    dataframe = dataframe.loc[~mask]
    dataframe.reset_index(drop=True, inplace=True)
    dataframe = dataframe.copy()

    drop = ['exclude']

    dataframe.drop(drop, axis=1, inplace=True)

    dataset.save(dataframe)

    if not reject.empty:
        dataset = Dataset('exclude')
        dataset.save(reject)


if __name__ == '__main__':
    with logger():
        main()
