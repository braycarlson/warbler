import logging
import pandas as pd
import swifter

from bootstrap import bootstrap
from datatype.dataset import Dataset
from logger import logger


log = logging.getLogger(__name__)


def get_segment(signal, settings, onset, offset, exclude):
    recording = []

    iterable = zip(onset, offset)

    for sequence, (onset, offset) in enumerate(iterable, 0):
        segment = signal.segment(onset, offset)

        if sequence in exclude:
            recording.append(
                [segment, onset, offset, sequence, True]
            )
        else:
            recording.append(
                [segment, onset, offset, sequence, False]
            )

    return recording


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

    dataframe = dataframe[~dataframe.ignore]
    dataframe.reset_index(inplace=True)

    segments = (
        dataframe
        .swifter
        .apply(
            lambda x: get_segment(
                x.signal,
                x.settings,
                x.onset,
                x.offset,
                x.exclude
            ),
            axis=1
        )
    ).to_numpy()

    folders = dataframe.folder.to_numpy()
    filenames = dataframe.filename.to_numpy()
    settings = dataframe.settings.to_numpy()

    identifiers = [
        [(folder, filename, setting)] * len(segments)
        for folder, filename, setting, segment in
        zip(folders, filenames, settings, segments)
    ]

    collection = []

    for identifier, segment in zip(identifiers, segments):
        for x, y in zip(identifier, segment):
            folder, filename, setting = x

            y.insert(0, folder)
            y.insert(1, filename)
            y.insert(2, setting)

            collection.append(y)

    columns = [
        'folder',
        'filename',
        'settings',
        'segment',
        'onset',
        'offset',
        'sequence',
        'exclude'
    ]

    dataset = Dataset('segment')

    dataframe = pd.DataFrame(
        collection,
        columns=columns
    )

    dataset.save(dataframe)


if __name__ == '__main__':
    with logger():
        main()
