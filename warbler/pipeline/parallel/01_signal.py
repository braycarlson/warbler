from __future__ import annotations

import logging
import pandas as pd

from natsort import os_sorted
from tqdm import tqdm
from warbler.bootstrap import bootstrap
from warbler.constant import CWD, DATASET, SPREADSHEET
from warbler.datatype.dataset import Dataset
from warbler.datatype.signal import Signal
from warbler.logger import logger


log = logging.getLogger(__name__)


@bootstrap
def main() -> None:
    dataset = Dataset('signal')

    collection = []

    individuals = [
        individual
        for individual in DATASET.glob('*/')
        if individual.is_dir()
    ]

    # Sort directories according to OS
    individuals = os_sorted(individuals)
    total = len(individuals)

    progress = tqdm(individuals, desc='Signal', total=total)

    for individual in progress:
        if not individual.is_dir():
            continue

        folder = individual.stem
        progress.set_description(folder)

        recordings = os_sorted(
            individual.glob('recordings/*.wav')
        )

        segmentations = os_sorted(
            individual.glob('segmentation/*.json')
        )

        for recording, segmentation in zip(recordings, segmentations):
            filename = recording.stem

            signal = Signal(recording)

            metadata = {
                'folder': None,
                'filename': None,
                'recording': None,
                'segmentation': None,
                'signal': None,
            }

            metadata['folder'] = folder
            metadata['filename'] = filename
            metadata['recording'] = recording.relative_to(CWD).as_posix()
            metadata['segmentation'] = segmentation.relative_to(CWD).as_posix()
            metadata['signal'] = signal.serialize()
            metadata['rate'] = signal.rate

            collection.append(metadata)

        dataframe = pd.DataFrame(collection)

        # Temporary
        spreadsheet = SPREADSHEET.joinpath('2017.xlsx')

        if spreadsheet.is_file():
            viability = pd.read_excel(
                spreadsheet,
                sheet_name='2017_recording_viability',
                engine='openpyxl'
            )

            mask = (viability['recording_viability'] == 'N')

            # Create a new dataframe for non-viable recordings
            ignore = Dataset('ignore')

            reject = dataframe.loc[mask]
            reject = reject.reset_index(drop=True)
            reject = reject.copy()

            reject = reject.drop('signal', axis=1)

            ignore.save(reject)

            # Mask the non-viable recordings from the dataframe
            filename = reject.filename.to_numpy()

            dataframe = dataframe[~dataframe.filename.isin(filename)]
            dataframe = dataframe.reset_index(drop=True)
            dataframe = dataframe.copy()

        dataset.save(dataframe)


if __name__ == '__main__':
    with logger():
        main()
