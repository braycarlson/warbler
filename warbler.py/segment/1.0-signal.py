import logging
import pandas as pd

from bootstrap import bootstrap
from constant import DATASET, relative, SPREADSHEET
from datatype.dataset import Dataset
from datatype.signal import Signal
from logger import logger
from natsort import os_sorted


log = logging.getLogger(__name__)


@bootstrap
def main():
    dataset = Dataset('signal')

    collection = []

    individuals = [
        individual
        for individual in DATASET.glob('*/')
        if individual.is_dir()
    ]

    # Sort directories according to OS
    individuals = os_sorted(individuals)

    for individual in individuals:
        if not individual.is_dir():
            continue

        folder = individual.stem

        recordings = os_sorted(
            individual.glob('recordings/*.wav')
        )

        segmentations = os_sorted(
            individual.glob('segmentation/*.json')
        )

        for recording, segmentation in zip(recordings, segmentations):
            filename = recording.stem
            print(f"Processing: {filename}")

            signal = Signal(recording)

            template = {
                'folder': None,
                'filename': None,
                'recording': None,
                'segmentation': None,
                'signal': None,
            }

            template['folder'] = folder
            template['filename'] = filename
            template['recording'] = relative(recording)
            template['segmentation'] = relative(segmentation)
            template['signal'] = signal

            collection.append(template)

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

            viability = dataframe.loc[mask]
            ignore = viability.filename.to_numpy()

            dataframe = dataframe[~dataframe.filename.isin(ignore)]
            dataframe.reset_index(inplace=True)

        dataset.save(dataframe)


if __name__ == '__main__':
    with logger():
        main()
