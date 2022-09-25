import logging
import pandas as pd

from abc import abstractmethod, ABC
from bootstrap import bootstrap
from constant import (
    CWD,
    DATASET,
    SETTINGS,
    SPREADSHEET
)
from datatype.file import File
from datatype.settings import Settings
from datatype.signal import Signal
from logger import logger
from natsort import os_sorted
from pathlib import PurePath


log = logging.getLogger(__name__)


class Strategy(ABC):
    @abstractmethod
    def get(self):
        pass

    @abstractmethod
    def process(self):
        pass

    @staticmethod
    def relative(path):
        pure = PurePath(path)
        return pure.relative_to(CWD)


class Spreadsheet(Strategy):
    def get(self):
        spreadsheet = SPREADSHEET.joinpath('2017.xlsx')

        dataframe = pd.read_excel(
            spreadsheet,
            sheet_name='2017_segment_viability',
            engine='openpyxl'
        )

        dataframe = dataframe.where(
            pd.notnull(dataframe),
            None
        )

        return zip(
            dataframe.individual.to_list(),
            dataframe.updated_filename.to_list(),
            dataframe.printout_page_number.to_list(),
            dataframe.segment_viability.to_list(),
            dataframe.segment_viability_reason.to_list(),
            dataframe.segment_notes.to_list()
        )

    def process(self):
        good = File('good.xz')
        mediocre = File('mediocre.xz')
        bad = File('bad.xz')

        path = SETTINGS.joinpath('dereverberate.json')
        dereverberate = Settings.from_file(path)

        for folder, filename, page, viability, reason, notes in self.get():
            print(f"Processing: {filename}")

            recording = DATASET.joinpath(
                folder,
                'recordings',
                filename + '.wav'
            )

            if not recording.is_file() or recording.stat().st_size == 0:
                log.info(f"{recording} was not found")
                raise FileNotFoundError(f"{recording} was not found")

            recording = PurePath(recording)

            segmentation = DATASET.joinpath(
                folder,
                'segmentation',
                filename + '.json'
            )

            if not segmentation.is_file() or segmentation.stat().st_size == 0:
                log.info(f"{segmentation} was not found")
                raise FileNotFoundError(f"{segmentation} was not found")

            segmentation = PurePath(segmentation)

            signal = Signal(recording)
            signal.dereverberate(dereverberate)

            template = {
                'folder': None,
                'filename': None,
                'page': None,
                'reason': None,
                'notes': None,
                'signal': None,
                'segmentation': None,
            }

            template['filename'] = filename
            template['folder'] = folder
            template['page'] = page
            template['reason'] = reason
            template['notes'] = notes
            template['recording'] = self.relative(recording)
            template['segmentation'] = self.relative(segmentation)
            template['signal'] = signal

            if viability == 'Y':
                good.insert(template)
            elif viability == 'P':
                mediocre.insert(template)
            else:
                bad.insert(template)

        good.sort()
        mediocre.sort()
        bad.sort()

        good.save()
        mediocre.save()
        bad.save()


class Folder(Strategy):
    def get(self):
        return os_sorted([
            individual
            for individual in DATASET.glob('*/')
            if individual.is_dir()
        ])

    def process(self):
        path = SETTINGS.joinpath('dereverberate.json')
        dereverberate = Settings.from_file(path)

        for individual in self.get():
            if not individual.is_dir():
                continue

            folder = individual.stem
            file = File(folder + '.xz')

            recordings = os_sorted(
                individual.glob('recordings/*.wav')
            )

            segmentations = os_sorted(
                individual.glob('segmentation/*.json')
            )

            individual = PurePath(individual)

            for recording, segmentation in zip(recordings, segmentations):
                filename = recording.stem
                print(f"Processing: {filename}")

                signal = Signal(recording)
                signal.dereverberate(dereverberate)

                template = {
                    'folder': None,
                    'filename': None,
                    'individual': None,
                    'signal': None,
                    'segmentation': None
                }

                template['folder'] = folder
                template['filename'] = filename
                template['recording'] = self.relative(recording)
                template['segmentation'] = self.relative(segmentation)
                template['signal'] = signal

                file.insert(template)

            file.sort()
            file.save()


class Source:
    def __init__(self):
        self._strategy = Spreadsheet()

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy):
        self._strategy = strategy

    def get(self):
        return self.strategy.get()

    def process(self):
        return self.strategy.process()


@bootstrap
def main():
    source = Source()
    source.process()


if __name__ == '__main__':
    with logger():
        main()
