import logging
import shutil

from bootstrap import bootstrap
from constant import (
    DATASET,
    DIRECTORIES,
    INDIVIDUALS,
    ORIGINAL
)
from logger import logger
from pydub import AudioSegment
from tqdm import tqdm


log = logging.getLogger(__name__)


@bootstrap
def main():
    aif = [
        file
        for file in ORIGINAL.glob('*/*.aif')
        if file.is_file()
    ]

    wav = [
        file
        for file in DATASET.glob('*/recordings/*.wav')
        if file.is_file()
    ]

    if len(aif) == len(wav):
        return

    for directory, individual in zip(DIRECTORIES, INDIVIDUALS):
        if individual.stem == directory.stem:
            # Note: pydub uses ffmpeg
            files = [file for file in directory.glob('*.aif')]
            total = len(files)

            for file in tqdm(files, total=total):
                filename = file.stem

                m = individual.joinpath('metadata')
                r = individual.joinpath('recordings')
                s = individual.joinpath('segmentation')

                # Recording(s)
                recording = r.joinpath(filename + '.wav')

                song = AudioSegment.from_file(file, format='aiff')
                song.export(recording, format='wav')

                # Parameter(s)
                settings = s.joinpath(filename + '.json')

                if not settings.is_file():
                    settings.touch()
                    shutil.copy('segmentation.json', settings)

                # Metadata
                metadata = m.joinpath(filename + '.json')

                if not metadata.is_file():
                    metadata.touch()


if __name__ == '__main__':
    with logger():
        main()
