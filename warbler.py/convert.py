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
    aif = [file for file in ORIGINAL.glob('*/*.aif')]
    wav = [file for file in DATASET.glob('*/recordings/*.wav')]

    if len(aif) == len(wav):
        return

    for directory, individual in zip(DIRECTORIES, INDIVIDUALS):
        if individual.stem == directory.stem:
            # Note: pydub uses ffmpeg
            files = [file for file in directory.glob('*.aif')]
            total = len(files)

            for file in tqdm(files, total=total):
                filename = file.stem

                r = individual.joinpath('recordings')
                m = individual.joinpath('metadata')
                p = individual.joinpath('parameters')

                # Recording(s)
                recording = r.joinpath(filename + '.wav')

                song = AudioSegment.from_file(file, format='aiff')
                song.export(recording, format='wav')

                # Parameter(s)
                parameter = p.joinpath(filename + '.json')

                if not parameter.is_file():
                    parameter.touch()
                    shutil.copy('parameters.json', parameter)

                # Metadata
                metadata = m.joinpath(filename + '.json')

                if not metadata.is_file():
                    metadata.touch()


if __name__ == '__main__':
    with logger():
        main()
