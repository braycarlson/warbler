import json
import logging
import shutil

from bootstrap import bootstrap
from collections import OrderedDict
from logger import logger
from path import DATA, DATASET, DIRECTORIES, INDIVIDUALS
from pydub import AudioSegment
from tqdm import tqdm


log = logging.getLogger(__name__)


@bootstrap
def main():
    aif = [file for file in DATASET.glob('*/*.aif')]
    wav = [file for file in DATA.glob('*/wav/*.wav')]

    a = len(aif)
    w = len(wav)

    if a == w:
        log.info(f"{a} .aif files")
        log.info(f"{w} .wav files")

        return

    for directory, individual in zip(DIRECTORIES, INDIVIDUALS):
        if individual.stem == directory.stem:
            # Note: pydub uses ffmpeg
            files = [file for file in directory.glob('*.aif')]
            total = len(files)

            for file in tqdm(files, total=total):
                filename = file.stem

                wav = individual.joinpath('wav')
                metadata = individual.joinpath('json')
                parameter = individual.joinpath('parameter')

                # Song(s)
                w = wav.joinpath(filename + '.wav')

                song = AudioSegment.from_file(file, format='aiff')
                song.export(w, format='wav')

                # Parameter(s)
                p = parameter.joinpath(filename + '.json')

                if not p.is_file():
                    p.touch()

                    shutil.copy(
                        'parameter.json',
                        p
                    )

                # Metadata
                m = metadata.joinpath(filename + '.json')

                if not m.is_file():
                    m.touch()


if __name__ == '__main__':
    with logger():
        main()
