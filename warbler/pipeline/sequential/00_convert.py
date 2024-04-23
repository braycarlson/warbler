from __future__ import annotations

import librosa
import logging
import soundfile as sf

from warbler.bootstrap import bootstrap
from warbler.constant import (
    DATASET,
    DIRECTORIES,
    INDIVIDUALS,
    ORIGINAL,
    SETTINGS
)
from warbler.datatype.settings import Settings
from logger import logger


log = logging.getLogger(__name__)


@bootstrap
def main() -> None:
    suffix = [
        '.aac',
        '.aif',
        '.aiff',
        '.flac',
        '.m4a',
        '.mp3',
        '.octet-stream',
        '.ogg',
        '.wav'
    ]

    raw = [
        file
        for file in ORIGINAL.glob('*/*')
        if file.is_file() and file.suffix.lower() in suffix
    ]

    wav = [
        file
        for file in DATASET.glob('*/recordings/*.wav')
        if file.is_file()
    ]

    if len(raw) == len(wav):
        return

    path = SETTINGS.joinpath('segmentation.json')
    settings = Settings.from_file(path)

    sr = settings.sample_rate

    for directory, individual in zip(DIRECTORIES, INDIVIDUALS):
        if individual.stem == directory.stem:
            files = [
                file
                for file in directory.glob('*')
                if file.is_file() and file.suffix.lower() in suffix
            ]

            for file in files:
                recording = individual.joinpath('recordings')
                segmentation = individual.joinpath('segmentation')

                # Recording(s)
                path = recording.joinpath(file.stem + '.wav')

                signal, rate = librosa.load(file, sr=sr)
                sf.write(path, signal, rate, format='wav')

                # Settings(s)
                path = segmentation.joinpath(file.stem + '.json')

                if not path.is_file():
                    path.touch()
                    settings.save(path)


if __name__ == '__main__':
    with logger():
        main()
