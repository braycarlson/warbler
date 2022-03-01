import json
import logging
import shutil

from logger import logger
from path import bootstrap, INDIVIDUALS, NOTES
from pathlib import Path
from pydub import AudioSegment


log = logging.getLogger(__name__)


def copy():
    for individual in INDIVIDUALS:
        notes = [
            file for file in individual.glob('notes/*.wav') if file.is_file()
        ]

        for note in notes:
            try:
                shutil.copy(
                    note,
                    NOTES
                )
            except OSError as e:
                print(f"Error: {e}")


@bootstrap
def main():
    for individual in INDIVIDUALS:
        files = [
            file for file in individual.joinpath('json').glob('*.json')
        ]

        directory = individual.joinpath('notes')

        for file in files:
            with open(file, 'r') as f:
                data = json.load(f)

                notes = (
                    data
                    .get('indvs')
                    .get(individual.stem)
                    .get('notes')
                )

                if notes is None:
                    continue

                wav = data.get('wav_loc')

                start = notes.get('start_times')
                end = notes.get('end_times')
                wavs = notes.get('files')

                for index, time in enumerate(zip(start, end), 0):
                    # Pad zero
                    index = str(index).zfill(2)
                    filename = f"{Path(wav).stem}_{index}.wav"

                    s, e = time

                    # pydub uses milliseconds
                    s = s * 1000
                    e = e * 1000

                    audio = AudioSegment.from_wav(wav)
                    audio = audio[s:e]

                    path = directory.joinpath(filename)

                    audio.export(
                        path,
                        format='wav'
                    )

                    path = path.as_posix()
                    wavs.append(path)

            with open(file, 'w+') as f:
                if len(start) == len(end) and len(start) == len(wavs):
                    json.dump(data, f, indent=2)


if __name__ == '__main__':
    with logger():
        # main()
        copy()
