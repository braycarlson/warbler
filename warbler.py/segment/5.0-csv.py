import csv
import json
import logging

from logger import logger
from path import bootstrap, DATA, INDIVIDUALS
from pathlib import Path


log = logging.getLogger(__name__)


@bootstrap
def main():
    with open(DATA.joinpath('notes.csv'), mode='w+') as file:
        writer = csv.writer(
            file,
            delimiter=';',
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
            lineterminator='\n',
        )

        writer.writerow([
            # Metadata for the individual note
            'filename',
            'label',
            'individual',
            'path',
            'start',
            'end',

            # Metadata for the original song
            'common',
            'scientific',
            'song',
            'sample_rate',
            'song_duration',
        ])

        for individual in INDIVIDUALS:
            files = [
                file for file in individual.joinpath('json').glob('*.json')
            ]

            name = individual.stem.replace('_STE2017', '')

            for file in files:
                with open(file, 'r') as f:
                    try:
                        data = json.load(f)
                    except json.decoder.JSONDecodeError:
                        print(f"Unable to open: {file}")
                        continue

                    common = data.get('common_name')
                    scientific = data.get('species')

                    # The metadata for the original song
                    wav = data.get('wav_loc')
                    sample_rate = data.get('samplerate_hz')
                    duration = data.get('length_s')

                    notes = data.get('indvs').get(individual.stem).get('notes')

                    # The metadata for the individual note
                    start = notes.get('start_times')
                    end = notes.get('end_times')
                    label = notes.get('labels')
                    note = notes.get('files')

                    for s, e, l, n in zip(start, end, label, note):
                        writer.writerow([
                            Path(n).name,
                            l,
                            name,
                            n,
                            s,
                            e,

                            common,
                            scientific,
                            Path(wav).stem,
                            sample_rate,
                            duration
                        ])

    print('Done')


if __name__ == '__main__':
    with logger():
        main()
