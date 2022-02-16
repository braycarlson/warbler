import csv
import json
import logging

from logger import logger
from path import bootstrap, DATA, INDIVIDUALS


log = logging.getLogger(__name__)


@bootstrap
def main():
    with open(DATA.joinpath('info_file.csv'), mode='w+') as file:
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
                    data = json.load(f)

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
                    sequence = notes.get('sequence_num')
                    note = notes.get('files')

                    for s, e, l, i, n in zip(start, end, label, sequence, note):
                        writer.writerow([
                            n,
                            l,
                            name,
                            s,
                            e,

                            common,
                            scientific,
                            wav,
                            sample_rate,
                            duration
                        ])


if __name__ == '__main__':
    with logger():
        main()
