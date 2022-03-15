import pandas as pd

from natsort import os_sorted
from path import INDIVIDUALS


def first():
    template = {
        'individual': [],
        'filename': [],
        'updated_filename': [],
        'printout_page_number': [],
        'printout_viability': [],
        'printout_viability_reason': [],
        'printout_notes': []
    }

    for individual in INDIVIDUALS:
        songs = [
            file for file in individual.joinpath('wav').glob('*.wav')
            if file.is_file()
        ]

        songs = os_sorted(songs)

        for index, song in enumerate(songs, 1):
            template['individual'].append(individual.stem)
            template['filename'].append(song.stem)
            template['updated_filename'].append(None)
            template['printout_page_number'].append(index)
            template['printout_viability'].append(None),
            template['printout_viability_reason'].append(None),
            template['printout_notes'].append(None),

    return pd.DataFrame.from_dict(template)


def second():
    template = {
        'individual': [],
        'filename': [],
        'updated_filename': [],
        'printout_page_number': [],
        'printout_viability': [],
        'printout_viability_reason': [],
        'printout_notes': [],
        'python_viability': [],
        'python_viability_reason': [],
        'python_notes': []
    }

    for individual in INDIVIDUALS:
        songs = [
            file for file in individual.joinpath('wav').glob('*.wav')
            if file.is_file()
        ]

        songs = os_sorted(songs)

        for index, song in enumerate(songs, 1):
            template['individual'].append(individual.stem)
            template['filename'].append(song.stem)
            template['updated_filename'].append(None)
            template['printout_page_number'].append(index)
            template['printout_viability'].append(None),
            template['printout_viability_reason'].append(None),
            template['printout_notes'].append(None),
            template['python_viability'].append(None),
            template['python_viability_reason'].append(None),
            template['python_notes'].append(None),

    return pd.DataFrame.from_dict(template)


def third():
    template = {
        'printout_viability': [
            'Y',
            'N',
            'P',
            None,
            None,
            None,
            None,
            None,
            None,
        ],
        'printout_meaning': [
            'Yes (fully viable)',
            'No (not viable)',
            'Partially viable',
            None,
            None,
            None,
            None,
            None,
            None,

        ],
        '': [
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
        'python_viability': [
            'CO',
            'CS',
            'CT',
            'FB',
            'N',
            'NP',
            'P',
            'QA',
            'Y'
        ],
        'python_meaning': [
            'cut out overlapping note',
            'can\'t segment',
            'cut off tail end(s) of sound file',
            'limit frequency bandwidth',
            'No (not viable)',
            'noise present',
            'Partially viable',
            'quiet (try increasing amplitude)',
            'Yes (fully viable)'
        ]
    }

    return pd.DataFrame.from_dict(template)


def main():
    one = first()
    two = second()
    three = third()

    with pd.ExcelWriter('test.xlsx') as writer:
        one.to_excel(
            writer,
            sheet_name='2017_printout_viability',
            index=False
        )

        two.to_excel(
            writer,
            sheet_name='2017_python_viability',
            index=False
        )

        three.to_excel(
            writer,
            sheet_name='2017_viability_keys',
            index=False
        )


if __name__ == '__main__':
    main()
