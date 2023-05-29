"""
Spreadsheet
-----------

"""

from __future__ import annotations

from constant import OUTPUT
from datatype.dataset import Dataset


def main() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    drop = [
        'segment',
        'spectrogram',
        'scale',
        'original',
        'original_array',
        'original_bytes',
        'filter_array',
        'filter_bytes'
    ]

    dataframe = dataframe.drop(drop, axis=1)

    path = OUTPUT.joinpath('dataset.xlsx')

    dataframe.to_excel(
        path,
        sheet_name='dataset'
    )


if __name__ == '__main__':
    main()
