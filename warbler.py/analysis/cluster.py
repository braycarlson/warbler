"""
This module provides a function that loads a dataset, selects a subset of the
data, and modifies the settings for each file in the subset.

Functions:
    - main(): loads a dataset, selects a subset of the data, and modifies the settings for each file in the subset.

Requirements:
    - Python 3.x
    - pandas library
"""

from constant import DATASET
from datatype.dataset import Dataset
from datatype.settings import Settings


def main():
    dataset = Dataset('segment')
    dataframe = dataset.load()

    dataframe = (
        dataframe[dataframe.hdbscan_label_2d == 7]
        .reset_index(drop=True)
    )

    filenames = dataframe.filename.unique().tolist()

    for filename in filenames:
        subset = dataframe.loc[
            dataframe['filename'] == filename,
            ['folder', 'sequence']
        ]

        folder = subset.folder.tolist()
        sequence = subset.sequence.tolist()

        path = DATASET.joinpath(
            folder[0],
            'segmentation',
            filename + '.json'
        )

        settings = Settings.from_file(path)

        settings.exclude.extend(sequence)

        # Remove duplicate(s) and sort list
        exclude = set(settings.exclude)
        exclude = list(exclude)
        exclude = sorted(exclude)

        settings.exclude = exclude

        settings.save(path)


if __name__ == '__main__':
    main()
