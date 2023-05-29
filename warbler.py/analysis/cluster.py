"""
Cluster
-------

"""

from __future__ import annotations

from constant import DATASET
from datatype.dataset import Dataset
from datatype.settings import Settings


def main() -> None:
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
