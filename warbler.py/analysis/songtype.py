from __future__ import annotations

import pandas as pd

from constant import OUTPUT
from datatype.dataset import Dataset
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def group_songtype(dataframe: pd.DataFrame) -> dict[str, list[str]]:
    column = ['filename', 'songtype']

    dataframe['songtype'] = dataframe.filename.str.extract(r'STE(\d+)', expand=False)
    count = dataframe.songtype.value_counts()
    valid = count[count > 1].index

    return dataframe[dataframe.songtype.isin(valid)][column]


def main() -> None:
    dataset = Dataset('signal')
    dataframe = dataset.load()

    subset = group_songtype(dataframe)

    subset = subset.sort_values(
        ['songtype'],
        ascending=True
    )

    path = OUTPUT.joinpath('songtype.csv')
    subset.to_csv(path, index=False)

    group = subset.groupby('songtype').filename.unique()


if __name__ == '__main__':
    main()
