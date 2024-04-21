from __future__ import annotations

from matplotlib.cm import viridis
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

    from typing_extensions import Any, Iterator


class Partition():
    def __init__(self, dataframe: pd.DataFrame = None, column: str = None):
        self.colormap = viridis
        self.column = column
        self.dataframe = dataframe

    @property
    def unit(self) -> str:
        legend = {
            'duration': 's',
            'bandwidth': 'Hz',
            'maximum': 'Hz',
            'mean': 'Hz',
            'midpoint': 'Hz',
            'minimum': 'Hz'
        }

        return legend.get(self.column)

    def palette(self, labels: Any) -> Iterator[list[str], list[float]]:
        total = len(labels)

        palette = [
            self.colormap(i / (total - 1))
            for i in range(total)
        ]

        return dict(
            zip(labels, palette)
        )

    def divide(self) -> tuple[list[pd.Series], list[str]]:
        minimum = self.dataframe[self.column].min()
        median = self.dataframe[self.column].median()
        maximum = self.dataframe[self.column].max()

        minimum_median = (minimum + median) / 2
        maximum_median = (maximum + median) / 2

        partitions = [
            self.dataframe.loc[
                (self.dataframe[self.column] >= minimum) &
                (self.dataframe[self.column] < minimum_median)
            ],
            self.dataframe.loc[
                (self.dataframe[self.column] >= minimum_median) &
                (self.dataframe[self.column] < median)
            ],
            self.dataframe.loc[
                (self.dataframe[self.column] >= median) &
                (self.dataframe[self.column] < maximum_median)
            ],
            self.dataframe.loc[
                (self.dataframe[self.column] >= maximum_median) &
                (self.dataframe[self.column] <= maximum)
            ]
        ]

        minimum = round(minimum, 2)
        median = round(median, 2)
        maximum = round(maximum, 2)

        minimum_median = round(minimum_median, 2)
        maximum_median = round(maximum_median, 2)

        sizes = [
            f"Min. to Min. Med. ({minimum} - {minimum_median}{self.unit})",
            f"Min. Med. to Med. ({minimum_median} - {median}{self.unit})",
            f"Med. to Med. Max. ({median} - {maximum_median}{self.unit})",
            f"Med. Max. to Max. ({maximum_median} - {maximum}{self.unit})",
        ]

        return (partitions, sizes)
