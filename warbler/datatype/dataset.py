from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import shutil

from abc import ABC, abstractmethod
from typing_extensions import TYPE_CHECKING
from warbler.constant import PARQUET
from warbler.datatype.settings import Settings
from warbler.datatype.signal import Signal

if TYPE_CHECKING:
    import pandas as pd

    from pathlib import Path


class DatasetStrategy(ABC):
    def __init__(self, filename: str = 'dataframe'):
        self.filename = filename
        self.extension = None

    @abstractmethod
    def chunk(self, chunksize: int) -> pd.DataFrame:
        raise NotImplementedError

    @property
    def path(self) -> Path:
        return PARQUET.joinpath(self.filename + self.extension)

    @abstractmethod
    def load(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def save(self, dataframe: pd.DataFrame) -> None:
        raise NotImplementedError


class ParquetStrategy(DatasetStrategy):
    def __init__(self, filename: str, chunksize: int = 50):
        self.columns = [
            'scale',
            'spectrogram',
            'original_array',
            'filter_array',
        ]
        self.chunksize = chunksize
        self.filename = filename
        self.extension = '.parquet'
        self.start = 0
        self.buffer = None

    def _serialize(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        for column in ['segment', 'signal']:
            if column in dataframe.columns:
                dataframe[column] = dataframe[column].apply(
                    lambda x: x.serialize()
                )

        for column in self.columns:
            if column in dataframe.columns:
                dataframe[column + '_shape'] = dataframe[column].apply(
                    lambda x: x.shape
                )

                dataframe[column] = dataframe[column].apply(
                    lambda x: x.ravel().tolist()
                )

        return dataframe

    def _deserialize(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        for column in ['segment', 'signal']:
            if column in dataframe.columns:
                dataframe[column] = dataframe[column].apply(Signal.deserialize)

        for column in self.columns:
            if column in dataframe.columns:
                shape = column + '_shape'

                if shape in dataframe.columns:
                    dataframe[column] = dataframe.apply(
                        lambda row,
                        column=column,
                        shape=shape: np.array(row[column]).reshape(row[shape]),
                        axis=1
                    )

        return dataframe

    def chunk(self) -> pd.DataFrame:
        file = pq.ParquetFile(self.path)
        length = file.num_row_groups

        for index in range(length):
            yield file.read_row_group(index).to_pandas()

    def get(self, index: int) -> pd.Series:
        if index < self.start or index >= self.start + self.chunksize:
            self.load_buffer(index)

        local = index - self.start
        return self.buffer.iloc[local]

    def get_column(self, name: str) -> list[str]:
        table = pq.read_table(self.path, columns=[name])
        dataframe = table.to_pandas()
        return dataframe[name].tolist()

    def load(self) -> pd.DataFrame:
        dataframe = pq.read_table(self.path).to_pandas()
        return self._deserialize(dataframe)

    def load_buffer(self, index: int) -> None:
        file = pq.ParquetFile(self.path)

        length = file.num_row_groups
        start = index // self.chunksize
        stop = (index + self.chunksize) // self.chunksize

        if start < length:
            partition = range(
                start,
                min(stop, length)
            )

            buffer = file.read_row_groups(partition).to_pandas()
            self.buffer = self._deserialize(buffer)

            self.buffer['settings'] = (
                self.buffer['segmentation']
                .apply(Settings.resolve)
            )

            self.start = start * self.chunksize
        else:
            message = 'Start index is out of bounds.'
            raise IndexError(message)

    def save(self, dataframe: pd.DataFrame) -> None:
        if 'settings' in dataframe.columns:
            drop = ['settings']
            dataframe = dataframe.drop(drop, axis=1)

        dataframe = self._serialize(dataframe)
        table = pa.Table.from_pandas(dataframe)

        pq.write_table(
            table,
            self.path,
            row_group_size=self.chunksize
        )


class Dataset:
    def __init__(self, filename: str):
        self.strategy = ParquetStrategy(filename)

    @property
    def path(self) -> Path:
        return self.strategy.path

    def chunk(self, chunksize: int) -> pd.DataFrame:
        return self.strategy.chunk(chunksize)

    def get(self, index: int) -> pd.Series:
        return self.strategy.get(index)

    def duplicate(self, filename: str) -> None:
        path = self.path.parent.joinpath(filename)
        shutil.copyfile(self.path, path)

    def get_column(self, name: str) -> list[str]:
        return self.strategy.get_column(name)

    def load(self) -> pd.DataFrame:
        return self.strategy.load()

    def load_buffer(self, index: int) -> None:
        return self.strategy.load_buffer(index)

    def save(self, dataframe: pd.DataFrame) -> None:
        self.strategy.save(dataframe)
