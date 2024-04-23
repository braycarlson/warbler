from __future__ import annotations

import lzma
import pickle
import shutil

from abc import ABC, abstractmethod
from pandas import HDFStore
from typing import TYPE_CHECKING
from warbler.constant import PICKLE

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
        return PICKLE.joinpath(self.filename + self.extension)

    @abstractmethod
    def load(self) -> pd.DataFrame:
        raise NotImplementedError

    @abstractmethod
    def save(self, dataframe: pd.DataFrame) -> None:
        raise NotImplementedError


class Compressed(DatasetStrategy):
    def __init__(self, filename: str):
        self.filename = filename
        self.extension = '.xz'

    def chunk(self, chunksize: int) -> pd.DataFrame:
        raise NotImplementedError

    def load(self) -> pd.DataFrame:
        if not self.path.is_file():
            return self.path.touch()

        with lzma.open(self.path, 'rb') as handle:
            return pickle.load(handle)

    def save(self, dataframe: pd.DataFrame) -> None:
        if not dataframe.empty:
            with lzma.open(self.path, 'wb') as handle:
                pickle.dump(dataframe, handle)


class Uncompressed(DatasetStrategy):
    def __init__(self, filename: str):
        self.filename = filename
        self.extension = '.pkl'

    def chunk(self, chunksize: int) -> pd.DataFrame:
        raise NotImplementedError

    def load(self) -> pd.DataFrame:
        if not self.path.is_file():
            return self.path.touch()

        with open(self.path, 'rb') as handle:
            return pickle.load(handle)

    def save(self, dataframe: pd.DataFrame) -> None:
        if not dataframe.empty:
            with open(self.path, 'wb') as handle:
                pickle.dump(dataframe, handle)


class HierarchicalDataFormat(DatasetStrategy):
    def __init__(self, filename: str, chunksize: int = 50):
        self.buffer = None
        self.chunksize = chunksize
        self.filename = filename
        self.extension = '.h5'
        self.start = 0

    def load_buffer(self, start: int) -> None:
        with HDFStore(self.filename, mode='r') as store:
            if '/dataframe' in store:
                stop = start + self.chunksize

                self.buffer = store.select(
                    'dataframe',
                    start=start,
                    stop=stop
                )

                self.start = start
            else:
                message = 'No dataframe found.'
                raise KeyError(message)

    def get(self, index: int) -> None:
        if index < self.start or index >= self.start + self.chunksize:
            self.load_buffer(index)

        local = index - self.start
        return self.buffer.iloc[local]

    def chunk(self, chunksize: int) -> pd.DataFrame:
        store = HDFStore(self.path, mode='r')

        try:
            if '/dataframe' in store:
                yield from store.select('dataframe', chunksize=chunksize)
            else:
                message = 'No dataframe found.'
                raise KeyError(message)
        finally:
            store.close()

    def load(self) -> pd.DataFrame:
        with HDFStore(self.path) as store:
            if '/dataframe' in store:
                return store['/dataframe']

            message = 'No dataframe found.'
            raise KeyError(message)

    def save(self, dataframe: pd.DataFrame) -> None:
        with HDFStore(self.path, mode='w') as store:
            if not dataframe.empty:
                store.put(
                    'dataframe',
                    dataframe,
                    format='table'
                )


class Dataset:
    def __init__(self, filename: str):
        self.strategy = HierarchicalDataFormat(filename)

    @property
    def path(self) -> Path:
        return self.strategy.path

    def chunk(self, chunksize: int) -> pd.DataFrame:
        return self.strategy.chunk(chunksize)

    def duplicate(self, filename: str) -> None:
        path = self.path.parent.joinpath(filename)
        shutil.copyfile(self.path, path)

    def load(self) -> pd.DataFrame:
        return self.strategy.load()

    def save(self, dataframe: pd.DataFrame) -> None:
        self.strategy.save(dataframe)
