"""
Dataset
-------

"""

from __future__ import annotations

import lzma
import pickle
import shutil

from abc import ABC, abstractmethod
from constant import PICKLE
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd
    import pathlib


class PicklingStrategy(ABC):
    """Abstract base class for pickling strategies."""

    def __init__(self, filename: str = 'dataframe'):
        self.filename = filename
        self.extension = None

    @property
    def path(self) -> pathlib.Path:
        """Get the full path of the pickle file.

        Returns:
            The full path of the pickle file.

        """

        return PICKLE.joinpath(self.filename + self.extension)

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """Load the data from the pickle file."""

        raise NotImplementedError

    @abstractmethod
    def save(self, dataframe: pd.DataFrame) -> None:
        """Save the data to the pickle file.

        Args:
            dataframe: The DataFrame to be saved.

        """

        raise NotImplementedError


class Compressed(PicklingStrategy):
    """A pickling strategy that compresses the data using lzma."""

    def __init__(self, filename: str):
        self.filename = filename
        self.extension = '.xz'

    def load(self) -> pd.DataFrame:
        """Load the data from the compressed pickle file.

        Returns:
            The loaded DataFrame.

        """

        if not self.path.is_file():
            return self.path.touch()

        with lzma.open(self.path, 'rb') as handle:
            return pickle.load(handle)

    def save(self, dataframe: pd.DataFrame) -> None:
        """Save the data to the compressed pickle file.

        Args:
            dataframe: The DataFrame to be saved.

        """

        with lzma.open(self.path, 'wb') as handle:
            pickle.dump(dataframe, handle)


class Uncompressed(PicklingStrategy):
    """A pickling strategy that saves the data without compression."""

    def __init__(self, filename: str):
        self.filename = filename
        self.extension = '.pkl'

    def load(self) -> pd.DataFrame:
        """Load the data from the uncompressed pickle file.

        Returns:
            The loaded DataFrame.

        """

        if not self.path.is_file():
            return self.path.touch()

        with open(self.path, 'rb') as handle:
            return pickle.load(handle)

    def save(self, dataframe: pd.DataFrame) -> None:
        """Save the data to the uncompressed pickle file.

        Args:
            dataframe: The DataFrame to be saved.

        """

        with open(self.path, 'wb') as handle:
            pickle.dump(dataframe, handle)


class Dataset():
    """A dataset class for managing pickled data."""

    def __init__(self, filename: str):
        self._strategy = Uncompressed(filename)

    @property
    def path(self) -> pathlib.Path:
        """Get the path of the dataset.

        Returns:
            The path of the dataset.

        """

        return self.strategy.path

    @property
    def strategy(self) -> PicklingStrategy:
        """Get the pickling strategy.

        Returns:
            The current pickling strategy.

        """

        return self._strategy

    @strategy.setter
    def strategy(self, strategy: PicklingStrategy) -> None:
        self._strategy = strategy

    def duplicate(self, filename: str) -> None:
        """Duplicate the dataset.

        Args:
            filename: The filename of the duplicated dataset.

        """

        path = self.path.parent.joinpath(filename)
        shutil.copyfile(self.path, path)

    def load(self) -> pd.DataFrame:
        """Load the dataset.

        Returns:
            The loaded DataFrame.

        """

        return self.strategy.load()

    def save(self, dataframe: pd.DataFrame) -> None:
        """Save the dataset.

        Args:
            dataframe: The DataFrame to be saved.

        """

        self.strategy.save(dataframe)
