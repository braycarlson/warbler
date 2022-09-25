import lzma
import pickle

from abc import ABC, abstractmethod
from constant import PICKLE


class PicklingStrategy(ABC):
    def __init__(self, path):
        self.filename = 'dataframe'
        self.extension = None

    @property
    def path(self):
        return PICKLE.joinpath(self.filename + self.extension)

    @abstractmethod
    def load(self):
        pass

    @abstractmethod
    def save(self, dataframe):
        pass


class Compressed(PicklingStrategy):
    def __init__(self):
        self.filename = 'dataframe'
        self.extension = '.xz'

    def load(self):
        if not self.path.is_file():
            self.path.touch()

        with lzma.open(self.path, 'rb') as handle:
            return pickle.load(handle)

    def save(self, dataframe):
        with lzma.open(self.path, 'wb') as handle:
            pickle.dump(dataframe, handle)


class Uncompressed(PicklingStrategy):
    def __init__(self):
        self.filename = 'dataframe'
        self.extension = '.pkl'

    def load(self):
        if not self.path.is_file():
            self.path.touch()

        with open(self.path, 'rb') as handle:
            return pickle.load(handle)

    def save(self, dataframe):
        with open(self.path, 'wb') as handle:
            pickle.dump(dataframe, handle)


class Dataframe():
    def __init__(self):
        self._strategy = Compressed()

    @property
    def path(self):
        return self.strategy.path

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy):
        self._strategy = strategy

    def load(self):
        return self.strategy.load()

    def save(self, dataframe):
        self.strategy.save(dataframe)
