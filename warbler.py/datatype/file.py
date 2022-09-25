import lzma
import pickle

from abc import ABC, abstractmethod
from constant import PICKLE


class PicklingStrategy(ABC):
    def __init__(self, path):
        self.filename = None
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


class File():
    def __init__(self, filename):
        self.data = []
        self.filename = filename
        self._path = None
        self._strategy = Uncompressed()

    @property
    def path(self):
        self._path = PICKLE.joinpath(self.filename)
        return self._path

    def extend(self, data):
        self.data.extend(data)

    def insert(self, data):
        self.data.append(data)

    def load(self):
        if not self.path.is_file():
            self.path.touch()

        with lzma.open(self.path, 'rb') as handle:
            self.data = pickle.load(handle)
            return self.data

    def sort(self):
        self.data.sort(
            key=lambda k: k['recording']
        )

    def save(self):
        if not self.path.is_file():
            self.path.touch()

        with lzma.open(self.path, 'wb') as handle:
            pickle.dump(self.data, handle)
