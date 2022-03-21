import json

from types import SimpleNamespace


class Parameters(SimpleNamespace):
    def __init__(self, path):
        self.path = str(path)
        self._load()

    def _load(self):
        with open(self.path, 'r') as handle:
            data = json.load(handle)
            super().__init__(**data)

    def save(self):
        path = self.__dict__.pop('path')

        with open(path, 'w+') as file:
            json.dump(
                self.__dict__,
                file,
                indent=4
            )

        setattr(self, 'path', path)
        self._load()
