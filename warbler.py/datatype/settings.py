import json

from types import SimpleNamespace


class Settings(SimpleNamespace):
    def __init__(self, **data):
        super().__init__(**data)

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

    @classmethod
    def from_file(cls, path):
        with open(path, 'r') as handle:
            data = json.load(handle)
            return cls(**data)

    @classmethod
    def from_object(cls, data):
        return cls(**data.__dict__)

    def update(self, data):
        for k in data:
            if k in self.__dict__:
                self.__dict__[k] = data[k]

    def save(self, path):
        with open(path, 'w+') as file:
            json.dump(
                self.__dict__,
                file,
                indent=4
            )
