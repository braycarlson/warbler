"""
Settings
--------

"""

from __future__ import annotations

import json

from constant import CWD
from types import SimpleNamespace
from typing import Any, Self, TYPE_CHECKING

if TYPE_CHECKING:
    import pathlib


def resolve(relative: str) -> Settings:
    """Resolves the path relative to the current working directory (CWD) and
    returns the corresponding Settings object loaded from that path.

    Args:
        relative: The relative path to the settings file.

    Returns:
        Settings: The Settings object loaded from the specified path.

    """

    path = CWD.joinpath(relative)
    return Settings.from_file(path)


class Settings(SimpleNamespace):
    def __init__(self, **data: Any):
        super().__init__(**data)

    @classmethod
    def from_dict(cls: type[Self], data: dict[str, Any]) -> Self:
        """Creates a Settings object from a dictionary.

        Args:
            data: The dictionary containing the settings data.

        Returns:
            Settings: The Settings object created from the dictionary.

        """

        return cls(**data)

    @classmethod
    def from_file(cls: type[Self], path: str | pathlib.Path) -> Self:
        """Creates a Settings object by loading the settings from a file.

        Args:
            path: The path to the settings file.

        Returns:
            Settings: The Settings object created from the file.

        """

        with open(path, 'r') as handle:
            data = json.load(handle)
            return cls(**data)

    @classmethod
    def from_object(cls: type[Self], data: object) -> Self:
        """Creates a Settings object from the __dict__ of an object.

        Args:
            data: The object containing the settings data.

        Returns:
            Settings: The Settings object created from the object.

        """

        return cls(**data.__dict__)

    def update(self, data: dict[str, Any]) -> None:
        """Updates the settings object with new data.

        Args:
            data: The dictionary containing the new data.

        """

        for k in data:
            if k in self.__dict__:
                self.__dict__[k] = data[k]

    def save(self, path: str | pathlib.Path) -> None:
        """Saves the settings object to a file.

        Args:
            path: The path to save the settings file.

        """

        with open(path, 'w+') as file:
            json.dump(
                self.__dict__,
                file,
                indent=4
            )
