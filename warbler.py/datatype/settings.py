"""
Settings
--------

"""

from __future__ import annotations

import json

from collections.abc import Mapping
from constant import CWD
from types import SimpleNamespace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pathlib

    from typing_extensions import Any, Self


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
    """A class for managing settings with nested attributes."""

    def __init__(self, **data: Any):
        """Initializes a new instance of the Settings class.

        Args:
            **data: Keyword arguments representing the initial settings.

        """

        super().__init__(**data)

    def __getitem__(self, k: Any):
        """Gets the value of a setting by key.

        Args:
            k: The key of the setting.

        Returns:
            The value of the setting.

        """

        if k not in self.__dict__:
            self.__dict__[k] = Settings()

        return self.__dict__[k]

    def __setitem__(self, k: Any, v: Any):
        """Sets the value of a setting by key.

        Args:
            k: The key of the setting.
            v: The value to set for the setting.

        Raises:
            TypeError: If the value is not a valid type.

        """

        if isinstance(v, Mapping):
            v = Settings(**v)
            self.__dict__.setdefault(k, v)
        else:
            self.__dict__[k] = v

    def __repr__(self):
        """Returns a string representation of the Settings object.

        Returns:
            A string representation of the Settings object.

        """

        return repr(self.__dict__)

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
