"""
Package
-------

"""

from __future__ import annotations

from constant import DATASET, OUTPUT
from zipfile import ZipFile


def main() -> None:
    files = [
        file
        for file in DATASET.rglob('*.json')
        if file.is_file()
    ]

    path = OUTPUT.joinpath('settings.zip')

    with ZipFile(path, 'w') as handle:
        for file in files:
            individual = file.parent.parent.relative_to(DATASET)
            arcname = individual.joinpath(file.name)

            handle.write(file, arcname=arcname)


if __name__ == '__main__':
    main()
