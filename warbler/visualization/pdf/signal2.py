from __future__ import annotations

import fitz
import numpy as np

from copy import deepcopy
from io import BytesIO
from itertools import permutations
from typing_extensions import TYPE_CHECKING
from warbler.constant import PDF, SETTINGS
from warbler.datatype.dataset import Dataset
from warbler.datatype.imaging import (
    create_figure,
    create_plot,
    create_signal_page
)
from warbler.datatype.settings import Settings

if TYPE_CHECKING:
    import pandas as pd


def to_string(method: list[str]) -> str:
    return ', '.join(method)


def create_document(subset: pd.DataFrame) -> None:
    individal = subset.folder.iloc[0]

    signal = subset.signal.to_numpy()
    setting = subset.settings.to_numpy()
    filename = subset.filename.to_numpy()

    iterable = zip(filename, signal, setting)

    path = SETTINGS.joinpath('spectrogram.json')
    settings = Settings.from_file(path)

    path = SETTINGS.joinpath('dereverberate.json')
    dereverberate = Settings.from_file(path)

    document = fitz.open()
    toc = []

    for index, (filename, signal, setting) in enumerate(iterable, 1):
        original = deepcopy(signal)

        callback = {
            'dereverberate': np.frompyfunc(
                lambda x: x.dereverberate(dereverberate),
                1,
                0
            ),

            'filter': np.frompyfunc(
                lambda x, y: x.filter(
                    y.butter_lowcut,
                    y.butter_highcut
                ),
                2,
                0
            ),

            'normalize': np.frompyfunc(
                lambda x: x.normalize(),
                1,
                0
            ),
        }

        methods = permutations([*callback], 3)

        collection = []

        spectrogram = np.frompyfunc(
            lambda x, y: create_plot(x, y),
            2,
            1
        )(signal, settings)

        image = np.frompyfunc(
            lambda x, y: create_figure(x, y),
            2,
            1
        )(spectrogram, 'original')

        collection.append(image)

        for method in methods:
            print(f"Applying: {method} to {filename}")

            signal = deepcopy(original)

            for function in method:
                if function == 'filter':
                    callback[function](signal, setting)
                else:
                    callback[function](signal)

            filtering = to_string(method)

            spectrogram = np.frompyfunc(
                lambda x, y: create_plot(x, y),
                2,
                1
            )(signal, settings)

            image = np.frompyfunc(
                lambda x, y: create_figure(x, y),
                2,
                1
            )(spectrogram, filtering)

            collection.append(image)

        page = create_signal_page(collection, filename)

        buffer = BytesIO()
        page.save(buffer, format='png')

        stream = buffer.getvalue()

        # Page
        current = document.new_page(
            width=page.width,
            height=page.height
        )

        current.insert_image(
            fitz.Rect(
                20,
                0,
                page.width,
                page.height
            ),
            stream=stream,
            keep_proportion=True
        )

        # Table of Contents
        item = [1, filename, index]
        toc.append(item)

    filename = f"{individal}_signal.pdf"
    path = PDF.joinpath(filename)

    document.set_toc(toc)
    document.save(path, deflate=True)
    document.close()


def main() -> None:
    dataset = Dataset('signal')
    dataframe = dataset.load()

    folders = dataframe.folder.unique()

    for folder in folders:
        subset = dataframe[dataframe.folder == folder]
        subset = subset.reset_index()
        subset = subset.copy()

        create_document(subset)


if __name__ == '__main__':
    main()
