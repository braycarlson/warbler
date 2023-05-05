import fitz
import matplotlib.pyplot as plt

from constant import SETTINGS
from copy import deepcopy
from datatype.dataset import Dataset
from datatype.imaging import (
    create_figure,
    create_image,
    create_plot,
    create_signal_page,
    filter_image
)
from datatype.plot import StandardSpectrogram
from datatype.settings import Settings
from datatype.spectrogram import create_spectrogram
from io import BytesIO


def to_string(method):
    return ', '.join(method)


def create_document(subset):
    individal = subset.folder.iat[0]

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
        print(f"Processing {filename}")

        original = deepcopy(signal)

        signal.filter(
            setting.butter_lowcut,
            setting.butter_highcut
        )

        signal.normalize()

        drv = deepcopy(signal)
        drv.dereverberate(dereverberate)

        collection = []

        spectrogram = create_plot(original, settings)
        image = create_figure(spectrogram, 'original')
        collection.append(image)

        spectrogram = create_plot(drv, settings)

        image = create_figure(
            spectrogram,
            'bandpass filter, normalize, dereverberate'
        )

        collection.append(image)

        spectrogram = create_spectrogram(drv, settings)

        plot = StandardSpectrogram()
        plot.signal = drv
        plot.spectrogram = spectrogram
        plot.create()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()

        image = create_figure(buffer, 'filtered')

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

    filename = individal + '.pdf'

    document.set_toc(toc)
    document.save(filename, deflate=True)
    document.close()


def main():
    dataset = Dataset('signal')
    dataframe = dataset.load()

    folders = dataframe.folder.unique()

    for folder in folders:
        subset = dataframe[dataframe.folder == folder]
        subset.reset_index(inplace=True)
        subset = subset.copy()

        create_document(subset)


if __name__ == '__main__':
    main()
