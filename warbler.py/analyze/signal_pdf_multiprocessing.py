import fitz
import matplotlib.pyplot as plt
import numpy as np
import warnings

from constant import SETTINGS
# from copy import deepcopy
from datatype.axes import SpectrogramAxes
from datatype.dataset import Dataset
from datatype.settings import Settings
from datatype.spectrogram import (
    Linear,
    Mel,
    Spectrogram
)
from io import BytesIO
from itertools import permutations
from multiprocessing import cpu_count, Pool
from PIL import Image, ImageDraw, ImageFont, ImageOps
from plot import LusciniaSpectrogram


warnings.simplefilter(
    action='ignore',
    category=FutureWarning
)

warnings.simplefilter(
    action='ignore',
    category=UserWarning
)

warnings.simplefilter(
    action='ignore',
    category=np.VisibleDeprecationWarning
)

# warnings.filterwarnings('error')


def create_page(collection):
    pages = {}

    for recording in collection:
        individal = collection[recording]

        images = [
            item.get('image')
            for item in individal
        ]

        padding = y = 150
        length = len(individal)

        width, height = np.vectorize(lambda x: x.size)(images)

        width = np.amax(width)
        height = np.amax(height)

        height = (height * length) + (padding * length) + y

        if width < 500:
            width = 500

        grid = Image.new(
            'L',
            color='white',
            size=(width, height),
        )

        for index, information in enumerate(individal):
            image = information.get('image')

            x = int(
                (width - image.width) / 2
            )

            grid.paste(
                image,
                (x, y)
            )

            y = y + image.height + padding

        grid = ImageOps.expand(
            grid,
            border=(50, 288),
            fill=(255)
        )

        draw = ImageDraw.Draw(grid)
        font = ImageFont.truetype('fonts/times.ttf', 72)
        text_width = draw.textlength(recording, font=font)

        size = (
            (width + 50 - text_width) // 2,
            50
        )

        draw.text(
            size,
            recording,
            (0),
            font=font
        )

        pages[recording] = grid

    return pages


def create_plot(signal, settings):
    spectrogram = Spectrogram()
    strategy = Linear(signal, settings)
    spectrogram.strategy = strategy

    spectrogram = spectrogram.generate()

    plot = LusciniaSpectrogram(signal, spectrogram)
    plot.create()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()

    return buffer


def create_image(spectrogram, text):
    image = Image.open(spectrogram)

    image = ImageOps.expand(
        image,
        border=(100, 100),
        fill=(255, 255, 255, 255)
    )

    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype('fonts/arial.ttf', 32)
    text_width = draw.textlength(text, font=font)

    size = (
        (image.width - text_width) // 2,
        10
    )

    draw.text(
        size,
        text,
        (0),
        font=font
    )

    return image


def create_collection(iterable):
    collection = {}

    path = SETTINGS.joinpath('spectrogram.json')
    settings = Settings.from_file(path)

    path = SETTINGS.joinpath('dereverberate.json')
    dereverberate = Settings.from_file(path)

    for index, (filename, signal, setting) in enumerate(iterable, 1):
        file = []

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

            # 'reduce': np.frompyfunc(
            #     lambda x: x.reduce(),
            #     1,
            #     0
            # )
        }

        methods = permutations([*callback], 3)

        spectrogram = np.frompyfunc(
            lambda x, y: create_plot(x, y),
            2,
            1
        )(signal, settings)

        image = np.frompyfunc(
            lambda x, y: create_image(x, y),
            2,
            1
        )(spectrogram, 'original')

        item = {
            'image': image,
            'index': index
        }

        file.append(item)

        for method in methods:
            print(f"Applying: {method} to {filename}", flush=True)

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
                lambda x, y: create_image(x, y),
                2,
                1
            )(spectrogram, filtering)

            item = {
                'image': image,
                'index': index
            }

            file.append(item)

        collection[filename] = file

    return collection


def to_string(method):
    return ', '.join(method)


def create_document(individal, tasklist):
    document = fitz.open()
    toc = []

    task = tasklist.get(50)
    pages = create_page(task)

    for index, filename in enumerate(pages, 1):
        page = pages[filename]

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

    document.set_toc(toc)

    document.save(
        individal + '.pdf',
        deflate=True
    )

    document.close()


def main():
    dataset = Dataset('signal')
    dataframe = dataset.load()

    folders = dataframe.folder.unique()

    tasklist = {}

    processes = int(cpu_count() / 4)

    with Pool(processes=processes) as pool:
        for folder in folders:
            subset = dataframe[dataframe.folder == folder]
            individal = subset.folder.iat[0]

            signal = subset.signal.to_numpy()
            settings = subset.settings.to_numpy()
            filename = subset.filename.to_numpy()

            iterable = zip(filename, signal, settings)

            tasklist[individal] = pool.apply_async(
                create_collection,
                args=(iterable,)
            )

        pool.close()
        pool.join()

    with Pool(processes=processes) as pool:
        for individal in tasklist:
            create_document(
                individal,
                tasklist[individal]
            )

        pool.close()
        pool.join()


if __name__ == '__main__':
    main()
