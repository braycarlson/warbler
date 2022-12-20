import fitz
import numpy as np
import warnings

from constant import SETTINGS
# from copy import deepcopy
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
        images = collection[recording]

        padding = y = 150
        length = len(images)

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

        for index, image in enumerate(images):
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


def create_grid(collection, text):
    column = 5
    padding = 10

    row = len(collection) // column

    if len(collection) % column:
        row = row + 1

    width, height = np.vectorize(lambda x: x.size)(collection)
    maximum_width = np.amax(width) + 10
    maximum_height = np.amax(height) + 10

    if maximum_width < 500:
        maximum_width = 500

    page_width = maximum_width * column + (padding * column) - padding
    page_height = maximum_height * row + (padding * row) - padding

    grid = Image.new(
        'L',
        color='white',
        size=(page_width, page_height),
    )

    x = 0
    y = 0

    for index, image in enumerate(collection):
        x_offset = int(
            (maximum_width - image.width) / 2
        )

        y_offset = int(
            (maximum_height - image.height) / 2
        )

        grid.paste(
            image,
            (x + x_offset, y + y_offset)
        )

        x = x + maximum_width + padding

        if (index + 1) % column == 0:
            y = y + maximum_height + padding
            x = 0

    grid = ImageOps.expand(
        grid,
        border=(20, 100),
        fill=(255)
    )

    draw = ImageDraw.Draw(grid)
    font = ImageFont.truetype('fonts/arial.ttf', 32)
    text_width = draw.textlength(text, font=font)

    size = (
        (page_width - text_width) // 2,
        0
    )

    draw.text(
        size,
        text,
        (0),
        font=font
    )

    return grid


def to_string(method):
    return ', '.join(method)


def create_image(spectrogram):
    image = Image.fromarray(~spectrogram)
    flip = ImageOps.flip(image)

    width, height = flip.size

    width = width * 2
    height = height * 2

    return flip.resize(
        (width, height),
        resample=Image.Resampling.BOX
    )


def create_spectrogram(signal, settings):
    spectrogram = Spectrogram()
    strategy = Linear(signal, settings)
    spectrogram.strategy = strategy

    return spectrogram.generate()


def create_collection(iterable):
    collection = {}

    path = SETTINGS.joinpath('spectrogram.json')
    settings = Settings.from_file(path)

    path = SETTINGS.joinpath('dereverberate.json')
    dereverberate = Settings.from_file(path)

    for (identifier, segment, setting) in iterable:
        folder, filename = identifier

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

        file = []

        for method in methods:
            print(f"Applying: {method} to {filename}", flush=True)

            for function in method:
                if function == 'filter':
                    callback[function](segment, setting)
                else:
                    callback[function](segment)

            filtering = to_string(method)

            spectrogram = np.frompyfunc(
                lambda x, y: create_spectrogram(x, y),
                2,
                1
            )(segment, settings)

            image = np.frompyfunc(
                lambda x: create_image(x),
                1,
                1
            )(spectrogram)

            grid = create_grid(image, filtering)
            file.append(grid)

        collection[filename] = file

    return collection


def create_document(individals):
    for individal in individals:
        recordings = individals[individal]

        if not recordings:
            continue

        document = fitz.open()
        toc = []

        for index, recording in enumerate(recordings, 1):
            for filename in recording:
                page = recording[filename]

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
    dataset = Dataset('segment')
    dataframe = dataset.load()

    folder = dataframe.folder.unique()

    tasklist = {
        individual: {}
        for individual in folder
    }

    processes = int(cpu_count() / 4)

    column = ['folder', 'filename']
    groups = dataframe.groupby(column, as_index=False)

    with Pool(processes=processes) as pool:
        for group in groups:
            identifier, subset = group
            folder, filename = identifier

            segment = subset.segment.to_numpy()
            settings = subset.settings.to_numpy()

            iterable = zip(
                [identifier],
                [segment],
                [settings],
            )

            tasklist[folder][filename] = pool.apply_async(
               create_collection,
               args=(iterable,)
            )

        pool.close()
        pool.join()

    pages = {}

    for individal in tasklist:
        pages[individal] = []

        for recording in tasklist[individal]:
            task = tasklist[individal][recording].get(50)
            page = create_page(task)

            pages[individal].append(page)

    create_document(pages)


if __name__ == '__main__':
    main()
