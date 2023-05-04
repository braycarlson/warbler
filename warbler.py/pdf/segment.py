import fitz
import numpy as np

from constant import PDF, SETTINGS
from copy import deepcopy
from datatype.dataset import Dataset
from datatype.imaging import create_image
from datatype.settings import Settings
from datatype.spectrogram import create_spectrogram
from io import BytesIO
from itertools import permutations
from PIL import Image, ImageDraw, ImageFont, ImageOps


def create_page(collection, text):
    padding = y = 150
    length = len(collection)

    width, height = np.vectorize(lambda x: x.size)(collection)

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

    for index, image in enumerate(collection):
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
    text_width = draw.textlength(text, font=font)

    size = (
        (width + 50 - text_width) // 2,
        50
    )

    draw.text(
        size,
        text,
        (0),
        font=font
    )

    return grid


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


def resize_image(spectrogram):
    image = create_image(spectrogram)
    flip = ImageOps.flip(image)

    width, height = flip.size

    width = width * 2
    height = height * 2

    return flip.resize(
        (width, height),
        resample=Image.Resampling.BOX
    )


def create_document(subset):
    individal = subset.folder.iat[0]
    group = subset.groupby('filename', as_index=False)

    document = fitz.open()
    toc = []

    path = SETTINGS.joinpath('spectrogram.json')
    settings = Settings.from_file(path)

    path = SETTINGS.joinpath('dereverberate.json')
    dereverberate = Settings.from_file(path)

    print(f"Processing: {individal}")

    for index, (filename, column) in enumerate(group, 1):
        setting = column.settings.to_numpy()

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

        collection = []

        original = column.segment.to_numpy()

        for method in methods:
            print(f"Applying: {method} to {filename}")

            segment = deepcopy(original)
            filtering = to_string(method)

            for function in method:
                if function == 'filter':
                    callback[function](segment, setting)
                else:
                    callback[function](segment)

            spectrogram = np.frompyfunc(
                lambda x, y: create_spectrogram(x, y),
                2,
                1
            )(segment, settings)

            images = np.frompyfunc(
                lambda x: resize_image(x),
                1,
                1
            )(spectrogram)

            grid = create_grid(images, filtering)
            collection.append(grid)

        page = create_page(collection, filename)

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

    filename = f"{individal}_segment.pdf"
    path = PDF.joinpath(filename)

    document.set_toc(toc)
    document.save(path, deflate=True)
    document.close()


def main():
    dataset = Dataset('segment')
    dataframe = dataset.load()

    folders = dataframe.folder.unique()

    for folder in folders:
        subset = dataframe[dataframe.folder == folder]
        subset.reset_index(inplace=True)
        subset = subset.copy()

        create_document(subset)


if __name__ == '__main__':
    main()
