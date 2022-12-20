import cv2
import fitz
import matplotlib.pyplot as plt
import numpy as np
import warnings

from constant import SETTINGS
from copy import deepcopy
from datatype.axes import SpectrogramAxes
from datatype.dataset import Dataset
from datatype.settings import Settings
from datatype.spectrogram import (
    Linear,
    # Mel,
    Spectrogram
)
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageOps
from plot import LusciniaSpectrogram
from skimage import filters


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


def process_image(image):
    # Denoise
    image = cv2.medianBlur(image, 3)
    image = cv2.fastNlMeansDenoising(image, None, 50, 7, 21)

    # Minimum
    minimum = image < 25
    image[np.where(minimum)] = 0

    # Brightness and contrast
    contrast = 1.5
    brightness = 0

    brightness = brightness + int(
        round(255 * (1 - contrast) / 2)
    )

    image = cv2.addWeighted(
        image,
        contrast,
        image,
        0,
        brightness
    )

    # Sharpen
    image = filters.unsharp_mask(
        image,
        radius=3.5,
        amount=3.5,
        channel_axis=False,
        preserve_range=False
    )

    image = (255 * image).clip(0, 255).astype(np.uint8)

    return image


def create_image(spectrogram, text):
    if isinstance(spectrogram, BytesIO):
        image = Image.open(spectrogram)

    if isinstance(spectrogram, np.ndarray):
        image = Image.fromarray(~spectrogram)

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
        ip = deepcopy(signal)

        drv.dereverberate(dereverberate)

        collection = []

        spectrogram = create_plot(original, settings)
        image = create_image(spectrogram, 'original')
        collection.append(image)

        spectrogram = create_plot(drv, settings)
        image = create_image(spectrogram, 'bandpass filter, normalize, dereverberate')
        collection.append(image)

        spectrogram = Spectrogram()
        strategy = Linear(drv, settings)
        spectrogram.strategy = strategy

        spectrogram = spectrogram.generate()

        image = Image.fromarray(spectrogram)
        image = process_image(spectrogram)

        plot = LusciniaSpectrogram(drv, image)
        plot.create()

        buffer = BytesIO()
        plt.savefig(buffer, format='png')
        plt.close()

        image = create_image(buffer, 'filtered')

        collection.append(image)

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
        create_document(subset)


if __name__ == '__main__':
    main()
