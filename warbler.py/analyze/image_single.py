import cv2
import matplotlib.pyplot as plt
import numpy as np

from constant import SETTINGS
from datatype.axes import SpectrogramAxes
from datatype.dataset import Dataset
from datatype.settings import Settings
from datatype.spectrogram import Linear, Mel, Spectrogram
from PIL import Image
from plot import LusciniaSpectrogram
from skimage import filters


def create_grid(collection):
    row = len(collection)
    column = 1

    width, height = np.vectorize(lambda x: x.size)(collection)
    width = np.amax(width)
    height = np.amax(height)

    offset = 20

    grid = Image.new(
        'L',
        color='white',
        size=(
            (column * width) + offset,
            (row * height) + (offset * 2)
        )
    )

    for index, image in enumerate(collection, 0):
        grid.paste(
            image,
            box=(
                (index % column * width) + offset,
                (index // column * height) + (offset * 2)
            )
        )

    return grid


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


def create_image(spectrogram):
    image = Image.fromarray(~spectrogram)
    return image


def create_spectrogram(signal, settings):
    spectrogram = Spectrogram()
    strategy = Linear(signal, settings)
    spectrogram.strategy = strategy

    return spectrogram.generate()


def main():
    dataset = Dataset('signal')
    dataframe = dataset.load()

    signal = dataframe.at[10, 'signal']
    signal.normalize()

    path = SETTINGS.joinpath('spectrogram.json')
    settings = Settings.from_file(path)

    spectrogram = create_spectrogram(signal, settings)

    # Original
    original = create_image(spectrogram)
    original = np.invert(original)

    # Processed
    processed = process_image(spectrogram)

    image = create_image(processed)
    image = np.invert(image)

    plot = LusciniaSpectrogram(signal, original)
    plot.create()

    plt.show()
    plt.close()

    plot = LusciniaSpectrogram(signal, image)
    plot.create()

    plt.show()
    plt.close()

    # collection = [original, image]
    # grid = create_grid(collection)

    # grid.show()


if __name__ == '__main__':
    main()
