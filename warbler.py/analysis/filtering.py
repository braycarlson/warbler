"""
Filtering
---------

"""

import cv2
import numpy as np
import numpy.typing as npt

from constant import SETTINGS
from datatype.dataset import Dataset
from datatype.imaging import (
    create_image,
    to_bytes,
    to_numpy
)
from datatype.settings import Settings
from datatype.spectrogram import (
    create_spectrogram,
    mel_matrix,
    pad,
    resize
)
from PIL import Image, ImageOps
from skimage import filters


def create_grid(collection: npt.NDArray):
    """Creates a grid of images from a collection.

    Args:
        collection: A numpy array of images.

    Returns:
        An image representing the grid of images.

    """

    column = 5
    padding = 0

    row = len(collection) // column

    if len(collection) % column:
        row = row + 1

    if all(isinstance(image, bytes) for image in collection):
        collection = [
            Image.fromarray(image)
            for image in collection
        ]

    shape = [np.shape(image) for image in collection]

    height, width = max(shape)
    maximum_width = np.amax(width)
    maximum_height = np.amax(height)

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
        image = Image.fromarray(image)

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

        x = x + image.width + padding

        if (index + 1) % column == 0:
            y = y + maximum_height + padding
            x = 0

    grid = ImageOps.expand(
        grid,
        border=(20, 20),
        fill=(255)
    )

    return grid


def filter_image(image: Image):
    """Filters an image by applying various image processing techniques.

    Args:
        image: An image object.

    Returns:
        A filtered image.

    """

    image = to_numpy(image)

    if image.mean() > 127.5:
        image = ~image

    # Blur
    image = cv2.medianBlur(image, 1)

    # Denoise
    image = cv2.fastNlMeansDenoising(image, None, 35, 7, 21)

    # Denoise
    image = cv2.fastNlMeansDenoising(image, None, 35, 7, 21)

    # Sharpen
    image = filters.unsharp_mask(
        ~image,
        radius=3.5,
        amount=3.5,
        channel_axis=False,
        preserve_range=False
    )

    image = (255 * image).clip(0, 255).astype(np.uint8)

    mask = image > 220
    image[np.where(mask)] = 255

    return image


def main() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    filename = 'STE06_1_DbWY2017'

    dataframe = (
        dataframe[dataframe['filename'] == filename]
        .reset_index(drop=True)
    )

    path = SETTINGS.joinpath('spectrogram.json')
    settings = Settings.from_file(path)

    matrix = mel_matrix(settings)

    dataframe['spectrogram'] = (
        dataframe['segment']
        .apply(
            lambda x: create_spectrogram(x, settings, matrix)
        )
    )

    dataframe['scale'] = (
        dataframe['spectrogram']
        .apply(
            lambda x: resize(x)
        )
    )

    padding = dataframe['scale'].apply(
        lambda x: len(x.T)
    ).max()

    dataframe['scale'] = (
        dataframe['scale']
        .apply(
            lambda x: pad(x, padding)
        )
    )

    dataframe['original'] = (
        dataframe['scale']
        .apply(
            lambda x: create_image(x)
        )
    )

    dataframe['filter_array'] = (
        dataframe['original']
        .apply(
            lambda x: filter_image(x)
        )
    )

    dataframe['filter_bytes'] = (
        dataframe['filter_array']
        .apply(
            lambda x: to_bytes(x)
        )
    )

    images = dataframe.filter_array.to_numpy()

    grid = create_grid(images, filename)
    grid.show()


if __name__ == '__main__':
    main()
