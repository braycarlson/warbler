from __future__ import annotations

import cv2
import math
import matplotlib.pyplot as plt
import numpy as np

from datatype.plot import StandardSpectrogram
from datatype.settings import Settings
from datatype.signal import Signal
from datatype.spectrogram import compress, create_spectrogram
from io import BytesIO
from PIL import Image as Pillow
from PIL import ImageChops, ImageDraw, ImageFont, ImageOps
from skimage import filters


def create_figure(plot: BytesIO | np.ndarray, text: str):
    """Creates a figure with an image and text.

    Args:
        plot: A BytesIO or numpy array representing the image.
        text: The text to be displayed on the figure.

    Returns:
        The figure image with the specified plot and text.

    """

    if isinstance(plot, BytesIO):
        image = Pillow.open(plot)

    if isinstance(plot, np.ndarray):
        image = Pillow.fromarray(~plot)

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


def create_grid(collection: np.ndarray):
    """Creates a grid of images from a collection.

    Args:
        collection: A numpy array containing images.

    Returns:
        The grid image containing the images from the collection.

    """

    row = len(collection)
    column = 7

    if all(isinstance(image, bytes) for image in collection):
        collection = [
            np.array(create_image(image), dtype='uint8')
            for image in collection
        ]

    shape = [np.shape(image) for image in collection]

    height, width = max(shape)
    width = np.amax(width)
    height = np.amax(height)

    grid = Pillow.new(
        'L',
        color=255,
        size=(
            (column * width),
            (row * height)
        )
    )

    for index, image in enumerate(collection, 0):
        image = create_image(image)
        image = ImageOps.invert(image)

        grid.paste(
            image,
            box=(
                (index % column * width),
                (index // column * height)
            )
        )

    position = (0, 0)

    background = Pillow.new(
        grid.mode,
        grid.size,
        grid.getpixel(position)
    )

    difference = ImageChops.difference(grid, background)
    difference = ImageChops.add(difference, difference, 2.0, -100)
    bbox = difference.getbbox()
    grid = grid.crop(bbox)

    grid = ImageOps.expand(
        grid,
        border=(15, 15),
        fill=(255)
    )

    return grid


def create_image(spectrogram: bytes | np.ndarray):
    """Creates an image from a spectrogram.

    Args:
        spectrogram: A bytes object or numpy array representing the
            spectrogram.

    Returns:
        The image created from the spectrogram.

    """

    if isinstance(spectrogram, np.ndarray):
        condition = (
            np.issubdtype(spectrogram.dtype, np.float32) |
            np.issubdtype(spectrogram.dtype, np.float64)
        )

        if condition:
            spectrogram = compress(spectrogram)

        image = Pillow.fromarray(spectrogram)
        image = ImageOps.invert(image)

    if isinstance(spectrogram, bytes):
        buffer = BytesIO(spectrogram)
        image = Pillow.open(buffer)

    return image
    # return ImageOps.flip(image)


def create_plot(
    signal: Signal,
    settings: Settings,
    matrix: np.ndarray = None
):
    """Creates a plot from a signal and its spectrogram.

    Args:
        signal: The signal object.
        settings: The settings object.
        matrix: Optional numpy array for matrix modification.

    Returns:
        The BytesIO object containing the plot image.

    """

    spectrogram = create_spectrogram(signal, settings, matrix)

    plot = StandardSpectrogram()
    plot.signal = signal
    plot.spectrogram = spectrogram
    plot.create()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()

    return buffer


def create_signal_page(collection: np.ndarray, text: str):
    """Creates a page with a collection of images.

    Args:
        collection: A numpy array containing images.
        text: The text to be displayed on the page.

    Returns:
        The page image with the specified collection of images and text.

    """

    padding = y = 150
    length = len(collection)

    if all(isinstance(image, bytes) for image in collection):
        collection = [
            np.array(create_image(image), dtype='uint8')
            for image in collection
        ]

    shape = [np.shape(image) for image in collection]

    height, width, _ = max(shape)
    width = np.amax(width)
    height = np.amax(height)

    height = (height * length) + (padding * length) + y

    if width < 500:
        width = 500

    grid = Pillow.new(
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


def draw_segment(
    spectrograms: np.ndarray,
    maximum: int = 0,
    zoom: int = 2,
):
    """Draws a segment of spectrograms in a grid.

    Args:
        spectrograms: A numpy array of spectrograms.
        maximum: Maximum number of spectrograms to include in the grid.
        zoom: Zoom factor for the grid.

    Returns:
        A tuple containing the figure and axis objects of the grid.

    """

    cmap = plt.cm.Greys
    # cmap = plt.cm.afmhot

    if maximum == 0:
        n = len(spectrograms)
        sqrt = math.sqrt(n)
        closest = math.ceil(sqrt) ** 2

        maximum = int(math.sqrt(closest))

    rowsize = np.shape(spectrograms[0])[0]
    maximum = maximum * np.shape(spectrograms[0])[1]

    canvas = np.zeros(
        (rowsize * maximum, maximum)
    )

    position = 0
    row = 0

    for _, spectrogram in enumerate(spectrograms):
        shape = np.shape(spectrogram)

        if position + shape[1] > maximum:
            if row == maximum - 1:
                break

            row = row + 1
            position = 0

        canvas[
            rowsize * (maximum - 1 - row):
            rowsize * ((maximum - 1 - row) + 1),
            position: position + shape[1],
        ] = spectrogram

        position = position + shape[1]

    if row < maximum - 1:
        canvas = canvas[(maximum - 1 - row) * rowsize:, :]

    figsize = (
        zoom * (maximum / rowsize),
        zoom * (row + 1)
    )

    fig, ax = plt.subplots(figsize=figsize)

    ax.matshow(
        canvas,
        aspect='auto',
        cmap=cmap,
        origin='lower',
        interpolation='nearest'
    )

    ax.axis('off')

    return (fig, ax)


def filter_image(image: Pillow):
    """Applies various filters to an image.

    Args:
        image: The image to be filtered.

    Returns:
        The filtered image as a numpy array.

    """

    image = to_numpy(image)

    # Denoise
    image = cv2.fastNlMeansDenoising(image, None, 35, 7, 21)

    # Denoise
    image = cv2.fastNlMeansDenoising(image, None, 35, 7, 21)

    # Blur
    image = cv2.medianBlur(image, 1)

    # Contrast control (1.0-3.0)
    alpha = 1.5

    # Brightness control (0-100)
    beta = 0

    image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

    # Sharpen
    image = filters.unsharp_mask(
        image,
        radius=3.5,
        amount=3.5,
        channel_axis=False,
        preserve_range=False
    )

    image = (255 * image).clip(0, 255).astype(np.uint8)

    mask = image > 220
    image[np.where(mask)] = 255

    return image


def resize_image(image: np.ndarray):
    """Resizes an image.

    Args:
        image: The image to be resized as a numpy array.

    Returns:
        The resized image as a bytes object.

    """

    image = Pillow.fromarray(image)

    width, height = image.size

    width = width * 2
    height = height * 2

    upscale = image.resize(
        (width, height),
        resample=Pillow.Resampling.BOX
    )

    buffer = BytesIO()
    upscale.save(buffer, format='png')

    return buffer.getvalue()


def to_numpy(image: Pillow):
    """Converts an image to a numpy array.

    Args:
        image: The image to be converted.

    Returns:
        The image as a numpy array.

    """

    if isinstance(image, bytes):
        buffer = BytesIO(image)
        image = Pillow.open(buffer)

    return np.array(image).astype('uint8')


def to_bytes(image: Pillow):
    """Converts an image to a bytes object.

    Args:
        image: The image to be converted.

    Returns:
        The image as a bytes object.

    """

    image = to_numpy(image)

    if np.mean(image) < 127.5:
        image = np.invert(image)

    image = Pillow.fromarray(image)

    buffer = BytesIO()
    image.save(buffer, format='png')

    return buffer.getvalue()
