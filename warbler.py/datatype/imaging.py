import cv2
import matplotlib.pyplot as plt
import numpy as np

from datatype.plot import StandardSpectrogram
from datatype.spectrogram import compress, create_spectrogram
from io import BytesIO
from PIL import Image as Pillow
from PIL import ImageChops, ImageDraw, ImageFont, ImageOps
from skimage import filters


def create_figure(plot, text):
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


def create_grid(collection):
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


def create_image(spectrogram):
    if isinstance(spectrogram, np.ndarray):
        condition = (
            np.issubdtype(spectrogram.dtype, np.float32) |
            np.issubdtype(spectrogram.dtype, np.float64)
        )

        if condition:
            spectrogram = compress(spectrogram)

        image = Pillow.fromarray(spectrogram)
        # image = ImageOps.invert(image)

    if isinstance(spectrogram, bytes):
        buffer = BytesIO(spectrogram)
        image = Pillow.open(buffer)

    return ImageOps.flip(image)


def create_plot(signal, settings, matrix=None):
    spectrogram = create_spectrogram(signal, settings, matrix)

    plot = StandardSpectrogram()
    plot.signal = signal
    plot.spectrogram = spectrogram
    plot.create()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    plt.close()

    return buffer


def create_signal_page(collection, text):
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


def filter_image(image):
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


def resize_image(image):
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


def to_numpy(image):
    if isinstance(image, bytes):
        buffer = BytesIO(image)
        image = Pillow.open(buffer)

    return np.array(image).astype('uint8')


def to_bytes(image):
    image = to_numpy(image)

    if np.mean(image) < 127.5:
        image = np.invert(image)

    image = Pillow.fromarray(image)

    buffer = BytesIO()
    image.save(buffer, format='png')

    return buffer.getvalue()
