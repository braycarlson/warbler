import cv2
import logging
import numpy as np
import swifter

from bootstrap import bootstrap
from constant import SETTINGS
from datatype.dataset import Dataset
from datatype.settings import Settings
from datatype.spectrogram import (
    pad,
    Linear,
    Mel,
    Spectrogram
)
from io import BytesIO
from logger import logger
from PIL import Image, ImageOps
from skimage import filters


log = logging.getLogger(__name__)


def create_spectrogram(segment, settings):
    spectrogram = Spectrogram()
    spectrogram.strategy = Mel(segment, settings)
    return spectrogram.generate()


def preprocess_image(image):
    image = ~np.array(image).astype('uint8')

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

    image = cv2.medianBlur(image, 1)

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
    buffer = BytesIO()
    image = Image.fromarray(~spectrogram)

    flip = ImageOps.flip(image)
    flip.save(buffer, format='png')

    return buffer.getvalue()


def crop_image(image):
    wrapper = BytesIO(image)
    image = Image.open(wrapper)

    x = image.width

    area = (x / 3.5, 100, x / 3.5, 0)
    crop = ImageOps.crop(image, area)

    buffer = BytesIO()
    crop.save(buffer, format='png')

    return buffer.getvalue()


def resize_spectrogram(spectrogram, factor=10):
    x, y = np.shape(spectrogram)
    shape = [int(np.log(y) * factor), x]

    return np.array(
        Image
        .fromarray(spectrogram)
        .resize(shape, Image.Resampling.LANCZOS)
    )


def resize_image(image):
    # wrapper = BytesIO(image)
    image = Image.fromarray(image)

    width, height = image.size

    width = width * 2
    height = height * 2

    upscale = image.resize(
        (width, height),
        resample=Image.Resampling.BOX
    )

    buffer = BytesIO()
    upscale.save(buffer, format='png')

    return buffer.getvalue()


def to_numpy(image):
    buffer = BytesIO(image)
    image = Image.open(buffer)

    return ~np.array(image).astype('uint8')


@bootstrap
def main():
    swifter.set_defaults(
        npartitions=None,
        dask_threshold=1,
        scheduler='processes',
        progress_bar=False,
        progress_bar_desc=False,
        allow_dask_on_strings=True,
        force_parallel=True,
    )

    dataset = Dataset('segment')
    dataframe = dataset.load()

    dataframe = dataframe[~dataframe.exclude]

    dataframe['duration'] = (
        dataframe['segment']
        .swifter
        .apply(lambda x: x.duration)
    )

    # Use "duration.ipynb" to determine the best mask
    mask = (
        (dataframe.duration < 0.0250) |
        (dataframe.duration > 0.1700)
    )

    dataframe = dataframe[~mask]
    dataframe.reset_index(inplace=True)
    dataframe = dataframe.copy()

    path = SETTINGS.joinpath('spectrogram.json')
    settings = Settings.from_file(path)

    dataframe['spectrogram'] = (
        dataframe['segment']
        .swifter
        .apply(
            lambda x: create_spectrogram(x, settings)
        )
    )

    dataframe['spectrogram'] = (
        dataframe['spectrogram']
        .swifter
        .apply(
            lambda x: resize_spectrogram(x)
        )
    )

    spectrogram = dataframe.spectrogram.to_numpy()
    length = [np.shape(i)[1] for i in spectrogram]
    padding = np.max(length)

    dataframe['spectrogram'] = (
        dataframe['spectrogram']
        .swifter
        .apply(
            lambda x: pad(x, padding)
        )
    )

    dataframe['image'] = (
        dataframe['spectrogram']
        .swifter
        .apply(
            lambda x: create_image(x)
        )
    )

    dataframe['denoise'] = (
        dataframe['spectrogram']
        .swifter
        .apply(
            lambda x: preprocess_image(x)
        )
    )

    dataframe['resize'] = (
        dataframe['denoise']
        .swifter
        .apply(
            lambda x: resize_image(x)
        )
    )

    # dataframe['image'] = (
    #     dataframe['image']
    #     .swifter
    #     .apply(
    #         lambda x: to_numpy(x)
    #     )
    # )

    # dataframe['denoise'] = (
    #     dataframe['denoise']
    #     .swifter
    #     .apply(
    #         lambda x: to_numpy(x)
    #     )
    # )

    dataset.save(dataframe)


if __name__ == '__main__':
    with logger():
        main()
