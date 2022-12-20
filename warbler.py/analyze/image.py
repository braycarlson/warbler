# import librosa
# import librosa.display
# import matplotlib.pyplot as plt
import cv2
import numpy as np
# import scipy.ndimage

from constant import SETTINGS
from copy import deepcopy
from datatype.dataset import Dataset
from datatype.settings import Settings
from datatype.spectrogram import Linear, Mel, Spectrogram
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont, ImageOps
# from scipy import ndimage as ndi
# from skimage import color, data, feature, filters, img_as_float, morphology
# from skimage.data import camera
# from skimage.feature import peak_local_max
# from skimage.morphology import disk, binary_dilation
# from skimage.restoration import inpaint, unwrap_phase
# from skimage.segmentation import chan_vese
# from skimage.util import compare_images, random_noise


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

    # # Maximum
    # maximum = image > 0
    # image[np.where(maximum)] = 255

    # Brightness and contrast
    contrast = 2.5
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

    return image


def create_image(spectrogram):
    image = Image.fromarray(~spectrogram)
    return image


def create_spectrogram(signal, settings):
    spectrogram = Spectrogram()
    strategy = Mel(signal, settings)
    spectrogram.strategy = strategy

    return spectrogram.generate()


def main():
    dataset = Dataset('signal')
    dataframe = dataset.load()

    minimum = 20
    maximum = minimum + 5

    subset = dataframe.iloc[minimum:maximum]

    # path = SETTINGS.joinpath('dereverberate.json')
    # dereverberate = Settings.from_file(path)

    path = SETTINGS.joinpath('spectrogram.json')
    settings = Settings.from_file(path)

    signal = subset.signal.to_numpy()

    np.frompyfunc(
        lambda x: x.normalize(),
        1,
        1
    )(signal)

    # np.frompyfunc(
    #     lambda x: x.dereverberate(dereverberate),
    #     1,
    #     0
    # )(signal)

    spectrogram = np.frompyfunc(
        lambda x, y: create_spectrogram(x, y),
        2,
        1
    )(signal, settings)

    # Original
    original = np.frompyfunc(
        lambda x: create_image(x),
        1,
        1
    )(spectrogram)

    # Processed
    processed = np.frompyfunc(
        lambda x: process_image(x),
        1,
        1
    )(spectrogram)

    image = np.frompyfunc(
        lambda x: create_image(x),
        1,
        1
    )(processed)

    for o, i in zip(original, image):
        collection = [o, i]
        grid = create_grid(collection)

        grid.show()


if __name__ == '__main__':
    main()
