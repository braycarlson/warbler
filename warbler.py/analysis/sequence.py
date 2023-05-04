import cv2
import matplotlib.pyplot as plt
import numpy as np

from constant import CWD, OUTPUT, PICKLE, SETTINGS
from datatype.axes import SpectrogramAxes
from datatype.dataset import Dataset
from datatype.imaging import (
    create_image,
    to_bytes,
    to_numpy
)
from datatype.segmentation import dynamic_threshold_segmentation
from datatype.settings import Settings
from datatype.signal import Signal
from datatype.spectrogram import (
    create_spectrogram,
    Linear,
    mel_matrix,
    Spectrogram
)
from PIL import Image as Pillow
from PIL import ImageDraw, ImageFont, ImageOps
from plot import (
    BandwidthSpectrogram,
    LusciniaSpectrogram,
    SegmentationSpectrogram,
)
from skimage import filters


def filter_image(image):
    image = to_numpy(image)

    if image.mean() > 127.5:
        image = ~image

    # Blur
    image = cv2.medianBlur(image, 1)

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

    image = cv2.fastNlMeansDenoising(image, None, 25, 7, 21)

    image = cv2.medianBlur(image, 3)

    return image


def create_grid(collection, text):
    column = row = 5

    if all(isinstance(image, bytes) for image in collection):
        collection = [
            np.array(create_image(image), dtype='uint8')
            for image in collection
        ]

    shape = [np.shape(image) for image in collection]

    height, width = max(shape)
    width = np.amax(width)
    height = np.amax(height)

    offset = 20

    grid = Pillow.new(
        'L',
        color=255,
        size=(
            (column * width) + offset,
            (row * height) + offset
        )
    )

    for index, image in enumerate(collection, 0):
        grid.paste(
            create_image(image),
            box=(
                (index % column * width) + offset,
                (index // column * height) + (offset * 2)
            )
        )

    grid = ImageOps.expand(
        grid,
        border=(50, 50),
        fill=(255)
    )

    draw = ImageDraw.Draw(grid)
    font = ImageFont.truetype('fonts/times.ttf', 32)
    text_width = draw.textlength(text, font=font)

    size = (
        (width // 2) + text_width + offset + 50,
        10
    )

    draw.text(
        size,
        text,
        (0),
        font=font
    )

    return grid


def main():
    projection = OUTPUT.joinpath('projection')
    projection.mkdir(parents=True, exist_ok=True)

    dataset = Dataset('signal')
    dataframe = dataset.load()

    projection = OUTPUT.joinpath('projection')
    filename = 'STE01_DbWY2017'

    recording = (
        dataframe
        .loc[dataframe.filename == filename, 'recording']
        .squeeze()
    )

    segmentation = (
        dataframe
        .loc[dataframe.filename == filename, 'segmentation']
        .squeeze()
    )

    onsets = (
        dataframe
        .loc[dataframe.filename == filename, 'onset']
        .squeeze()
    )

    offsets = (
        dataframe
        .loc[dataframe.filename == filename, 'offset']
        .squeeze()
    )

    path = CWD.joinpath(recording)
    signal = Signal(path)

    # path = SETTINGS.joinpath('dereverberate.json')
    # dereverberate = Settings.from_file(path)

    path = CWD.joinpath(segmentation)
    custom = Settings.from_file(path)

    # Original signal
    spectrogram = create_spectrogram(signal, custom)
    image = create_image(spectrogram)

    plot = LusciniaSpectrogram(signal, image)
    plot.create()

    file = projection.joinpath('01_original.png')

    plt.title(
        'Original',
        fontweight=600,
        fontsize=12,
        pad=25
    )

    plt.savefig(
        file,
        bbox_inches='tight',
        dpi=300,
        format='png'
    )

    # Bandwidth spectrogram
    spectrogram = Spectrogram()
    strategy = Linear(signal, custom)
    spectrogram.strategy = strategy

    spectrogram = spectrogram.generate(normalize=False)

    image = create_image(spectrogram)
    image = ImageOps.invert(image)

    plot = BandwidthSpectrogram(signal, image, custom)
    plot.create()

    file = projection.joinpath('02_pre_bandpass_filter.png')

    plt.title(
        'Pre-Bandpass Filter',
        fontweight=600,
        fontsize=12,
        pad=25
    )

    plt.savefig(
        file,
        bbox_inches='tight',
        dpi=300,
        format='png'
    )

    # Bandpass filter
    signal.filter(
        custom.butter_lowcut,
        custom.butter_highcut
    )

    spectrogram = create_spectrogram(signal, custom)
    image = create_image(spectrogram)

    plot = LusciniaSpectrogram(signal, image)
    plot.create()

    file = projection.joinpath('03_bandpass_filter.png')

    plt.title(
        'Bandpass Filter',
        fontweight=600,
        fontsize=12,
        pad=25
    )

    plt.savefig(
        file,
        bbox_inches='tight',
        dpi=300,
        format='png'
    )

    # Normalize
    signal.normalize()

    spectrogram = create_spectrogram(signal, custom)
    image = create_image(spectrogram)

    plot = LusciniaSpectrogram(signal, image)
    plot.create()

    file = projection.joinpath('04_normalize.png')

    plt.title(
        'Normalize',
        fontweight=600,
        fontsize=12,
        pad=25
    )

    plt.savefig(
        file,
        bbox_inches='tight',
        dpi=300,
        format='png'
    )

    # # Dereverberate
    # path = SETTINGS.joinpath('dereverberate.json')
    # dereverberate = Settings.from_file(path)

    # signal.dereverberate(dereverberate)

    # spectrogram = create_spectrogram(signal, custom)
    # image = create_image(spectrogram)

    # plot = LusciniaSpectrogram(signal, image)
    # plot.create()

    # file = projection.joinpath('05_dereverberate.png')

    # plt.title(
    #     'Dereverberate',
    #     fontweight=600,
    #     fontsize=12,
    #     pad=25
    # )

    # plt.savefig(
    #     file,
    #     bbox_inches='tight',
    #     dpi=300,
    #     format='png'
    # )

    # Segmentation
    dts = dynamic_threshold_segmentation(
        signal,
        custom,
        full=True
    )

    spectrogram = create_spectrogram(signal, custom)
    image = create_image(spectrogram)

    dts['spectrogram'] = image

    plot = SegmentationSpectrogram(signal, dts, custom)
    plot.create()

    file = projection.joinpath('06_segmentation.png')

    plt.title(
        'Segmentation',
        fontweight=600,
        fontsize=12,
        pad=25
    )

    plt.savefig(
        file,
        bbox_inches='tight',
        dpi=300,
        format='png'
    )

    # # Image filter
    # image = filter_image(image)

    # plot = LusciniaSpectrogram(signal, ~image)
    # plot.create()

    # file = projection.joinpath('07_image_filter.png')

    # plt.title(
    #     'Image Filter',
    #     fontweight=600,
    #     fontsize=12,
    #     pad=25
    # )

    # plt.savefig(
    #     file,
    #     bbox_inches='tight',
    #     dpi=300,
    #     format='png'
    # )

    path = SETTINGS.joinpath('spectrogram.json')
    settings = Settings.from_file(path)

    path = PICKLE.joinpath('matrix.npy')

    if path.is_file():
        matrix = np.load(path, allow_pickle=True)
    else:
        matrix = mel_matrix(settings)
        np.save(path, matrix, allow_pickle=True)

    original = []
    filtered = []

    for index, (onset, offset) in enumerate(zip(onsets, offsets), 0):
        if index in custom.exclude:
            continue

        segment = signal.segment(onset, offset)
        spectrogram = create_spectrogram(segment, settings, matrix)

        image = create_image(spectrogram)
        o = to_bytes(image)

        original.append(o)

        image = filter_image(image)
        image = to_bytes(image)

        filtered.append(image)

    grid = create_grid(original, 'Original')

    file = projection.joinpath('07_original_notes.png')

    grid.save(
        file,
        format='png'
    )

    grid = create_grid(filtered, 'Filtered')

    file = projection.joinpath('08_filtered_notes.png')

    grid.save(
        file,
        format='png'
    )


if __name__ == '__main__':
    main()
