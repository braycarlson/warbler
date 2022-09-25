import io
import logging
import pandas as pd

from bootstrap import bootstrap
from constant import CWD, SETTINGS
from datatype.dataframe import Dataframe
from datatype.file import File
from datatype.settings import Settings
from datatype.spectrogram import pad
from datatype.spectrogramming import Linear, Mel
from logger import logger
from PIL import Image, ImageOps


log = logging.getLogger(__name__)


def create_image(spectrogram):
    buffer = io.BytesIO()
    image = Image.fromarray(~spectrogram)

    flip = ImageOps.flip(image)
    flip.save(buffer, format='png')

    return buffer.getvalue()


def crop_image(image):
    wrapper = io.BytesIO(image)
    image = Image.open(wrapper)

    x = image.width

    area = (x / 3.5, 100, x / 3.5, 0)
    crop = ImageOps.crop(image, area)

    buffer = io.BytesIO()
    crop.save(buffer, format='png')

    return buffer.getvalue()


def resize_image(image):
    wrapper = io.BytesIO(image)
    image = Image.open(wrapper)

    width, height = image.size

    width = width * 2
    height = height * 2

    upscale = image.resize(
        (width, height),
        resample=Image.Resampling.BOX
    )

    buffer = io.BytesIO()
    upscale.save(buffer, format='png')

    return buffer.getvalue()


@bootstrap
def main():
    file = Dataframe()

    aggregate = File('aggregate.xz')
    recordings = aggregate.load()

    notes = []

    total = len(recordings)

    padding = 0

    for index, recording in enumerate(recordings, 0):
        print(f"Processing: {index}/{total}")

        signal = recording.get('signal')

        path = CWD.joinpath(
            recording.get('segmentation')
        )

        settings = Settings.from_file(path)

        if settings.bandpass_filter:
            signal.filter(
                settings.butter_lowcut,
                settings.butter_highcut
            )

        if settings.reduce_noise:
            signal.reduce()

        exclude = recording.get('exclude')
        onset = recording.get('onset')
        offset = recording.get('offset')
        iterable = zip(onset, offset)

        for sequence, (onset, offset) in enumerate(iterable, 0):
            template = {}

            segment = signal.segment(onset, offset)
            # segment.normalize()

            path = SETTINGS.joinpath('spectrogram.json')
            settings = Settings.from_file(path)

            strategy = Mel(segment, settings)
            spectrogram = strategy.generate()
            spectrogram = (spectrogram * 255).astype('uint8')

            _, y = spectrogram.shape

            if y > padding:
                padding = y

            template['filename'] = recording.get('filename')
            template['folder'] = recording.get('folder')
            template['recording'] = recording.get('recording')
            template['segmentation'] = recording.get('segmentation')
            template['signal'] = recording.get('signal')
            template['sequence'] = sequence
            template['onset'] = onset
            template['offset'] = offset
            template['duration'] = offset - onset
            template['segment'] = segment
            template['spectrogram'] = spectrogram

            if sequence in exclude:
                template['exclude'] = True
            else:
                template['exclude'] = False

            notes.append(template)

    dtype = {
        'filename': 'object',
        'folder': 'object',
        'recording': 'object',
        'segmentation': 'object',
        'signal': 'object',
        'sequence': 'int8',
        'onset': 'float16',
        'offset': 'float16',
        'duration': 'float16',
        'segment': 'object',
        'spectrogram': 'object',
        'exclude': 'boolean'
    }

    dataframe = pd.DataFrame(notes)

    dataframe.astype(dtype=dtype)

    dataframe['spectrogram'] = dataframe['spectrogram'].map(
        lambda x: pad(x, padding)
    )

    dataframe['image'] = dataframe['spectrogram'].map(
        lambda x: create_image(x)
    )

    if isinstance(strategy, Mel):
        dataframe['resize'] = dataframe['image'].map(
            lambda x: resize_image(x)
        )

    if isinstance(strategy, Linear):
        dataframe['crop'] = dataframe['image'].map(
            lambda x: crop_image(x)
        )

    file.save(dataframe)


if __name__ == '__main__':
    with logger():
        main()
