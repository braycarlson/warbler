"""
Image: Single
-------------

"""

from __future__ import annotations

import matplotlib.pyplot as plt

from constant import CWD, SETTINGS
from datatype.dataset import Dataset
from datatype.imaging import (
    create_grid,
    create_image,
    filter_image,
    to_bytes
)
from datatype.plot import StandardSpectrogram
from datatype.settings import Settings
from datatype.signal import Signal
from datatype.spectrogram import create_spectrogram
from io import BytesIO
from pathlib import Path
from PIL import Image


def main() -> None:
    dataset = Dataset('signal')
    dataframe = dataset.load()

    original = [
        Signal(CWD.joinpath(recording))
        for recording in dataframe.recording.tolist()
    ]

    signal = dataframe.signal.tolist()

    path = SETTINGS.joinpath('spectrogram.json')
    settings = Settings.from_file(path)

    for o, s in zip(original, signal):
        spectrogram = create_spectrogram(signal, settings)

        # Original
        plot = StandardSpectrogram()
        plot.signal = signal
        plot.spectrogram = spectrogram
        figure = plot.create()

        plt.axis('off')

        obuffer = BytesIO()

        figure.savefig(
            obuffer,
            bbox_inches='tight',
            dpi=300,
            format='png',
            pad_inches=0
        )

        oimage = Image.open(obuffer)

        plt.close()

        # Filtered
        image = create_image(spectrogram)
        filtered = filter_image(image)

        plot = StandardSpectrogram()
        plot.signal = signal
        plot.spectrogram = filtered
        figure = plot.create()

        plt.axis('off')

        fbuffer = BytesIO()

        figure.savefig(
            fbuffer,
            bbox_inches='tight',
            dpi=300,
            format='png',
            pad_inches=0
        )

        fimage = Image.open(fbuffer)

        plt.close()

        image = Image.new(
            'L',
            (oimage.width, oimage.height - 150 + oimage.height - 150)
        )

        image.paste(
            oimage,
            (0, 0)
        )

        image.paste(
            fimage,
            (0, fimage.height - 300)
        )

        image.show()


if __name__ == '__main__':
    main()
