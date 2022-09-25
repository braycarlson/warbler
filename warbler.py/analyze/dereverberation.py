import fitz
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from constant import CWD, PICKLE, RECORDINGS, SETTINGS
from datatype.axes import SpectrogramAxes
from datatype.file import File
from datatype.settings import Settings
from datatype.signal import Signal
from datatype.spectrogram import Spectrogram
from io import BytesIO
from natsort import os_sorted
from pathlib import Path
from plot import LusciniaSpectrogram
from zipfile import ZipFile


def is_matching(archives):
    filelist = []

    for archive in archives:
        with ZipFile(archive, 'r') as handle:
            for file in handle.filelist:
                filelist.append(file.filename)

    unique = set(filelist)

    count = [
        filelist.count(filename)
        for filename in unique
    ]

    match = set(count)

    if len(match) == 1:
        return filelist

    return []


def get_spectrogram(signal, settings):
    if settings.bandpass_filter:
        signal.filter(
            settings.butter_lowcut,
            settings.butter_highcut
        )

    if settings.reduce_noise:
        signal.reduce()

    spectrogram = Spectrogram(signal, settings)
    spectrogram = spectrogram.generate()

    plot = LusciniaSpectrogram(signal, spectrogram)
    plot.create()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')

    buffer.seek(0)
    plt.close()

    return buffer.getvalue()


def create_dataset():
    archives = os_sorted([
        file
        for file in PICKLE.glob('*.zip')
        if file.is_file() and file.suffix == '.zip'
    ])

    filelist = is_matching(archives)

    if len(filelist) == 0:
        raise

    unique = set(filelist)

    filelist = list(unique)
    filelist.sort()

    filename = [
        recording.stem
        for recording in RECORDINGS
    ]

    dataframe = pd.DataFrame(
        {'filename': filename}
    )

    path = SETTINGS.joinpath('spectrogram.json')
    settings = Settings.from_file(path)

    for index, archive in enumerate(archives, 0):
        for filename in filelist:
            with ZipFile(archive, 'r') as handle:
                handle.extract(filename)

            file = File(filename)
            recordings = file.load()

            for recording in recordings:
                name = recording.get('filename')
                print(f"Processing: {name}")

                if index == 0:
                    original = CWD.joinpath(
                        recording.get('recording')
                    )

                    original = Signal(original)
                    buffer = get_spectrogram(original, settings)

                    dataframe.loc[
                        dataframe['filename'] == name,
                        'original'
                    ] = buffer

                signal = recording.get('signal')
                buffer = get_spectrogram(signal, settings)

                dataframe.loc[
                    dataframe['filename'] == name,
                    archive.stem
                ] = buffer

            Path(filename).unlink(missing_ok=True)

    with open('dataset.pkl', 'wb') as handle:
        pickle.dump(dataframe, handle)


def create_document():
    with open('dataset.pkl', 'rb') as handle:
        dataset = pickle.load(handle)

    filename, *spectrogram = dataset.to_numpy().T
    columns = dataset.columns.to_numpy()

    index = np.argwhere(columns == 'filename')
    columns = np.delete(columns, index)

    document = fitz.open()

    toc = []

    for count, name in enumerate(filename, 0):
        print(f"Processing: {name}")

        # Dimensions
        width = 650
        height = 100
        offset = 40
        text = 200
        expand = 325

        # Page
        current = document.new_page(
            width=700,
            height=1000
        )

        # Title
        current.insert_textbox(
            fitz.Rect(
                0.0,
                40,
                700,
                100
            ),
            name,
            fontname='Times-Roman',
            fontsize=24,
            align=1
        )

        for index, column in enumerate(columns, 0):
            if index > 0:
                height = height + expand
                text = text + (expand / 2)

            row = spectrogram[index][count]
            stream = BytesIO(row)

            current.insert_image(
                fitz.Rect(
                    (700 - width) / 2,
                    offset / 2,
                    width,
                    height + expand
                ).normalize(),
                stream=stream,
                keep_proportion=True
            )

            current.insert_textbox(
                fitz.Rect(
                    0,
                    text,
                    680,
                    1000
                ) * current.rotation_matrix,
                column,
                fontsize=12,
                align=0,
                rotate=270
            )

        # Table of Contents
        item = [1, name, count]
        toc.append(item)

    document.set_toc(toc)
    document.save('dereverbation.pdf')
    document.close()


def main():
    # create_dataset()
    create_document()


if __name__ == '__main__':
    main()
