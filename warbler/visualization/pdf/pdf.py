from __future__ import annotations

import fitz
import io
import matplotlib.pyplot as plt
import pandas as pd

from warbler.constant import (
    IMAGES,
    PDF,
    RECORDINGS,
    SETTINGS,
    SPREADSHEET
)
from warbler.datatype.plot import (
    StandardSpectrogram,
    SegmentationSpectrogram
)
from warbler.datatype.segmentation import DynamicThresholdSegmentation
from warbler.datatype.settings import Settings
from warbler.datatype.signal import Signal
from warbler.datatype.spectrogram import Linear, Spectrogram


def main() -> None:
    spreadsheet = SPREADSHEET.joinpath('2017.xlsx')

    dataframe = pd.read_excel(
        spreadsheet,
        sheet_name='2017_python_viability',
        engine='openpyxl'
    )

    keys = dataframe.individual.unique()

    individuals = dataframe.individual.tolist()
    filenames = dataframe.updated_filename.tolist()
    pages = dataframe.printout_page_number.tolist()

    # Image box
    width = 325
    height = 100
    offset = 40
    expand = 325

    for key in keys:
        document = fitz.open()

        toc = []

        iterable = zip(
            individuals,
            RECORDINGS,
            IMAGES,
            filenames,
            pages
        )

        for _, (individual, recording, image, filename, page) in enumerate(iterable, 1):
            if key == individual and filename == recording.stem == image.stem:
                print(f"Processing: {filename}")

                current = document.new_page(
                    width=700,
                    height=700
                )

                # Set mediabox
                current.set_mediabox(
                    fitz.Rect(
                        0.0,
                        0.0,
                        700,
                        750
                    )
                )

                # Title
                current.insert_textbox(
                    fitz.Rect(
                        0.0,
                        offset,
                        700,
                        100
                    ),
                    filename,
                    fontname='Times-Roman',
                    fontsize=24,
                    align=1
                )

                # Standard
                current.insert_textbox(
                    fitz.Rect(
                        10,
                        210,
                        680,
                        750
                    ) * current.rotation_matrix,
                    'Standard',
                    fontsize=10,
                    align=0,
                    rotate=270
                )

                current.insert_image(
                    fitz.Rect(
                        0,
                        offset / 2,
                        width + expand,
                        height + expand
                    ).normalize(),
                    filename=image,
                    keep_proportion=True
                )

                # Python: Filtered
                path = SETTINGS.joinpath('segmentation.json')
                settings = Settings.from_file(path)

                signal = Signal(recording)

                if settings.bandpass_filter:
                    signal.filter(
                        settings.butter_lowcut,
                        settings.butter_highcut
                    )

                if settings.reduce_noise:
                    signal.reduce()

                strategy = Linear(signal, settings)

                spectrogram = Spectrogram()
                spectrogram.strategy = strategy
                spectrogram = spectrogram.generate()

                plot = StandardSpectrogram()
                plot.signal = signal
                plot.spectrogram = spectrogram
                plot.create()

                stream = io.BytesIO()
                plt.savefig(stream, format='png')

                stream.seek(0)
                plt.close()

                current.insert_textbox(
                    fitz.Rect(
                        15,
                        325,
                        680,
                        750
                    ) * current.rotation_matrix,
                    'Python: Filtered',
                    fontsize=10,
                    align=0,
                    rotate=270
                )

                current.insert_image(
                    fitz.Rect(
                        11,
                        height + offset,
                        (width + expand) - 3.5,
                        (height + offset) * 2 + expand
                    ),
                    stream=stream,
                    keep_proportion=True
                )

                # Python: Segmented
                algorithm = DynamicThresholdSegmentation()
                algorithm.signal = signal
                algorithm.settings = settings
                algorithm.start()

                try:
                    algorithm.component.get('spectrogram')
                    algorithm.component.get('onset')
                    algorithm.component.get('offset')
                except Exception:
                    print(f"Unable to process: {filename}")
                    continue

                plot = SegmentationSpectrogram()
                plot.algorithm = algorithm
                plot.signal = signal
                plot.create()

                plt.xticks(
                    fontfamily='Arial',
                    fontsize=14,
                    fontweight=600
                )

                plt.yticks(
                    fontfamily='Arial',
                    fontsize=14,
                    fontweight=600
                )

                stream = io.BytesIO()
                plt.savefig(stream, format='png')

                stream.seek(0)
                plt.close()

                current.insert_textbox(
                    fitz.Rect(
                        15,
                        458,
                        680,
                        750
                    ) * current.rotation_matrix,
                    'Python: Segmented',
                    fontsize=10,
                    align=0,
                    rotate=270
                )

                current.insert_image(
                    fitz.Rect(
                        11,
                        (height + offset) * 2,
                        (width + expand) - 3.5,
                        (height + offset) * 3 + expand
                    ),
                    stream=stream,
                    keep_proportion=True
                )

                # Table of Contents
                item = [1, filename, int(page)]
                toc.append(item)

            document.set_toc(toc)

        path = PDF.joinpath(key + '.pdf')
        document.save(path, deflate=True)
        document.close()


if __name__ == '__main__':
    main()
