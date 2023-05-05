import fitz
import io
import matplotlib.pyplot as plt
import pandas as pd

from constant import (
    IMAGES,
    PDF,
    RECORDINGS,
    SETTINGS,
    SPREADSHEET
)
from datatype.plot import (
    StandardSpectrogram,
    SegmentationSpectrogram
)
from datatype.settings import Settings
from datatype.signal import Signal
from datatype.spectrogram import Spectrogram
from vocalseg.dynamic_thresholding import dynamic_threshold_segmentation


def main():
    spreadsheet = SPREADSHEET.joinpath('2017.xlsx')

    dataframe = pd.read_excel(
        spreadsheet,
        sheet_name='2017_python_viability',
        engine='openpyxl'
    )

    keys = dataframe.individual.unique()

    individuals = dataframe.individual.values
    filenames = dataframe.updated_filename.values
    pages = dataframe.printout_page_number.values

    # Image box
    width = 325
    height = 100
    offset = 40
    expand = 325

    for key in keys:
        document = fitz.open()

        toc = []

        iterable = zip(individuals, RECORDINGS, IMAGES, filenames, pages)

        for index, (individual, recording, image, filename, page) in enumerate(iterable, 1):
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

                spectrogram = Spectrogram(signal, settings)
                spectrogram = spectrogram.generate()

                plot = StandardSpectrogram(signal, spectrogram)
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
                threshold = dynamic_threshold_segmentation(
                    signal.data,
                    signal.rate,
                    n_fft=settings.n_fft,
                    hop_length_ms=settings.hop_length_ms,
                    win_length_ms=settings.win_length_ms,
                    ref_level_db=settings.ref_level_db,
                    pre=settings.preemphasis,
                    min_level_db=settings.min_level_db,
                    min_level_db_floor=settings.min_level_db_floor,
                    db_delta=settings.db_delta,
                    silence_threshold=settings.silence_threshold,
                    # spectral_range=settings.spectral_range,
                    min_silence_for_spec=settings.min_silence_for_spec,
                    max_vocal_for_spec=settings.max_vocal_for_spec,
                    min_syllable_length_s=settings.min_syllable_length_s
                )

                try:
                    threshold.get('spec')
                    threshold.get('onsets')
                    threshold.get('offsets')
                except Exception:
                    print(f"Unable to process: {filename}")
                    continue

                plot = SegmentationSpectrogram()
                plot.signal = signal
                plot.threshold = threshold
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
        document.save(path)
        document.close()


if __name__ == '__main__':
    main()
