import fitz
import io
import matplotlib.pyplot as plt
import pandas as pd

from dataclass.signal import Signal
from parameters import Parameters
from path import (
    ANALYZE,
    BASELINE,
    DATA,
    IMAGES,
    SONGS
)
from spectrogram.axes import SpectrogramAxes
from spectrogram.plot import (
    create_luscinia_spectrogram,
    plot_segmentation
)
from vocalseg.dynamic_thresholding import dynamic_threshold_segmentation


def main():
    spreadsheet = DATA.joinpath('2017.xlsx')

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

        for index, (individual, song, image, filename, page) in enumerate(zip(individuals, SONGS, IMAGES, filenames, pages), 1):
            if key == individual and filename == song.stem == image.stem:
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

                # Luscinia
                current.insert_textbox(
                    fitz.Rect(
                        10,
                        210,
                        680,
                        750
                    ) * current.rotation_matrix,
                    'Luscinia',
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
                parameters = Parameters(BASELINE)
                create_luscinia_spectrogram(song, parameters)

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
                signal = Signal(song)

                signal.filter(
                    parameters.butter_lowcut,
                    parameters.butter_highcut
                )

                dts = dynamic_threshold_segmentation(
                    signal.data,
                    signal.rate,
                    n_fft=parameters.n_fft,
                    hop_length_ms=parameters.hop_length_ms,
                    win_length_ms=parameters.win_length_ms,
                    ref_level_db=parameters.ref_level_db,
                    pre=parameters.preemphasis,
                    min_level_db=parameters.min_level_db,
                    silence_threshold=parameters.silence_threshold,
                    min_syllable_length_s=parameters.min_syllable_length_s,
                )

                try:
                    dts.get('spec')
                    dts.get('onsets')
                    dts.get('offsets')
                except Exception:
                    print(f"Unable to process: {filename}")
                    continue

                plot_segmentation(signal, dts)

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

        path = ANALYZE.joinpath('pdf', key + '.pdf')
        document.save(path)
        document.close()


if __name__ == '__main__':
    main()
