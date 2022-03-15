import fitz
import io
import matplotlib
import pandas as pd

from multiprocessing import cpu_count, Pool
from natsort import os_sorted
from parameters import BASELINE
from path import DATA, INDIVIDUALS, PRINTOUTS
from spectrogram.axes import SpectrogramAxes
from spectrogram.plot import (
    create_luscinia_spectrogram,
    # create_spectrogram
)
from tqdm import tqdm


def process(data):
    individual, filename, page = data
    path = individual.joinpath('wav', filename)

    if not path.is_file():
        print(path)
        raise

    template = {
        'filename': filename,
        'path': path,
        'page': int(page) - 1,
        'stream': io.BytesIO()
    }

    stream = template.get('stream')

    plt = create_luscinia_spectrogram(path, BASELINE)
    plt.savefig(stream, format='png')

    stream.seek(0)
    plt.close()

    return template


def main():
    # https://github.com/matplotlib/matplotlib/issues/21950
    matplotlib.use('Agg')

    spreadsheet = DATA.joinpath('2017.xlsx')

    dataframe = pd.read_excel(
        spreadsheet,
        engine='openpyxl'
    )

    width = 321.900390625
    height = 98.97236633300781

    # Set page size
    page_size = fitz.Rect(
        0.0,
        0.0,
        375,
        165
    )

    # Add text
    text_box = fitz.Rect(
        0.0,
        5.0,
        375,
        80
    )

    # Add image
    image_box = fitz.Rect(
        9.5,
        8.0,
        width - 2,
        height
    )

    processes = int(cpu_count() / 2)
    maxtasksperchild = 200

    for individual, printout in zip(INDIVIDUALS, PRINTOUTS):
        row = dataframe[dataframe.Individual == individual.stem]

        pdf = os_sorted([
            file for file in printout.glob('*.pdf')
            if 'Clean' in file.stem
        ])

        for p in pdf:
            handle = fitz.open(p)
            total = len(handle)

            with Pool(
                processes=processes,
                maxtasksperchild=maxtasksperchild
            ) as pool:
                data = [
                    (individual, filename, page)
                    for _, filename, page in zip(
                        range(0, total),
                        row.updatedFileName.values,
                        row.pageNumber.values
                    )
                ]

                total = len(data)

                results = tqdm(
                    pool.imap(
                        process,
                        data
                    ),
                    total=total
                )

                for result in results:
                    page = result.get('page')
                    filename = result.get('filename')
                    stream = result.get('stream')

                    current = handle.load_page(page)
                    current.set_mediabox(page_size)

                    current.insert_textbox(
                        text_box,
                        filename,
                        fontsize=8,
                        fontname='Times-Roman',
                        fontfile=None,
                        align=1
                    )

                    current.insert_image(
                        image_box,
                        stream=stream
                    )

                    filename = individual.stem + '.pdf'
                    handle.save(filename)

                pool.close()
                pool.join()


if __name__ == '__main__':
    main()
