import fitz
import io
import matplotlib
import pandas as pd

from natsort import os_sorted
from path import ANALYZE, BASELINE, DATA, INDIVIDUALS, PRINTOUTS
from spectrogram.axes import SpectrogramAxes
from spectrogram.plot import create_luscinia_spectrogram


def get():
    pdf = [
        file for file in ANALYZE.glob('pdf/*.pdf')
        if file.is_file()
    ]

    for document in pdf[:1]:
        handle = fitz.open(document)

        for metadata in handle.get_toc():
            level, filename, page = metadata
            print(filename, page)


def set():
    spreadsheet = DATA.joinpath('2017.xlsx')

    dataframe = pd.read_excel(
        spreadsheet,
        engine='openpyxl'
    )

    for index, (individual, printout) in enumerate(zip(INDIVIDUALS, PRINTOUTS), 0):
        row = dataframe[dataframe.individual == individual.stem]

        pdf = [
            file for file in printout.glob('*.pdf')
            if file.is_file() and 'Clean' in file.stem
        ]

        for document in pdf:
            handle = fitz.open(document)
            total = len(handle)

            toc = []

            for _, filename, page in zip(
                range(0, total),
                row.updated_filename.values,
                row.printout_page_number.values
            ):
                item = [1, filename, int(page)]
                toc.append(item)

            handle.set_toc(toc)

        handle.save(individual.stem + '.pdf')


def main():
    spreadsheet = DATA.joinpath('2017.xlsx')

    dataframe = pd.read_excel(
        spreadsheet,
        sheet_name='2017_python_viability',
        engine='openpyxl'
    )

    filenames = dataframe.updated_filename.values
    pages = dataframe.printout_page_number.values

    images = [
        file for file in ANALYZE.glob('printout/*/*/*.jpg')
        if file.is_file()
    ]

    images = os_sorted(images)

    for index, (filename, page, image) in enumerate(zip(filenames, pages, images), 1):
        if filename == image.stem:
            continue
        else:
            print(filename)

    # for index, individual in enumerate(INDIVIDUALS, 0):
    #     # row = dataframe[dataframe.individual == individual.stem]

    #     print(individual)

        # for index, (filename, image) in enumerate(zip(dataframe.updated_filename.values, images), 0):
        #     print(filename, image.stem)

            # if filename == image.stem:
            #     continue
            #     # print(filename)
            # else:
            #     print('NOT THE SAME: ', filename)

    # for document in pdf[:1]:
    #     handle = fitz.open(document)

    #     for metadata in handle.get_toc():
    #         level, filename, page = metadata
    #         print(filename, page)








    # width = 321.900390625
    # height = 98.97236633300781

    # # Set page size
    # page_size = fitz.Rect(
    #     0.0,
    #     0.0,
    #     375,
    #     165
    # )

    # # Add text
    # text_box = fitz.Rect(
    #     0.0,
    #     5.0,
    #     375,
    #     80
    # )

    # # Add image
    # image_box = fitz.Rect(
    #     9.5,
    #     8.0,
    #     width - 2,
    #     height
    # )


if __name__ == '__main__':
    main()
