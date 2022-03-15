import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import pickle

from bootstrap import bootstrap
from multiprocessing import cpu_count, Pool
from parameters import Parameters
from path import (
    DATA,
    PICKLE,
    SPECTROGRAM
)
from spectrogram.axes import SpectrogramAxes
from spectrogram.plot import create_luscinia_spectrogram


def serialize(individual, filename, page, viability, reason, notes):
    template = {
        'individual': None,
        'filename': None,
        'page': None,
        'viability': None,
        'reason': None,
        'notes': None,
        'song': None,
        'parameter': None,
        'metadata': None,
        'spectrogram': None
    }

    song = DATA.joinpath(individual, 'wav', filename + '.wav')
    parameter = DATA.joinpath(individual, 'parameter', filename + '.json')
    metadata = DATA.joinpath(individual, 'json', filename + '.json')

    template['individual'] = individual
    template['filename'] = filename
    template['page'] = page
    template['reason'] = reason
    template['notes'] = notes
    template['song'] = song
    template['parameter'] = parameter
    template['metadata'] = metadata

    if viability == 'Y':
        location = SPECTROGRAM.joinpath('good', filename + '.png')
        template['spectrogram'] = location
        template['viability'] = 'good'
    elif viability == 'P':
        location = SPECTROGRAM.joinpath('mediocre', filename + '.png')
        template['spectrogram'] = location
        template['viability'] = 'mediocre'
    else:
        location = SPECTROGRAM.joinpath('bad', filename + '.png')
        template['spectrogram'] = location
        template['viability'] = 'bad'

    parameters = Parameters(parameter)
    create_luscinia_spectrogram(song, parameters)

    plt.title(
        filename,
        fontsize=14,
        weight=600,
        pad=15,
    )

    if reason is not None:
        figtext = 'Note: ' + reason

        plt.figtext(
            0.03,
            -0.09,
            figtext,
            color='black',
            fontfamily='Arial',
            fontsize=12,
            fontweight=600,
        )

    plt.savefig(
        location,
        bbox_inches='tight',
        pad_inches=0.5
    )

    plt.close('all')
    return template


def save(file, data):
    if not file.is_file():
        file.touch()

    handle = open(file, 'wb')
    pickle.dump(data, handle)
    handle.close()


@bootstrap
def main():
    spreadsheet = DATA.joinpath('2017.xlsx')

    dataframe = pd.read_excel(
        spreadsheet,
        sheet_name='2017_python_viability',
        engine='openpyxl'
    )

    dataframe = dataframe.where(
        pd.notnull(dataframe),
        None
    )

    individual = dataframe.get('individual')
    filename = dataframe.get('updated_filename')
    page = dataframe.get('printout_page_number')
    viability = dataframe.get('python_viability')
    reason = dataframe.get('python_viability_reason')
    notes = dataframe.get('python_notes')

    aggregate = []
    good = []
    mediocre = []
    bad = []

    # https://github.com/matplotlib/matplotlib/issues/21950
    matplotlib.use('Agg')

    processes = int(cpu_count() / 2)
    maxtasksperchild = 15

    with Pool(processes=processes, maxtasksperchild=maxtasksperchild) as pool:
        tasks = []

        for i, f, p, v, r, n in zip(individual, filename, page, viability, reason, notes):
            tasks.append(
                pool.apply_async(
                    serialize,
                    args=(i, f, p, v, r, n)
                )
            )

        pool.close()
        pool.join()

    with Pool(processes=processes, maxtasksperchild=maxtasksperchild) as pool:
        for task in tasks:
            template = task.get(10)
            viability = template.get('viability')

            aggregate.append(template)

            if viability == 'good':
                good.append(template)
            elif viability == 'mediocre':
                mediocre.append(template)
            else:
                bad.append(template)

        pool.close()
        pool.join()

    # Pickle
    a = PICKLE.joinpath('aggregate.pkl')
    g = PICKLE.joinpath('good.pkl')
    m = PICKLE.joinpath('mediocre.pkl')
    b = PICKLE.joinpath('bad.pkl')

    save(a, aggregate)
    save(g, good)
    save(m, mediocre)
    save(b, bad)


if __name__ == '__main__':
    main()
