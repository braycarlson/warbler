import matplotlib
import pandas as pd
import pickle

from multiprocessing import cpu_count, Pool
from parameters import BASELINE
from path import (
    bootstrap,
    BAD,
    DATA,
    GOOD,
    MEDIOCRE,
    PICKLE,
)
from spectrogram.spectrogram import create_spectrogram


def serialize(i, f, p, v, r, n):
    template = {
        "individual": None,
        "filename": None,
        "page": None,
        "viability": None,
        "reason": None,
        "notes": None,
        "path": None,
        "spectrogram": None
    }

    path = DATA.joinpath(i, 'wav', f)

    if str(v) == 'nan':
        v = None

    if str(r) == 'nan':
        r = None

    if str(n) == 'nan':
        n = None

    template["individual"] = i
    template["filename"] = f
    template["page"] = p
    template["reason"] = r
    template["notes"] = n
    template["path"] = path

    if v is None:
        location = GOOD.joinpath(path.stem + '.png')
        template["spectrogram"] = location
        template["viability"] = 'good'
    elif v == 'P':
        location = MEDIOCRE.joinpath(path.stem + '.png')
        template["spectrogram"] = location
        template["viability"] = 'mediocre'
    else:
        location = BAD.joinpath(path.stem + '.png')
        template["spectrogram"] = location
        template["viability"] = 'bad'

    plt = create_spectrogram(path, BASELINE)

    text = f

    if r is not None:
        if n is None:
            text = text + f": {r}"
        else:
            text = text + f" ({n})"

    plt.figtext(
        0.03,
        -0.09,
        text,
        color='black',
        fontsize=12,
        fontweight=600,
        fontfamily='monospace'
    )

    plt.tight_layout()

    plt.savefig(
        location,
        bbox_inches='tight',
        pad_inches=0.5
    )

    plt.close()

    return template


@bootstrap
def main():
    spreadsheet = DATA.joinpath('2017.xlsx')
    dataframe = pd.read_excel(spreadsheet, engine="openpyxl")

    individual = dataframe.get('Individual')
    filename = dataframe.get('fileName')
    page = dataframe.get('pageNumber')
    viability = dataframe.get('Viability')
    reason = dataframe.get('viabilityReason')
    notes = dataframe.get('otherNotes')

    good = []
    mediocre = []
    bad = []

    # https://github.com/matplotlib/matplotlib/issues/21950
    matplotlib.use('Agg')

    processes = cpu_count()

    tasks = []

    processes = int(cpu_count() / 2)
    maxtasksperchild = 50

    with Pool(processes=processes, maxtasksperchild=maxtasksperchild) as pool:
        for i, f, p, v, r, n in zip(individual, filename, page, viability, reason, notes):
            path = DATA.joinpath(i, 'wav', f)
            print(f"Processing: {path}")

            # TO-DO
            if path.stem == 'STE01.1_LLbLg2017':
                continue

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

            if viability == 'good':
                good.append(template)
            elif viability == 'mediocre':
                mediocre.append(template)
            else:
                bad.append(template)

        pool.close()
        pool.join()

    # Pickle
    g = PICKLE.joinpath('good.pkl')
    m = PICKLE.joinpath('mediocre.pkl')
    b = PICKLE.joinpath('bad.pkl')

    if not g.is_file():
        g.touch()

    if not m.is_file():
        m.touch()

    if not b.is_file():
        b.touch()

    handle = open(g, 'wb')
    pickle.dump(good, handle)
    handle.close()

    handle = open(m, 'wb')
    pickle.dump(mediocre, handle)
    handle.close()

    handle = open(b, 'wb')
    pickle.dump(bad, handle)
    handle.close()


if __name__ == '__main__':
    main()
