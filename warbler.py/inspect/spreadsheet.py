import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd

from parameters import Parameters
from path import INDIVIDUALS, INSPECT, SEGMENT
from pathlib import Path
from scipy.io import wavfile
from vocalseg.utils import (
    butter_bandpass_filter,
    spectrogram,
    int16tofloat32,
    plot_spec
)


# Parameters
file = SEGMENT.joinpath('parameters.json')
parameters = Parameters(file)


def create_spectrogram(path):
    rate, data = wavfile.read(path)

    data = butter_bandpass_filter(
        int16tofloat32(data),
        parameters.butter_lowcut,
        parameters.butter_highcut,
        rate
    )

    spec = spectrogram(
        data,
        rate,
        n_fft=parameters.n_fft,
        hop_length_ms=parameters.hop_length_ms,
        win_length_ms=parameters.win_length_ms,
        ref_level_db=parameters.ref_level_db,
        pre=parameters.pre,
        min_level_db=parameters.min_level_db,
    )

    np.shape(spec)

    figsize = (15, 3)
    fig, ax = plt.subplots(figsize=figsize)
    plot_spec(spec, fig, ax)

    plt.ylim([0, 1000])

    ticks_y = ticker.FuncFormatter(
        lambda x, pos: '{0:g}'.format(x / 1e2)
    )

    ax.yaxis.set_major_formatter(ticks_y)

    ax.set_xlabel("Time (ms)")
    ax.set_ylabel('Frequency (kHz)')

    return plt


def main():
    directory = Path('spectrograms')
    directory.mkdir(parents=True, exist_ok=True)

    spreadsheet = INSPECT.joinpath('2017.xlsx')
    dataframe = pd.read_excel(spreadsheet, engine="openpyxl")

    filename = dataframe.get('fileName')
    viability = dataframe.get('Viability')

    partial = [
        f for f, v in zip(filename, viability)
        if v == 'P'
    ]

    length = len(partial)

    files = [
        list(
            files.glob('wav/*.wav')
        ) for files in INDIVIDUALS
    ]

    path = []

    for file in files:
        for songs in file:
            filename = songs.stem + songs.suffix

            if filename in partial:
                path.append(songs)

    found = len(path)

    print(f"Found {found}/{length} songs.")

    for i, p in enumerate(path, 1):
        print(f"Saving: {i}/{found}")

        if p.stem == 'STE01.1_LLbLg2017':
            continue

        spectrogram = create_spectrogram(p)

        spectrogram.savefig(
            directory.joinpath(p.stem),
            bbox_inches='tight'
        )

        spectrogram.close()


if __name__ == '__main__':
    main()
