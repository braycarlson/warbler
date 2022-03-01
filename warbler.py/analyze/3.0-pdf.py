import fitz
import io
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

from parameters import Parameters
from path import INDIVIDUALS, PRINTOUTS, SEGMENT
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

    figsize = (20, 3)
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
    # https://github.com/matplotlib/matplotlib/issues/21950
    matplotlib.use('Agg')

    for individual, printout in zip(INDIVIDUALS, PRINTOUTS):
        wav = [
            wav for wav in individual.glob('wav/*.wav')
        ]

        pdf = [
            file for file in printout.glob('*.pdf')
            if 'Clean' in file.stem
        ]

        for p in pdf:
            handle = fitz.open(p)
            pages = len(handle)

            for page, song in zip(range(0, pages), wav):
                current = handle.load_page(page)

                # width = 321.900390625
                height = 98.97236633300781

                size = fitz.Rect(
                    0.0,
                    0.0,
                    375,
                    150
                )

                current.set_mediabox(size)

                rectangle = fitz.Rect(
                    6.0,
                    7.0,
                    399,
                    height
                )

                stream = io.BytesIO()

                plt = create_spectrogram(song)
                plt.tight_layout()
                plt.savefig(stream, format='png')

                stream.seek(0)

                current.insert_image(rectangle, stream=stream)

                filename = individual.stem + '.pdf'
                handle.save(filename)

                plt.close()


if __name__ == '__main__':
    main()
