import keyboard
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pickle
import time

from parameters import Parameters
from path import INDIVIDUALS, SEGMENT
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
    file = Path("E:/code/personal/warbler.py/data/LLbLg_STE2017/wav/STE01.1_LLbLg2017.wav")
    plt = create_spectrogram(file)

    plt.show()

    # # Pickle
    # file = Path('audio.pkl')

    # if not file.is_file():
    #     file.touch()

    #     individual = [
    #         individual.glob('wav/*.wav') for individual in INDIVIDUALS
    #     ]

    #     wav = [list(file) for file in individual]

    #     handle = open(file, 'wb')
    #     pickle.dump(wav, handle)
    #     handle.close()

    # with open(file, 'rb') as file:
    #     audio = pickle.load(file)

    # matplotlib.interactive(True)
    # quit = False

    # acceptable = []
    # unacceptable = []

    # for individual in audio:
    #     if quit:
    #         break

    #     for file in individual:
    #         time.sleep(0.5)
    #         # print(file.stem)

    #         plt = create_spectrogram(file)

    #         manager = plt.get_current_fig_manager()
    #         manager.window.state('zoomed')

    #         fig = plt.gcf()

    #         plt.ion()
    #         plt.pause(0.001)

    #         if keyboard.read_key() == 'q':
    #             quit = True

    #             plt.close(fig)
    #             break
    #         elif keyboard.read_key() == 'n':
    #             print(f"{file} is viable")
    #             acceptable.append(file)

    #             plt.close(fig)
    #             continue
    #         else:
    #             print(f"{file} is not viable")
    #             unacceptable.append(file)

    #             plt.close(fig)
    #             continue

    # plt.show()

    # print(f"Acceptable: {acceptable}")

    # print(f"Unacceptable: {unacceptable}")


if __name__ == '__main__':
    main()
