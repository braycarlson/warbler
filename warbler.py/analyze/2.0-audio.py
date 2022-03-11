import numpy as np
import shutil

import matplotlib.pyplot as plt
from parameters import Parameters
from path import DATA, SEGMENT
from pathlib import Path
from scipy.io import wavfile
from spectrogram.axes import SpectrogramAxes
from spectrogram.plot import (
    create_luscinia_spectrogram,
    # create_spectrogram
)
from vocalseg.utils import (
    butter_bandpass_filter,
    int16tofloat32
)

# Print without truncation
np.set_printoptions(threshold=np.inf)

# Parameters
file = SEGMENT.joinpath('parameters.json')
parameters = Parameters(file)


def main():
    parameters.n_fft = 4096
    parameters.hop_length_ms = 1
    parameters.win_length_ms = 5
    parameters.ref_level_db = 50
    parameters.pre = 0.97
    parameters.min_level_db = -50
    parameters.butter_lowcut = 1500
    parameters.butter_highcut = 10000

    # path = DATA.joinpath('LLbLg_STE2017/wav/STE01.1_LLbLg2017.wav')
    path = DATA.joinpath('RbRY_STE2017/wav/STE01_RbRY2017.wav')

    create_luscinia_spectrogram(path, parameters)

    plt.savefig(
        path.stem,
        bbox_inches='tight',
        pad_inches=0.5
    )

    plt.show()
    plt.close()

    rate, data = wavfile.read(path)

    data = butter_bandpass_filter(
        int16tofloat32(data),
        parameters.butter_lowcut,
        parameters.butter_highcut,
        rate
    )

    original = 'original' + '_' + path.name
    filtered = 'filtered' + '_' + path.name

    Path(original).unlink(missing_ok=True)
    Path(filtered).unlink(missing_ok=True)

    # Copy the original .wav
    shutil.copy(path, original)

    # Save the filtered .wav
    wavfile.write(filtered, rate, data)


if __name__ == '__main__':
    main()
