import scipy.io.wavfile as wavfile

from dataclass.signal import Signal
from parameters import BASELINE
from path import DATA
from spectrogram.axes import SpectrogramAxes


def main():
    path = DATA.joinpath('TbRbRb_STE2017/wav/STE18_TbRbRb2017.wav')

    parameters = BASELINE
    signal = Signal(path)

    signal.filter(
        parameters.butter_lowcut,
        parameters.butter_highcut
    )

    timestamp = 2
    frame = signal.rate * timestamp

    left, right = signal.data[:frame - 1], signal.data[frame:]

    wavfile.write(
        'left.wav',
        signal.rate,
        left
    )

    wavfile.write(
        'right.wav',
        signal.rate,
        right
    )


if __name__ == '__main__':
    main()
