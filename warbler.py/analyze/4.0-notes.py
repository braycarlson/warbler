from parameters import BASELINE
from path import CWD
from spectrogram.spectrogram import create_spectrogram


def main():
    # notes = [
    #     file for file in CWD.joinpath('notes').glob('*.wav')
    # ]

    # print(notes)

    BASELINE.update('n_fft', 4096)

    path = CWD.joinpath('notes/STE01c_DgDgY2017_00.wav')
    plt = create_spectrogram(path, BASELINE)

    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
