from parameters import BASELINE
from path import DATA
from pathlib import Path
from spectrogram.plot import (
    create_luscinia_spectrogram,
    create_spectrogram
)


def main():
    path = Path(
        'E:/code/personal/warbler.py/data/DbWY_STE2017/wav/STE06_DbWY2017.wav'
    )

    plot = create_luscinia_spectrogram(path, BASELINE)
    # plot = create_spectrogram(path, BASELINE)

    plot.show()
    plot.close()


if __name__ == '__main__':
    main()
