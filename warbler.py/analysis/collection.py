"""
Collection
----------

"""

import matplotlib.pyplot as plt
import numpy.typing as npt

from constant import SETTINGS
from datatype.dataset import Dataset
from datatype.plot import StandardSpectrogram
from datatype.settings import Settings
from datatype.signal import Signal
from datatype.spectrogram import Linear, Spectrogram


def plot(
    signal: Signal,
    settings: Settings,
    matrix: npt.NDArray | None = None
) -> None:
    """Generates and displays a spectrogram plot.

    Args:
        signal: The signal used for generating the spectrogram.
        settings: The settings for the spectrogram generation.
        matrix: An optional matrix used for the linear transformation.

    Returns:
        None.

    """

    spectrogram = Spectrogram()
    strategy = Linear(signal, settings, matrix)
    spectrogram.strategy = strategy

    spectrogram = spectrogram.generate()

    plot = StandardSpectrogram()
    plot.signal = signal
    plot.spectrogram = spectrogram
    plot.create()

    plt.show()
    plt.close()


def main() -> None:
    dataset = Dataset('signal')
    dataframe = dataset.load()

    subset = dataframe.iloc[0:5]

    path = SETTINGS.joinpath('spectrogram.json')
    settings = Settings.from_file(path)

    # Bandpass filter
    column = ['signal', 'settings']

    subset[column].apply(
        lambda x: x.signal.filter(
            x.settings.butter_lowcut,
            x.settings.butter_highcut
        ),
        axis=1
    )

    # Reduce noise
    subset['signal'].apply(
        lambda x: x.reduce()
    )

    subset['signal'].apply(
        lambda x: plot(x, settings)
    )


if __name__ == '__main__':
    main()
