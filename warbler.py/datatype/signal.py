"""
Signal
------

"""

from __future__ import annotations

import io
import librosa
import noisereduce as nr
import numpy as np
import pathlib
import scipy
import soundfile as sf

from abc import abstractmethod, ABC
from nara_wpe.wpe import wpe
from nara_wpe.utils import istft, stft
from scipy.io import wavfile
from scipy.signal import butter, lfilter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt

    from datatype.settings import Settings
    from typing_extensions import BinaryIO


class Strategy(ABC):
    """A base class for strategies on loading an audio signal."""

    @abstractmethod
    def load(
        self,
        path: str | BinaryIO | pathlib.Path | None
    ) -> tuple[int, npt.NDArray]:
        """An abstract method for loading an audio signal.

        Args:
            path: The path to the audio file.

        Returns:
            The sample rate of the audio and the data itself.

        """

        raise NotImplementedError


class Librosa(Strategy):
    """A class that loads audio using Librosa."""

    def load(
        self,
        path: str | BinaryIO | pathlib.Path | None
    ) -> tuple[int, npt.NDArray]:
        """Load audio using Librosa.

        Args:
            path: The path to the audio file.

        Returns:
            The sample rate of the audio and the data itself.

        """

        if isinstance(path, pathlib.Path):
            path = str(path)

        data, rate = librosa.core.load(path, sr=None)
        return rate, data


class Scipy(Strategy):
    """A class that loads audio using Scipy."""

    def load(
        self,
        path: str | BinaryIO | pathlib.Path | None
    ) -> tuple[int, npt.NDArray]:
        """Load audio using Scipy.

        Args:
            path: The path to the audio file.

        Returns:
            The sample rate of the audio and the data itself.

        """

        rate, data = wavfile.read(path)
        return rate, data


class Soundfile(Strategy):
    """A class that loads audio using Soundfile."""

    def load(
        self,
        path: str | BinaryIO | pathlib.Path | None
    ) -> tuple[int, npt.NDArray]:
        """Load audio using Soundfile.

        Args:
            path: The path to the audio file.

        Returns:
            The sample rate of the audio and the data itself.

        """

        if isinstance(path, pathlib.Path):
            path = str(path)

        data, rate = sf.read(path)
        return rate, data


class Signal:
    """A class representing an audio signal.

    Args:
        path: The path to the audio file.

    Attributes:
        _strategy: The current strategy for loading audio data.
        _path: The path to the audio file.
        _rate: The sample rate of the audio signal.
        _data: The audio signal data as a NumPy array.

    """

    def __init__(self, path: str | BinaryIO | pathlib.Path | None = None):
        self._strategy = Scipy()
        self._path = path
        self._rate, self._data = self._load()

    def _load(self) -> tuple[int, npt.NDArray]:
        """Load audio data using the selected strategy.

        Returns:
            The sample rate of the audio and the data itself.

        Raises:
            FileNotFoundError: If the specified file does not exist.

        """

        if isinstance(self.path, pathlib.Path) and not self.path.exists():
            message = f"{self.path} was not found"
            raise FileNotFoundError(message)

        rate, data = self.strategy.load(self.path)

        if np.issubdtype(data.dtype, np.integer):
            data = np.array(data / 32768.0).astype('float32')

        return rate, data

    @property
    def strategy(self) -> Strategy:
        """Get the current strategy for loading audio data.

        Returns:
            The current strategy for loading audio.

        """

        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
        self._strategy = strategy

    @property
    def data(self) -> npt.NDArray:
        """Get the audio data.

        Returns:
            The audio data as a NumPy array.

        """

        return self._data

    @data.setter
    def data(self, data: npt.NDArray) -> None:
        self._data = data

    @property
    def path(self) -> str | BinaryIO | pathlib.Path | None:
        """Get the path to the audio file.

        Returns:
            A string or pathlib.Path object representing the path to the
            audio file.

        """

        return self._path

    @path.setter
    def path(self, path: str | BinaryIO | pathlib.Path) -> None:
        self._path = path

    @property
    def rate(self) -> int:
        """Get the sample rate of the audio signal.

        Returns:
            The sample rate as an integer.

        """

        return self._rate

    @property
    def duration(self) -> float:
        """Get the duration of the audio signal in seconds.

        Returns:
            The duration of the audio signal in seconds as a float.

        """

        self._duration = len(self.data) / float(self.rate)
        return self._duration

    def filter(self, lowcut: float, highcut: float, order: int = 5) -> None:
        """Apply a bandpass filter to the audio signal.

        Args:
            lowcut: The lower cutoff frequency of the filter.
            highcut: The upper cutoff frequency of the filter.
            order: The order of the filter.

        Returns:
            None.

        """

        if highcut > int(self.rate / 2):
            highcut = int(self.rate / 2)

        nyquist = 0.5 * self.rate
        low = lowcut / nyquist
        high = highcut / nyquist

        numerator, denominator = butter(
            order,
            [low, high],
            btype='band'
        )

        self.data = lfilter(
            numerator,
            denominator,
            self.data
        )

    def frequencies(self) -> tuple[float, float]:
        """Calculate the minimum and maximum frequencies in the signal.

        Args:
            None.

        Returns:
            A tuple containing the minimum and maximum frequencies as a floats.
            If no frequencies are above the threshold, returns (nan, nan).

        """

        # Calculate the time step between each sample
        timestep = 1.0 / self.rate

        # Create an array of times for each sample
        time = np.arange(
            0,
            self.duration,
            timestep
        )

        # The Fast Fourier Transform (FFT) of the input signal
        fft = abs(
            scipy.fft.fft(self.data)
        )

        # Get the first half of the FFT coefficients
        bisection = fft[
            range(self.data.size // 2)
        ]

        bisection = abs(bisection)

        # The frequencies corresponding to the FFT coefficients
        frequencies = scipy.fftpack.fftfreq(
            self.data.size,
            time[1] - time[0]
        )

        # Get the first half of the frequencies
        halved = frequencies[
            range(self.data.size // 2)
        ]

        amplitude = np.array(bisection)
        threshold = np.where(amplitude > 5)

        if halved[threshold].size == 0:
            return (np.nan, np.nan)

        minimum = min(halved[threshold])
        maximum = max(halved[threshold])

        return (minimum, maximum)

    def normalize(self) -> None:
        """Normalize the audio signal to have maximum amplitude 1.

        Args:
            None.

        Returns:
            None.

        """

        self.data = librosa.util.normalize(
            self.data
        )

    def reduce(self, **kwargs) -> None:
        """Apply noise reduction to the audio signal.

        Args:
            **kwargs: Keyword arguments to be passed to the
                `reduce_noise` function.

        Returns:
            None.

        """

        self.data = nr.reduce_noise(
            y=self.data,
            sr=self.rate,
            **kwargs
        )

    def dereverberate(self, settings: Settings) -> None:
        """Apply dereverberation to the audio signal.

        Args:
            settings: An object containing settings for the
                dereverberation algorithm.

        Returns:
            None

        """

        data = [self.data]
        y = np.stack(data)

        size = settings.size
        shift = settings.shift

        options = {
            'size': size,
            'shift': shift
        }

        y = stft(y, **options).transpose(2, 0, 1)

        taps = settings.taps
        delay = settings.delay
        iterations = settings.iterations

        x = wpe(
            y,
            taps=taps,
            delay=delay,
            iterations=iterations,
            statistics_mode='full'
        ).transpose(1, 2, 0)

        z = istft(
            x,
            size=size,
            shift=shift
        )

        self.data = z[0]

    def segment(self, onset: float, offset: float) -> Signal:
        """Get a segment of the signal between the onset and offset.

        Args:
            onset: The start time of the segment in seconds.
            offset: The end time of the segment in seconds.

        Returns:
            A new `Signal` representing the extracted segment.

        """

        data = self.data[
            int(onset * self.rate): int(offset * self.rate)
        ]

        data = data.astype('float32')

        buffer = io.BytesIO()

        wavfile.write(
            buffer,
            self.rate,
            data
        )

        return Signal(buffer)

    def save(self, path: str | pathlib.Path) -> None:
        """Save the audio signal to a .wav file.

        Args:
            path: The path to save the .wav file.

        Returns:
            None.

        """

        wavfile.write(
            path,
            self.rate,
            self.data
        )
