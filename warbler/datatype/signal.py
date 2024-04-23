from __future__ import annotations

import base64
import io
import librosa
import noisereduce as nr
import numpy as np
import scipy
import scipy.signal
import soundfile as sf
import struct

from scipy.signal import windows

if not hasattr(scipy.signal, 'blackman'):
    scipy.signal.blackman = windows.blackman

from abc import abstractmethod, ABC
from nara_wpe.wpe import wpe
from nara_wpe.utils import istft, stft
from pathlib import Path
from scipy.io import wavfile
from scipy.signal import butter, lfilter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt

    from typing_extensions import BinaryIO, Self
    from warbler.datatype.settings import Settings


class Strategy(ABC):
    @abstractmethod
    def load(
        self,
        path: str | BinaryIO | Path | None
    ) -> tuple[int, npt.NDArray]:
        raise NotImplementedError


class Librosa(Strategy):
    def load(
        self,
        path: str | BinaryIO | Path | None
    ) -> tuple[int, npt.NDArray]:
        if isinstance(path, Path):
            path = str(path)

        data, rate = librosa.core.load(path, sr=None)
        return rate, data


class Scipy(Strategy):
    def load(
        self,
        path: str | BinaryIO | Path | None
    ) -> tuple[int, npt.NDArray]:
        rate, data = wavfile.read(path)
        return rate, data


class Soundfile(Strategy):
    def load(
        self,
        path: str | BinaryIO | Path | None
    ) -> tuple[int, npt.NDArray]:
        if isinstance(path, Path):
            path = str(path)

        data, rate = sf.read(path)
        return rate, data


class Signal:
    def __init__(self, path: str | BinaryIO | Path | None = None):
        self.strategy = Scipy()
        self.path = path

        if path is not None:
            self.rate, self.data = self._load()

    def __len__(self) -> int:
        return len(self.data)

    def _load(self) -> tuple[int, npt.NDArray]:
        if isinstance(self.path, Path) and not self.path.exists():
            message = f"{self.path} was not found"
            raise FileNotFoundError(message)

        rate, data = self.strategy.load(self.path)

        if np.issubdtype(data.dtype, np.integer):
            data = np.array(data / 32768.0).astype('float32')

        return rate, data

    @classmethod
    def deserialize(cls: type[Self], data: str) -> Signal:
        data_bytes = base64.b64decode(data)
        buffer = io.BytesIO(data_bytes)

        length = buffer.read(4)
        length = struct.unpack('I', length)[0]
        path = buffer.read(length).decode('utf-8')

        rate = buffer.read(4)
        rate = struct.unpack('I', rate)[0]

        length = buffer.read(4)
        length = struct.unpack('I', length)[0]

        data = f'{length}f'
        samples = buffer.read(4 * length)
        data = struct.unpack(data, samples)

        data = np.array(data, dtype='float32')

        signal = cls()
        signal.path = path
        signal.rate = rate
        signal.data = data
        return signal

    @property
    def duration(self) -> float:
        self._duration = len(self.data) / float(self.rate)
        return self._duration

    def filter(self, lowcut: float, highcut: float, order: int = 5) -> None:
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
        timestep = 1.0 / self.rate

        time = np.arange(
            0,
            self.duration,
            timestep
        )

        fft = abs(
            scipy.fft.fft(self.data)
        )

        bisection = fft[
            range(self.data.size // 2)
        ]

        bisection = abs(bisection)

        frequencies = scipy.fftpack.fftfreq(
            self.data.size,
            time[1] - time[0]
        )

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

    def mean(self) -> npt.NDArray:
        fft = scipy.fft.fft(self.data)

        frequencies = scipy.fft.fftfreq(
            len(self.data),
            1 / self.rate
        )

        mask = frequencies > 0
        frequencies = frequencies[mask]
        power = np.abs(fft[mask])**2

        return np.sum(frequencies * power) / np.sum(power)

    def normalize(self) -> None:
        self.data = librosa.util.normalize(
            self.data
        )

    def reduce(self, **kwargs) -> None:
        self.data = nr.reduce_noise(
            y=self.data,
            sr=self.rate,
            **kwargs
        )

    def dereverberate(self, settings: Settings) -> None:
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
        data = self.data[
            int(onset * self.rate):
            int(offset * self.rate)
        ]

        data = data.astype('float32')

        buffer = io.BytesIO()

        wavfile.write(
            buffer,
            self.rate,
            data
        )

        return Signal(buffer)

    def save(self, path: str | Path) -> None:
        wavfile.write(
            path,
            self.rate,
            self.data
        )

    def serialize(self) -> None:
        buffer = io.BytesIO()

        path = str(self.path).encode('utf-8')
        length = len(path)

        data = struct.pack('I', length)
        buffer.write(data)
        buffer.write(path)

        data  = struct.pack('I', self.rate)
        buffer.write(data)

        length = len(self.data)
        data = struct.pack('I', length)
        buffer.write(data)

        data = f'{length}f'
        data = struct.pack(data, *self.data)
        buffer.write(data)

        buffer = buffer.getvalue()
        return base64.b64encode(buffer).decode('utf-8')
