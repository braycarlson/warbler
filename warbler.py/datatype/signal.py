import io
import librosa
import noisereduce as nr
import numpy as np
import pathlib
import soundfile as sf

from abc import abstractmethod, ABC
from nara_wpe.wpe import wpe
from nara_wpe.utils import istft, stft
from scipy.io import wavfile
from scipy.signal import butter, lfilter


class Strategy(ABC):
    @abstractmethod
    def load(self, path):
        pass


class Librosa(Strategy):
    def load(self, path):
        if isinstance(path, pathlib.Path):
            path = str(path)

        data, rate = librosa.core.load(path)
        return rate, data


class Scipy(Strategy):
    def load(self, path):
        rate, data = wavfile.read(path)
        return rate, data


class Soundfile(Strategy):
    def load(self, path):
        if isinstance(path, pathlib.Path):
            path = str(path)

        data, rate = sf.read(path)
        return rate, data


class Signal:
    def __init__(self, path):
        self._strategy = Scipy()
        self._path = path
        self._rate, self._data = self._load()

    def _load(self):
        if isinstance(self.path, pathlib.Path) and not self.path.exists():
            raise FileNotFoundError(f"{self.path} was not found")

        rate, data = self.strategy.load(self.path)

        if np.issubdtype(data.dtype, np.int16):
            data = np.array(data / 32768.0).astype('float32')

        return rate, data

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy):
        self._strategy = strategy

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def path(self):
        return self._path

    @property
    def rate(self):
        return self._rate

    @property
    def duration(self):
        self._duration = len(self.data) / float(self.rate)
        return self._duration

    def filter(self, lowcut, highcut, order=5):
        '''
        https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
        '''

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

    def normalize(self):
        self.data = librosa.util.normalize(
            self.data
        )

    def reduce(self, **kwargs):
        self.data = nr.reduce_noise(
            y=self.data,
            sr=self.rate,
            **kwargs
        )

    def dereverberate(self, settings):
        '''
        https://github.com/fgnt/nara_wpe/blob/master/examples/WPE_Numpy_offline.ipynb
        '''

        data = [self.data]
        y = np.stack(data)

        size = settings.size
        shift = settings.shift

        options = {
            'size': size,
            'shift': shift
        }

        Y = stft(y, **options).transpose(2, 0, 1)

        taps = settings.taps
        delay = settings.delay
        iterations = settings.iterations

        Z = wpe(
            Y,
            taps=taps,
            delay=delay,
            iterations=iterations,
            statistics_mode='full'
        ).transpose(1, 2, 0)

        z = istft(
            Z,
            size=size,
            shift=shift
        )

        self.data = z[0]

    def segment(self, onset, offset):
        data = self.data[
            int(onset * self.rate): int(offset * self.rate)
        ]

        buffer = io.BytesIO()

        wavfile.write(
            buffer,
            self.rate,
            data
        )

        return Signal(buffer)

    def save(self, path):
        wavfile.write(
            path,
            self.rate,
            self.data
        )
