import librosa
import numpy as np

from abc import ABC, abstractmethod
from scipy.signal import lfilter


def flatten(spectrogram):
    return np.reshape(
        spectrogram,
        (
            np.shape(spectrogram)[0],
            np.prod(
                np.shape(spectrogram)[1:]
            )
        )
    )


def pad(spectrogram, padding):
    _, y = np.shape(spectrogram)

    length = padding - y
    left = np.floor(float(length) / 2).astype('int')
    right = np.ceil(float(length) / 2).astype('int')

    return np.pad(
        spectrogram,
        [
            (0, 0),
            (left, right)
        ],
        'constant',
        constant_values=0
    )


class Strategy(ABC):
    def __init__(self, signal, settings):
        self._data = None
        self.signal = signal
        self.settings = settings

    def decibel_to_amplitude(self):
        self.data = np.power(
            10.0,
            self.data * 0.05
        )

    def denormalize(self):
        self.data = (
            np.clip(self.data, 0, 1) * -self.settings.min_level_db
        ) + self.settings.min_level_db

    def normalize(self):
        self.data = np.clip(
            (self.data - self.settings.min_level_db) / -self.settings.min_level_db,
            0,
            1
        )

    def reduce(self):
        minimum = np.min(self.data)
        maximum = np.max(self.data)

        data = (self.data - minimum) / (maximum - minimum)
        self.data = (data * 255).astype('uint8')

    def preemphasis(self):
        self.data = lfilter(
            [1, -self.settings.preemphasis], [1], self.signal.data
        )

    def inverse_preemphasis(self):
        self.data = lfilter(
            [1],
            [1, -self.settings.preemphasis],
            self.signal.data
        )

    def stft(self):
        y = self.data
        n_fft = self.settings.n_fft
        hop_length = int(self.settings.hop_length_ms / 1000 * self.signal.rate)
        win_length = int(self.settings.win_length_ms / 1000 * self.signal.rate)

        self.data = librosa.stft(
            y=y,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )

    def _istft(self):
        y = self.data
        hop_length = int(self.settings.hop_length_ms / 1000 * self.signal.rate)
        win_length = int(self.settings.win_length_ms / 1000 * self.signal.rate)

        self.data = librosa.istft(
            y,
            hop_length=hop_length,
            win_length=win_length
        )

    @abstractmethod
    def amplitude_to_decibel(self):
        pass

    @abstractmethod
    def generate(self, normalize=True):
        pass


class Linear(Strategy):
    def __init__(self, signal, settings):
        self.data = None
        self.signal = signal
        self.settings = settings

    def amplitude_to_decibel(self):
        data = np.abs(self.data)

        self.data = 20 * np.log10(
            np.maximum(1e-5, data)
        ) - self.settings.ref_level_db

    def generate(self, normalize=True):
        self.preemphasis()
        self.stft()
        self.amplitude_to_decibel()

        if normalize is True:
            self.normalize()
            self.reduce()

        return self.data


class Mel(Strategy):
    def __init__(self, signal, settings):
        self.data = None
        self.signal = signal
        self.settings = settings

    def amplitude_to_decibel(self):
        self.data = np.abs(self.data)
        self.data = self._linear_to_mel(self.data)

        self.data = 20 * np.log10(
            np.maximum(1e-5, self.data)
        ) - self.settings.ref_level_db

    def create_basis(self):
        basis = librosa.filters.mel(
            sr=self.signal.rate,
            n_fft=self.settings.n_fft,
            n_mels=self.settings.num_mel_bins,
            fmin=self.settings.mel_lower_edge_hertz,
            fmax=self.settings.mel_upper_edge_hertz,
        )

        basis = basis.T / np.sum(basis, axis=1)
        return np.nan_to_num(basis).T

    def _linear_to_mel(self, basis):
        basis = self.create_basis()
        return np.dot(basis, self.data)

    def generate(self, normalize=True):
        self.preemphasis()
        self.stft()
        self.amplitude_to_decibel()

        if normalize is True:
            self.normalize()
            self.reduce()

        return self.data


class Segment(Strategy):
    def __init__(self, signal, settings):
        self.data = None
        self.signal = signal
        self.settings = settings

    def amplitude_to_decibel(self):
        data = np.abs(self.data)

        self.data = 20 * np.log10(
            np.maximum(1e-5, data)
        ) - self.settings.ref_level_db

    def generate(self, normalize=True):
        self.preemphasis()
        self.stft()
        self.amplitude_to_decibel()

        if normalize is True:
            self.normalize()
            self.reduce()

        return self.data


class Spectrogram:
    def __init__(self):
        self._strategy = None

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy):
        self._strategy = strategy

    def generate(self, normalize=True):
        return self.strategy.generate(normalize=normalize)
