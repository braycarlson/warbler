import librosa
import numpy as np

from PIL import Image
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


class Spectrogram:
    def __init__(self, signal, settings):
        self._data = None
        self._signal = signal
        self._settings = settings

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    def generate(self):
        self._spectrogram_nn()
        self._normalize()
        # self.resize()

        if self._settings.mask_spec:
            self.mask()

        self.normalize()
        return self.data

    def mask(self, threshold=0.9, offset=1e-10):
        mask = self.data >= (self.data.max(axis=0, keepdims=1) * threshold + offset)
        self.data = self.data * mask

    def normalize(self):
        minimum = np.min(self.data)
        maximum = np.max(self.data)

        data = (self.data - minimum) / (maximum - minimum)
        self.data = (data * 255).astype('uint8')

    def resize(self, scaling=10):
        x, y = np.shape(self.data)

        shape = [
            int(np.log(y) * scaling),
            x
        ]

        self.data = np.array(
            Image.fromarray(self.data).resize(
                    shape,
                    Image.ANTIALIAS
                )
            )

    def _spectrogram_nn(self):
        y = self._preemphasis()

        matrix = self._stft(y)
        magnitude = np.abs(matrix)

        self.data = (
            self._amplitude_to_decibel(magnitude) -
            self._settings.ref_level_db
        )

    def _preemphasis(self):
        return lfilter(
            [1, -self._settings.preemphasis],
            [1],
            self._signal.data
        )

    def _stft(self, y):
        hop_length = int(
            self._settings.hop_length_ms / 1000 * self._signal.rate
        )

        win_length = int(
            self._settings.win_length_ms / 1000 * self._signal.rate
        )

        return librosa.stft(
            y=y,
            n_fft=self._settings.n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )

    def _normalize(self):
        self.data = np.clip(
            (self.data - self._settings.min_level_db) / -self._settings.min_level_db,
            0,
            1
        )

    @staticmethod
    def _amplitude_to_decibel(magnitude):
        return 20 * np.log10(
            np.maximum(1e-5, magnitude)
        )
