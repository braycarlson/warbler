import librosa
import numpy as np

from scipy.signal import lfilter


class Spectrogram:
    def __init__(self, signal, parameters):
        self._signal = signal
        self._parameters = parameters

    @property
    def data(self):
        spectrogram = self.spectrogram_nn()
        self._data = self._normalize(spectrogram)
        return self._data

    def spectrogram_nn(self):
        y = self.preemphasis()

        matrix = self._stft(y)
        magnitude = np.abs(matrix)

        spectrogram = (
            self._amplitude_to_decibel(magnitude) -
            self._parameters.ref_level_db
        )

        return spectrogram

    def preemphasis(self):
        return lfilter(
            [1, -self._parameters.preemphasis],
            [1],
            self._signal.data
        )

    def _stft(self, y):
        hop_length = int(
            self._parameters.hop_length_ms / 1000 * self._signal.rate
        )

        win_length = int(
            self._parameters.win_length_ms / 1000 * self._signal.rate
        )

        return librosa.stft(
            y=y,
            n_fft=self._parameters.n_fft,
            hop_length=hop_length,
            win_length=win_length,
        )

    def _normalize(self, spectrogram):
        return np.clip(
            (spectrogram - self._parameters.min_level_db) / -self._parameters.min_level_db, 0, 1
        )

    @staticmethod
    def _amplitude_to_decibel(magnitude):
        return 20 * np.log10(
            np.maximum(1e-5, magnitude)
        )
