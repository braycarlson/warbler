import librosa
import numpy as np

from abc import ABC, abstractmethod
from scipy.signal import lfilter


class Spectrogram(ABC):
    def __init__(self, signal, parameter):
        self.signal = signal
        self.parameter = parameter

    def generate(self):
        spectrogram = self.nn()
        return self.normalize(spectrogram)

    def stft(self):
        return librosa.stft(
            y=self.signal.data,
            n_fft=self.parameter.n_fft,
            hop_length=int(
                self.parameter.hop_length_ms / 1000 * self.signal.rate
            ),
            win_length=int(
                self.parameter.win_length_ms / 1000 * self.signal.rate
            ),
        )

    @staticmethod
    def amplitude_to_decibel(amplitude):
        return 20 * np.log10(
            np.maximum(1e-5, amplitude)
        )

    @staticmethod
    def decibel_to_amplitude(decibel):
        return np.power(
            10.0,
            decibel * 0.05
        )

    @staticmethod
    def linear_to_mel(spectrogram, basis):
        return np.dot(basis, spectrogram)

    def normalize(self, spectrogram):
        return np.clip(
            (spectrogram - self.parameter.min_level_db) / -self.parameter.min_level_db,
            0,
            1
        )

    def denormalize(self, spectrogram):
        return (
            np.clip(spectrogram, 0, 1) * -self.parameter.min_level_db
        ) + self.parameter.min_level_db

    def preemphasis(self):
        return lfilter(
            [1, -self.parameter.preemphasis],
            [1],
            self.signal.data
        )

    @abstractmethod
    def nn(self):
        pass


class Linear(Spectrogram):
    def nn(self):
        D = self.stft()

        S = self.amplitude_to_decibel(
            np.abs(D)
        ) - self.parameter.ref_level_db

        return S


class Mel(Spectrogram):
    def build(self):
        basis = librosa.filters.mel(
            sr=self.signal.rate,
            n_fft=self.parameter.n_fft,
            n_mels=self.parameter.num_mel_bins,
            fmin=self.parameter.mel_lower_edge_hertz,
            fmax=self.parameter.mel_upper_edge_hertz,
        )

        return np.nan_to_num(
            basis.T / np.sum(basis, axis=1)
        ).T

    def nn(self):
        D = self.stft()
        basis = self.build()

        S = self.amplitude_to_decibel(
            self.linear_to_mel(
                np.abs(D),
                basis
            )
        ) - self.parameter.ref_level_db

        return S
