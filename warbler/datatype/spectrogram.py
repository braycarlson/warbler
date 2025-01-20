from __future__ import annotations

import librosa
import numpy as np
import os
import tensorflow as tf

from abc import ABC, abstractmethod
from librosa import mel_frequencies
from PIL import Image
from scipy.signal import lfilter
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt

    from typing_extensions import Self
    from warbler.datatype.settings import Settings
    from warbler.datatype.signal import Signal


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def compress(spectrogram: npt.NDArray) -> npt.NDArray:
    minimum = np.min(spectrogram)
    maximum = np.max(spectrogram)

    spectrogram = (spectrogram - minimum) / (maximum - minimum)

    # return spectrogram
    return (spectrogram * 255).astype('uint8')


def create_spectrogram(
    signal: Signal,
    settings: Settings,
    matrix: npt.NDArray | None = None
) -> npt.NDArray:
    spectrogram = Spectrogram()
    strategy = Linear(signal, settings, matrix)
    spectrogram.strategy = strategy

    return spectrogram.generate()


def flatten(spectrogram: npt.NDArray) -> npt.NDArray:
    return np.reshape(
        spectrogram,
        (
            np.shape(spectrogram)[0],
            np.prod(
                np.shape(spectrogram)[1:]
            )
        )
    )


def mel_matrix(settings: Settings) -> npt.NDArray:
    # https://stackoverflow.com/questions/54047654/tensorflow-different-results-with-the-same-random-seed

    # Create a filter to convolve with the spectrogram
    mel_matrix = tf.signal.linear_to_mel_weight_matrix(
        num_mel_bins=settings.num_mel_bins,
        num_spectrogram_bins=int(settings.n_fft / 2) + 1,
        sample_rate=settings.sample_rate,
        lower_edge_hertz=settings.mel_lower_edge_hertz,
        upper_edge_hertz=settings.mel_upper_edge_hertz,
        dtype=tf.dtypes.float32,
        name=None
    )

    # Get the center frequencies of mel bands
    mel_f = mel_frequencies(
        n_mels=settings.num_mel_bins + 2,
        fmin=settings.mel_lower_edge_hertz,
        fmax=settings.mel_upper_edge_hertz
    )

    # Slaney-style is scaled to be approximately constant energy per channel
    enorm = tf.dtypes.cast(
        tf.expand_dims(
            tf.constant(
                2.0 / (
                    mel_f[2: settings.num_mel_bins + 2] -
                    mel_f[: settings.num_mel_bins]
                )
            ),
            0,
        ),
        tf.float32
    )

    mel_matrix = tf.multiply(mel_matrix, enorm)

    mel_matrix = tf.divide(
        mel_matrix,
        tf.reduce_sum(mel_matrix, axis=0)
    )

    return mel_matrix.numpy()


def pad(spectrogram: npt.NDArray, padding: int) -> npt.NDArray:
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


def resize(spectrogram: npt.NDArray, factor: int = 10) -> npt.NDArray:
    x, y = np.shape(spectrogram)
    shape = [int(np.log(y) * factor), x]

    return np.array(
        Image
        .fromarray(spectrogram)
        .resize(shape, Image.Resampling.LANCZOS)
    )


class SpectrogramStrategy(ABC):
    def __init__(
        self,
        signal: Signal | None = None,
        settings: Settings | None = None,
        matrix: npt.NDArray | None = None,
        is_normalize: bool = True
    ):
        self.data = None
        self.matrix = matrix
        self.is_normalize = is_normalize
        self.signal = signal
        self.settings = settings

    def build(self) -> Self:
        if self.matrix is not None:
            self.data = np.dot(
                self.data.T,
                self.matrix
            ).T

        return self

    def decibel_to_amplitude(self) -> Self:
        self.data = np.power(
            10.0,
            self.data * 0.05
        )

        return self

    def denormalize(self) -> Self:
        self.data = (
            np.clip(self.data, 0, 1) * -self.settings.min_level_db
        ) + self.settings.min_level_db

        return self

    def normalize(self) -> Self:
        if self.is_normalize:
            self.data = np.clip(
                (self.data - self.settings.min_level_db) /
                -self.settings.min_level_db,
                0,
                1
            )

        return self

    def preemphasis(self) -> Self:
        self.data = lfilter(
            [1, -self.settings.preemphasis], [1], self.signal.data
        )

        return self

    def inverse_preemphasis(self) -> Self:
        self.data = lfilter(
            [1],
            [1, -self.settings.preemphasis],
            self.signal.data
        )

        return self

    def stft(self) -> Self:
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

        return self

    def _istft(self) -> Self:
        y = self.data
        hop_length = int(self.settings.hop_length_ms / 1000 * self.signal.rate)
        win_length = int(self.settings.win_length_ms / 1000 * self.signal.rate)

        self.data = librosa.istft(
            y,
            hop_length=hop_length,
            win_length=win_length
        )

        return self

    @abstractmethod
    def amplitude_to_decibel(self) -> Self:
        raise NotImplementedError


class Linear(SpectrogramStrategy):
    def __init__(
        self,
        signal: Signal | None = None,
        settings: Settings | None = None,
        matrix: npt.NDArray | None = None,
        is_normalize: bool = True
    ):
        super().__init__(
            signal,
            settings,
            matrix,
            is_normalize
        )

    def amplitude_to_decibel(self) -> Self:
        data = np.abs(self.data)

        self.data = 20 * np.log10(
            np.maximum(1e-5, data)
        ) - self.settings.ref_level_db

        return self


class Mel(SpectrogramStrategy):
    def __init__(
        self,
        signal: Signal | None = None,
        settings: Settings | None = None,
        matrix: npt.NDArray | None = None,
        is_normalize: bool = True
    ):
        super().__init__(
            signal,
            settings,
            matrix,
            is_normalize
        )

    def _create_basis(self) -> npt.NDArray:
        basis = librosa.filters.mel(
            sr=self.signal.rate,
            n_fft=self.settings.n_fft,
            n_mels=self.settings.num_mel_bins,
            fmin=self.settings.mel_lower_edge_hertz,
            fmax=self.settings.mel_upper_edge_hertz,
        )

        basis = basis.T / np.sum(basis, axis=1)
        return np.nan_to_num(basis).T

    def _linear_to_mel(self, basis: npt.NDArray) -> npt.NDArray:
        basis = self._create_basis()
        return np.dot(basis, self.data)

    def amplitude_to_decibel(self) -> Self:
        self.data = np.abs(self.data)
        self.data = self._linear_to_mel(self.data)

        self.data = 20 * np.log10(
            np.maximum(1e-5, self.data)
        ) - self.settings.ref_level_db

        return self


class Segment(SpectrogramStrategy):
    def __init__(
        self,
        signal: Signal | None = None,
        settings: Settings | None = None,
        matrix: npt.NDArray | None = None,
        is_normalize: bool = True
    ):
        super().__init__(
            signal,
            settings,
            matrix,
            is_normalize
        )

    def amplitude_to_decibel(self) -> Self:
        data = np.abs(self.data)

        self.data = 20 * np.log10(
            np.maximum(1e-5, data)
        ) - self.settings.ref_level_db

        return self


class Spectrogram:
    def __init__(self, strategy : SpectrogramStrategy | None = None):
        self.strategy = strategy

    @property
    def matrix(self) -> npt.NDArray:
        return self.strategy.matrix

    @matrix.setter
    def matrix(self, matrix: npt.NDArray) -> None:
        self.strategy.matrix = matrix

    @property
    def normalize(self) -> bool:
        return self.strategy.normalize

    @normalize.setter
    def normalize(self, normalize: bool) -> None:
        self.strategy.normalize = normalize

    @property
    def signal(self) -> Signal:
        return self.strategy.signal

    @signal.setter
    def signal(self, signal: Signal) -> None:
        self.strategy.signal = signal

    @property
    def settings(self) -> Settings:
        return self.strategy.settings

    @settings.setter
    def settings(self, settings: Settings) -> None:
        self.strategy.settings = settings

    def generate(self) -> npt.NDArray:
        return (
            self.strategy
            .preemphasis()
            .stft()
            .amplitude_to_decibel()
            .normalize()
            .build()
            .data
        )
