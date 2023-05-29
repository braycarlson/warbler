"""
Spectrogram
-----------

"""

from __future__ import annotations

import librosa
import numpy as np
import os
import tensorflow as tf

from abc import ABC, abstractmethod
from librosa import mel_frequencies
from PIL import Image
from scipy.signal import lfilter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt

    from datatype.settings import Settings
    from datatype.signal import Signal
    from typing_extensions import Self


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def compress(spectrogram: npt.NDArray) -> npt.NDArray:
    """Compresses the spectrogram by normalizing it.

    The values will be normalized between 0 and 1, and then
    scaling them to the range [0, 255].

    Args:
        spectrogram: The input spectrogram.

    Returns:
        The compressed spectrogram.

    """

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
    """Creates a spectrogram from the signal, settings and matrix.

    Args:
        signal The input audio signal.
        settings: The settings for spectrogram creation.
        matrix: The matrix to be applied to the spectrogram.

    Returns:
        The generated spectrogram.

    """

    spectrogram = Spectrogram()
    strategy = Linear(signal, settings, matrix)
    spectrogram.strategy = strategy

    return spectrogram.generate()


def flatten(spectrogram: npt.NDArray) -> npt.NDArray:
    """Flattens the spectrogram by reshaping it into a 2D array.

    Args:
        spectrogram: The input spectrogram.

    Returns:
        The flattened spectrogram.

    """

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
    """Computes the mel filterbank matrix based on custom settings.

    Args:
        settings: The settings for mel matrix computation.

    Returns:
        The mel filterbank matrix.

    """

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
    """Pads the spectrogram with zeros to the specified padding size.

    Args:
        spectrogram: The input spectrogram.
        padding: The padding size.

    Returns:
        The padded spectrogram.

    """

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
    """Resizes the spectrogram by applying a scaling factor.

    Args:
        spectrogram: The input spectrogram.
        factor: The scaling factor.

    Returns:
        The resized spectrogram.

    """

    x, y = np.shape(spectrogram)
    shape = [int(np.log(y) * factor), x]

    return np.array(
        Image
        .fromarray(spectrogram)
        .resize(shape, Image.Resampling.LANCZOS)
    )


class Strategy(ABC):
    """Abstract base class representing a strategy for generating spectrograms.

    Args:
        signal: `Signal` object representing the audio signal.
        settings: `Settings` object containing configuration parameters.
        matrix: Transformation matrix.
        normalize: A flag indicating whether to normalize the spectrogram.

    Attributes:
        _data: The spectrogram data
        _matrix: Transformation matrix.
        _normalize: A flag indicating whether to normalize the spectrogram.
        _signal: `Signal` object representing the audio signal.
        _settings: `Settings` object containing configuration parameters.

    """

    def __init__(
        self,
        signal: Signal | None = None,
        settings: Settings | None = None,
        matrix: npt.NDArray | None = None,
        normalize: bool = True
    ):
        self._data = None
        self._matrix = matrix
        self._normalize = normalize
        self._signal = signal
        self._settings = settings

    @property
    def matrix(self) -> npt.NDArray:
        """Get the transformation matrix used for the strategy.

        Returns:
            The transformation matrix.

        """

        return self._matrix

    @matrix.setter
    def matrix(self, matrix: npt.NDArray) -> None:
        self._matrix = matrix

    @property
    def normalize(self) -> bool:
        """Determine if normalization is enabled for the strategy.

        Returns:
            True if normalization is enabled, False otherwise.

        """

        return self._normalize

    @normalize.setter
    def normalize(self, normalize: bool) -> None:
        self._normalize = normalize

    @property
    def signal(self) -> Signal:
        """Get the Signal object associated with the strategy.

        Returns:
            Signal: The Signal object.

        """

        return self._signal

    @signal.setter
    def signal(self, signal: Signal) -> None:
        self._signal = signal

    @property
    def settings(self) -> Settings:
        """Get the Settings object associated with the strategy.

        Returns:
            Settings: The Settings object.

        """

        return self._settings

    @settings.setter
    def settings(self, settings: Settings) -> None:
        self._settings = settings

    def build(self) -> Self:
        """Build the spectrogram using the specified strategy.

        Returns:
            The modified Strategy instance.

        """

        if self.matrix is not None:
            self.data = np.dot(
                self.data.T,
                self.matrix
            ).T

        return self

    def decibel_to_amplitude(self) -> Self:
        """Convert the spectrogram from decibel to amplitude scale.

        Args:
            None.

        Returns:
            The modified Strategy instance.

        """

        self.data = np.power(
            10.0,
            self.data * 0.05
        )

        return self

    def denormalize(self) -> Self:
        """Denormalize the spectrogram.

        Args:
            None.

        Returns:
            The modified Strategy instance.

        """

        self.data = (
            np.clip(self.data, 0, 1) * -self.settings.min_level_db
        ) + self.settings.min_level_db

        return self

    def normalize(self) -> Self:
        """Normalize the spectrogram.

        Args:
            None.

        Returns:
            The modified Strategy instance.

        """

        if self.normalize:
            self.data = np.clip(
                (self.data - self.settings.min_level_db) /
                -self.settings.min_level_db,
                0,
                1
            )

        return self

    def preemphasis(self) -> Self:
        """Apply a preemphasis filter to the audio signal.

        Args:
            None.

        Returns:
            The modified Strategy instance.

        """

        self.data = lfilter(
            [1, -self.settings.preemphasis], [1], self.signal.data
        )

        return self

    def inverse_preemphasis(self) -> Self:
        """Apply an inverse preemphasis filter to the audio signal.

        Args:
            None.

        Returns:
            The modified Strategy instance.

        """

        self.data = lfilter(
            [1],
            [1, -self.settings.preemphasis],
            self.signal.data
        )

        return self

    def stft(self) -> Self:
        """Compute the Short-Time Fourier Transform of the audio signal.

        Args:
            None.

        Returns:
            The modified Strategy instance.

        """

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
        """Compute the inverse Short-Time Fourier Transform of the spectrogram.

        Args:
            None.

        Returns:
            The modified Strategy instance.

        """

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
        """Convert the spectrogram from amplitude to decibel scale."""

        raise NotImplementedError


class Linear(Strategy):
    """Linear strategy for generating spectrograms.

    This strategy applies linear transformation to the input data and
    converts it to decibel scale.

    Args:
        *args: Variable length arguments passed to the parent class.
        **kwargs: Keyword arguments passed to the parent class.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def amplitude_to_decibel(self) -> Self:
        """Convert amplitude values to decibel scale.

        Returns:
            The modified Linear instance.

        """

        data = np.abs(self.data)

        self.data = 20 * np.log10(
            np.maximum(1e-5, data)
        ) - self.settings.ref_level_db

        return self


class Mel(Strategy):
    """Mel strategy for generating spectrograms.

    This strategy applies Mel transformation to the input data and converts it
    to decibel scale.

    Args:
        *args: Variable length arguments passed to the parent class.
        **kwargs: Keyword arguments passed to the parent class.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_basis(self) -> npt.NDArray:
        """Create the Mel basis matrix.

        Returns:
            The Mel basis matrix.

        """

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
        """Convert linear spectrogram to Mel spectrogram.

        Args:
            basis: The Mel basis matrix.

        Returns:
            The Mel spectrogram.

        """

        basis = self._create_basis()
        return np.dot(basis, self.data)

    def amplitude_to_decibel(self) -> Self:
        """Convert amplitude values to decibel scale using Mel transformation.

        Args:
            None.

        Returns:
            The modified Mel instance.

        """

        self.data = np.abs(self.data)
        self.data = self._linear_to_mel(self.data)

        self.data = 20 * np.log10(
            np.maximum(1e-5, self.data)
        ) - self.settings.ref_level_db

        return self


class Segment(Strategy):
    """Segment strategy for generating spectrograms.

    This strategy applies segment-wise processing to the input data and
    converts it to decibel scale.

    Args:
        *args: Variable length arguments passed to the parent class.
        **kwargs: Keyword arguments passed to the parent class.

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def amplitude_to_decibel(self) -> Self:
        """Convert amplitude values to decibel scale.

        Args:
            None.

        Returns:
            The modified Segment instance.

        """

        data = np.abs(self.data)

        self.data = 20 * np.log10(
            np.maximum(1e-5, data)
        ) - self.settings.ref_level_db

        return self


class Spectrogram:
    """Spectrogram generator.

    This class provides an interface for generating spectrograms using
    different strategies.

    Args:
        strategy: The strategy to be used for spectrogram generation.

    """

    def __init__(self, strategy : Strategy | None = None):
        self._strategy = strategy

    @property
    def matrix(self) -> npt.NDArray:
        """Get the Mel spectrogram matrix.

        Returns:
            The spectrogram matrix.

        """

        return self.strategy.matrix

    @matrix.setter
    def matrix(self, matrix: npt.NDArray) -> None:
        self.strategy.matrix = matrix

    @property
    def normalize(self) -> bool:
        """Get the normalization flag.

        Returns:
            True if the spectrogram is normalized, False otherwise.

        """

        return self.strategy.normalize

    @normalize.setter
    def normalize(self, normalize: bool) -> None:
        self.strategy.normalize = normalize

    @property
    def signal(self) -> Signal:
        """Get the Signal object.

        Returns:
            The Signal object.

        """

        return self.strategy.signal

    @signal.setter
    def signal(self, signal: Signal) -> None:
        self.strategy.signal = signal

    @property
    def settings(self) -> Settings:
        """Get the Settings object.

        Returns:
            The Settings object.

        """

        return self.strategy.settings

    @settings.setter
    def settings(self, settings: Settings) -> None:
        self.strategy.settings = settings

    @property
    def strategy(self) -> Strategy:
        """Get the Strategy object.

        Returns:
            The Strategy object.

        """

        return self._strategy

    @strategy.setter
    def strategy(self, strategy: Strategy) -> None:
        self._strategy = strategy

    def generate(self) -> npt.NDArray:
        """Generate the spectrogram.

        Args:
            None

        Returns:
            The generated spectrogram.

        """

        return (
            self.strategy
            .preemphasis()
            .stft()
            .amplitude_to_decibel()
            .normalize()
            .build()
            .data
        )
