"""
Segmentation
------------

"""

from __future__ import annotations

import numpy as np

from datatype.spectrogram import Segment, Spectrogram
from scipy import ndimage
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy.typing as npt

    from datatype.settings import Settings
    from datatype.signal import Signal
    from typing_extensions import Any


class DynamicThresholdSegmentation:
    def __init__(self, signal: Signal | None = None, settings: Settings | None = None):
        self._component: dict[Any, Any] = {}
        self._settings = settings
        self._signal = signal

    @property
    def component(self) -> dict[Any, Any]:
        return self._component

    @component.setter
    def component(self, component: dict[Any, Any]) -> None:
        self._component = component

    @property
    def settings(self) -> Settings:
        return self._settings

    @settings.setter
    def settings(self, settings: Settings) -> None:
        self._settings = settings

    @property
    def signal(self) -> Signal:
        return self._signal

    @signal.setter
    def signal(self, signal: Signal) -> None:
        self._signal = signal

    def _calculate_onset_offset(self, signal: npt.NDArray) -> npt.NDArray:
        """Calculates the onsets and offsets of a signal.

        Args:
            signal: The input signal.

        Returns:
            An array containing the onsets and offsets.

        """

        signal = signal > self.settings.silence_threshold
        elements, nelements = ndimage.label(signal)

        zero = [0]

        if nelements == 0:
            return np.array(
                [zero, zero]
            )

        onset, offset = np.array(
            [
                np.where(elements == element)[0][np.array([0, -1])] +
                np.array([0, 1])
                for element in np.unique(elements)
                if element != 0
            ]
        ).T

        return np.array([onset, offset])

    def start(self) -> dict[str, Any]:
        """Performs dynamic threshold segmentation on a signal.

        Args:
            None.

        Returns:
            A dictionary containing the segmented template.

        """

        # Make a copy of the original spectrogram
        segment = Segment(self.signal, self.settings)

        spectrogram = Spectrogram()
        spectrogram.strategy = segment
        original = spectrogram.generate()

        fft = self.signal.rate / int(
            self.settings.hop_length_ms / 1000 * self.signal.rate
        )

        if self.settings.spectral_range is not None:
            x, _ = np.shape(original)

            resolution = (self.signal.rate / 2) / x
            lower, upper = self.settings.spectral_range

            original = original[
                int(lower / resolution):
                int(upper / resolution),
                :,
            ]

        # Possible thresholding configurations starting at the highest
        configuration = np.arange(
            self.settings.min_level_db,
            self.settings.min_level_db_floor,
            self.settings.db_delta
        )

        for _, min_level_db in enumerate(configuration):
            segment.settings.min_level_db = min_level_db
            sample = spectrogram.generate()

            # Subtract the median
            sample = sample - np.median(sample, axis=1).reshape(
                (len(sample), 1)
            )

            sample[sample < 0] = 0

            # Get the vocal envelope
            vocal_envelope = np.max(sample, axis=0) * np.sqrt(
                np.mean(sample, axis=0)
            )

            # Normalize envelope
            vocal_envelope = vocal_envelope / np.max(vocal_envelope)

            # Determine how much silence exists in the signal
            onsets, offsets = self._calculate_onset_offset(
                vocal_envelope > self.settings.silence_threshold
            ) / fft

        onset, offset = self._calculate_onset_offset(
            vocal_envelope > self.settings.silence_threshold
        ) / fft

        # Threshold out short syllables
        mask = (offsets - onsets) >= self.settings.min_syllable_length_s
        vocal_envelope = vocal_envelope.astype('float32')

        self.component['onset'] = onset[mask]
        self.component['offset'] = offset[mask]
        self.component['spectrogram'] = sample
        self.component['vocal_envelope'] = vocal_envelope

        return self
