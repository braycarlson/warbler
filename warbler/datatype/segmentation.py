from __future__ import annotations

import numpy as np

from scipy import ndimage
from typing_extensions import TYPE_CHECKING
from warbler.datatype.spectrogram import Segment, Spectrogram

if TYPE_CHECKING:
    import numpy.typing as npt

    from typing_extensions import Any
    from warbler.datatype.settings import Settings
    from warbler.datatype.signal import Signal


class DynamicThresholdSegmentation:
    def __init__(
        self,
        signal: Signal | None = None,
        settings: Settings | None = None
    ):
        self.component: dict[Any, Any] = {}
        self.settings = settings
        self.signal = signal

    def _calculate_onset_offset(self, signal: npt.NDArray) -> npt.NDArray:
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
