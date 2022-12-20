import numpy as np

from datatype.spectrogram import Segment
from scipy import ndimage


def dynamic_threshold_segmentation(signal, settings):
    # Make a copy of the original spectrogram
    segment = Segment(signal, settings)
    original = segment.generate()

    fft_rate = signal.rate / int(
        settings.hop_length_ms / 1000 * signal.rate
    )

    if settings.spectral_range is not None:
        spec_bin_hz = (signal.rate / 2) / np.shape(original)[0]

        original = original[
            int(settings.spectral_range[0] / spec_bin_hz):
            int(settings.spectral_range[1] / spec_bin_hz),
            :,
        ]

    # Loop through possible thresholding configurations starting at the highest
    for _, min_level_db in enumerate(
        np.arange(
            settings.min_level_db,
            settings.min_level_db_floor,
            settings.db_delta
        )
    ):
        segment.settings.min_level_db = min_level_db
        spectrogram = segment.generate()

        # Subtract the median
        spectrogram = spectrogram - np.median(spectrogram, axis=1).reshape(
            (len(spectrogram), 1)
        )

        spectrogram[spectrogram < 0] = 0

        # Get the vocal envelope
        vocal_envelope = np.max(spectrogram, axis=0) * np.sqrt(
            np.mean(spectrogram, axis=0)
        )

        # Normalize envelope
        vocal_envelope = vocal_envelope / np.max(vocal_envelope)

        # Look at how much silence exists in the signal
        onsets, offsets = onsets_offsets(
            vocal_envelope > settings.silence_threshold
        ) / fft_rate

    onset, offset = onsets_offsets(
        vocal_envelope > settings.silence_threshold
    ) / fft_rate

    # Threshold out short syllables
    mask = (offsets - onsets) >= settings.min_syllable_length_s

    return {
        'onset': onset[mask],
        'offset': offset[mask],
    }


def onsets_offsets(signal):
    elements, nelements = ndimage.label(signal)

    if nelements == 0:
        return np.array(
            [
                [0],
                [0]
            ]
        )

    onset, offset = np.array(
        [
            np.where(elements == element)[0][np.array([0, -1])] + np.array([0, 1])
            for element in np.unique(elements)
            if element != 0
        ]
    ).T

    return np.array([onset, offset])
