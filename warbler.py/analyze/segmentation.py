import matplotlib.pyplot as plt

from constant import DATASET, SETTINGS
from datatype.axes import SpectrogramAxes
from datatype.settings import Settings
from datatype.signal import Signal
from datatype.spectrogram import Linear, Mel, Spectrogram
from plot import (
    SegmentationSpectrogram,
    VocalEnvelopeSpectrogram
)
from vocalseg.dynamic_thresholding import dynamic_threshold_segmentation


def main():
    # path = DATASET.joinpath('DbWY_STE2017/segmentation/STE01_DbWY2017.json')
    # settings = Settings.from_file(path)

    path = SETTINGS.joinpath('spectrogram.json')
    settings = Settings.from_file(path)

    path = SETTINGS.joinpath('dereverberate.json')
    dereverberate = Settings.from_file(path)

    path = DATASET.joinpath('DbWY_STE2017/recordings/STE03_DbWY2017.wav')
    signal = Signal(path)

    # if settings.bandpass_filter:
    #     signal.filter(
    #         settings.butter_lowcut,
    #         settings.butter_highcut
    #     )

    # if settings.reduce_noise:
    #     signal.reduce()

    signal.dereverberate(dereverberate)

    dts = dynamic_threshold_segmentation(
        signal.data,
        signal.rate,
        n_fft=settings.n_fft,
        hop_length_ms=settings.hop_length_ms,
        win_length_ms=settings.win_length_ms,
        ref_level_db=settings.ref_level_db,
        pre=settings.preemphasis,
        min_level_db=settings.min_level_db,
        min_level_db_floor=settings.min_level_db_floor,
        db_delta=settings.db_delta,
        silence_threshold=settings.silence_threshold,
        # spectral_range=settings.spectral_range,
        min_silence_for_spec=settings.min_silence_for_spec,
        max_vocal_for_spec=settings.max_vocal_for_spec,
        min_syllable_length_s=settings.min_syllable_length_s
    )

    spectrogram = Spectrogram()
    strategy = Linear(signal, settings)
    spectrogram.strategy = strategy

    spectrogram = spectrogram.generate()

    plot = VocalEnvelopeSpectrogram(signal, dts)
    plot.create()

    plt.show()
    plt.close()


if __name__ == '__main__':
    main()
