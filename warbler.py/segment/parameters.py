# Dr. Sainburg uses these parameters for
# dynamic thresholding across most animals.
# This includes AVGN, AVGN_paper and the
# vocalization segmentation repository
#
#   n_fft = 4096
#   hop_length_ms = 1
#   win_length_ms = 4
#   pre = 0.97
#   silence_threshold = 0.01
#   min_silence_for_spec = 0.1

PARAMETERS = {
    # FFT window size
    "n_fft": 4096,

    # Number audio of frames in ms between STFT columns
    "hop_length_ms": 1,

    # Size of FFT window (ms)
    #   - 4 is good
    #
    #   - ~1 will begin to distort the spectrogram
    "win_length_ms": 5,

    # Reference level dB of audio
    #   - 40 is good
    #
    #   - ~20 introduces and amplifies noise
    #   - ~60 decreases the amplitude
    "ref_level_db": 50,

    # Coefficient for preemphasis filter
    "pre": 0.97,

    # Default dB minimum of spectrogram (threshold anything below)
    #   - ~ -45 to -50 is a good range for Adelaide's warblers
    #
    #   - < -45 decreases the amplitude
    #   - > -55 introduces and amplifies noise
    "min_level_db": -55,

    # Threshold for spectrogram to consider noise as silence
    "silence_threshold": 0.01,

    # Shortest expected length of silence in a song (used to set dynamic threshold)
    "min_silence_for_spec": 0.01,

    # Longest expected vocalization in seconds
    "max_vocal_for_spec": 0.125,

    # Shortest expected length of syllable
    "min_syllable_length_s": 0.01,

    # Spectral range to care about for spectrogram
    "spectral_range": [500, 10000],

    # Others
    "min_level_db_floor": -20,
    "db_delta": 5,
    "num_mel_bins": 64,
    "mel_lower_edge_hertz": 500,
    "mel_upper_edge_hertz": 10000,
    "butter_lowcut": 500,
    "butter_highcut": 10000,
    "mask_spec": True,
    "nex": -1,
    "n_jobs": -1,
    "verbosity": 1
}
