"""
Sanity
------

"""

from __future__ import annotations

import numpy as np
import pandas as pd
import sys

from avgn.dataset import DataSet
from avgn.signalprocessing.create_spectrogram_dataset import (
    create_label_df,
    get_row_audio,
    # log_resize_spec,
    make_spec,
    # mask_spec,
    # pad_spectrogram,
    # prepare_wav
)
# from avgn.signalprocessing.filtering import prepare_mel_matrix
from avgn.signalprocessing.spectrogramming import _build_mel_basis, melspectrogram
from avgn.utils.hparams import HParams
from constant import AVGN, DATASET, PICKLE, SETTINGS
from datatype.dataset import Dataset
from datatype.settings import Settings
from datatype.signal import Signal
from datatype.spectrogram import (
    # compress,
    Linear,
    Mel,
    # mel_matrix,
    pad,
    resize,
    Segment,
    Spectrogram
)
from joblib import Parallel, delayed
from tqdm.autonotebook import tqdm


def create_baseline() -> None:
    path = SETTINGS.joinpath('spectrogram.json')
    settings = Settings.from_file(path)

    hparams = HParams(
        n_fft=settings.n_fft,
        hop_length_ms=settings.hop_length_ms,
        win_length_ms=settings.win_length_ms,
        ref_level_db=settings.ref_level_db,
        preemphasis=settings.preemphasis,
        min_level_db=settings.min_level_db,
        min_level_db_floor=settings.min_level_db_floor,
        db_delta=settings.db_delta,
        silence_threshold=settings.silence_threshold,
        min_silence_for_spec=settings.min_silence_for_spec,
        max_vocal_for_spec=settings.max_vocal_for_spec,
        min_syllable_length_s=settings.min_syllable_length_s,
        spectral_range=settings.spectral_range,
        num_mel_bins=settings.num_mel_bins,
        mel_lower_edge_hertz=settings.mel_lower_edge_hertz,
        mel_upper_edge_hertz=settings.mel_upper_edge_hertz,
        butter_lowcut=settings.butter_lowcut,
        butter_highcut=settings.butter_highcut
    )

    output = AVGN.joinpath('data')
    output.mkdir(parents=True, exist_ok=True)

    path = output.joinpath('warbler')

    dataset = DataSet(
        path,
        build_mel_matrix=True,
        hparams=hparams
    )

    n_jobs = -1
    verbosity = 0

    with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
        syllable_dfs = parallel(
            delayed(create_label_df)(
                dataset.data_files[key].data,
                hparams=dataset.hparams,
                dict_features_to_retain=['filename'],
                unit='notes',
                key=key,
            )
            for key in tqdm(dataset.data_files.keys())
        )

    syllable_df = pd.concat(syllable_dfs)

    with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
        syllable_dfs = parallel(
            delayed(get_row_audio)(
                syllable_df[syllable_df.key == key],
                dataset.data_files[key].data['wav_loc'],
                dataset.hparams
            )
            for key in tqdm(syllable_df.key.unique())
        )

    syllable_df = pd.concat(syllable_dfs)

    syllables_wav = syllable_df.audio.tolist()
    syllables_rate = syllable_df.rate.tolist()

    with Parallel(n_jobs=n_jobs, verbose=verbosity) as parallel:
        syllables_spec = parallel(
            delayed(make_spec)(
                syllable,
                rate,
                hparams=dataset.hparams,
                mel_matrix=dataset.mel_matrix,
                use_mel=False,
                use_tensorflow=False,
            )
            for syllable, rate in tqdm(
                zip(syllables_wav, syllables_rate),
                total=len(syllables_rate),
                desc='Getting syllable spectrograms',
                leave=False,
            )
        )

    syllable_df['spectrogram'] = syllables_spec

    dataset = Dataset('baseline')
    dataset.save(syllable_df)


def flatten(arg):
    if not isinstance(arg, list):
        return [arg]

    return [x for sub in arg for x in flatten(sub)]


def create_recording_path(folder, filename):
    return (
        DATASET
        .joinpath(folder)
        .joinpath('recordings')
        .joinpath(filename + '.wav')
    )


def create_settings_path(folder, filename):
    return (
        DATASET
        .joinpath(folder)
        .joinpath('segmentation')
        .joinpath(filename + '.json')
    )


# def full() -> None:
#     # create_baseline()

#     np.set_printoptions(
#         suppress=True,
#         threshold=sys.maxsize
#     )

#     filename = '1_STE01a_RRO2017'
#     onset = 0.135692
#     offset = 0.181587

#     # AVGN dataset
#     dataset = Dataset('avgn')
#     dataframe = dataset.load()

#     mask = (
#         (dataframe.filename == filename) &
#         (round(dataframe.start_time, 6) == onset) &
#         (round(dataframe.end_time, 6) == offset)
#     )

#     subset = dataframe.loc[mask].squeeze()

#     s_spectrogram = subset.spectrogram

#     # warbler.py dataset
#     dataset = Dataset('segment')
#     dataframe = dataset.load()

#     mask = (
#         (dataframe.filename == filename) &
#         (round(dataframe.onset, 6) == onset) &
#         (round(dataframe.offset, 6) == offset)
#     )

#     subset = dataframe[mask].squeeze()

#     folder = subset.folder
#     filename = subset.filename

#     recording = create_recording_path(folder, filename)
#     settings = create_settings_path(folder, filename)

#     # Default settings
#     path = SETTINGS.joinpath('spectrogram.json')
#     default = Settings.from_file(path)

#     signal = Signal(recording)

#     signal.filter(
#         default.butter_lowcut,
#         default.butter_highcut
#     )

#     settings = Settings.from_file(settings)

#     onset = subset.onset
#     offset = subset.offset

#     segment = signal.segment(onset, offset)

#     # Mel filterbank
#     path = PICKLE.joinpath('matrix.npy')
#     matrix = np.load(path, allow_pickle=True)

#     strategy = Linear(segment, default, matrix)

#     spectrogram = Spectrogram()
#     spectrogram.strategy = strategy
#     spectrogram = spectrogram.generate()

#     spectrogram = resize(spectrogram)

#     padding = 50
#     spectrogram = pad(spectrogram, padding)

#     c_spectrogram = (spectrogram * 255).astype('uint8')

#     equal = np.array_equal(s_spectrogram, c_spectrogram)
#     print(equal)

#     with open('sainburg.txt', 'w+') as handle:
#         s_spectrogram = flatten(s_spectrogram.tolist())
#         s_spectrogram = ', '.join(map(str, s_spectrogram))

#         handle.write(s_spectrogram)

#     with open('carlson.txt', 'w+') as handle:
#         c_spectrogram = flatten(c_spectrogram.tolist())
#         c_spectrogram = ', '.join(map(str, c_spectrogram))

#         handle.write(c_spectrogram)


def main():
    # create_baseline()

    np.set_printoptions(
        suppress=True,
        threshold=sys.maxsize
    )

    path = SETTINGS.joinpath('spectrogram.json')
    settings = Settings.from_file(path)

    hparams = HParams(
        n_fft=settings.n_fft,
        hop_length_ms=settings.hop_length_ms,
        win_length_ms=settings.win_length_ms,
        ref_level_db=settings.ref_level_db,
        preemphasis=settings.preemphasis,
        min_level_db=settings.min_level_db,
        min_level_db_floor=settings.min_level_db_floor,
        db_delta=settings.db_delta,
        silence_threshold=settings.silence_threshold,
        min_silence_for_spec=settings.min_silence_for_spec,
        max_vocal_for_spec=settings.max_vocal_for_spec,
        min_syllable_length_s=settings.min_syllable_length_s,
        spectral_range=settings.spectral_range,
        num_mel_bins=settings.num_mel_bins,
        mel_lower_edge_hertz=settings.mel_lower_edge_hertz,
        mel_upper_edge_hertz=settings.mel_upper_edge_hertz,
        butter_lowcut=settings.butter_lowcut,
        butter_highcut=settings.butter_highcut,
        sample_rate=settings.sample_rate
    )

    filename = '1_STE01a_RRO2017'
    onset = 0.135692
    offset = 0.181587

    # AVGN dataset
    dataset = Dataset('avgn')
    dataframe = dataset.load()

    mask = (
        (dataframe.filename == filename) &
        (round(dataframe.start_time, 6) == onset) &
        (round(dataframe.end_time, 6) == offset)
    )

    subset = dataframe.loc[mask].squeeze()

    s_audio = subset.audio

    basis = _build_mel_basis(
        hparams,
        settings.sample_rate,
    )

    s_spectrogram = melspectrogram(
        s_audio,
        settings.sample_rate,
        hparams,
        basis
    )


    # warbler.py dataset
    dataset = Dataset('segment')
    dataframe = dataset.load()

    mask = (
        (dataframe.filename == filename) &
        (round(dataframe.onset, 6) == onset) &
        (round(dataframe.offset, 6) == offset)
    )

    subset = dataframe[mask].squeeze()

    folder = subset.folder
    filename = subset.filename

    recording = create_recording_path(folder, filename)
    settings = create_settings_path(folder, filename)

    # Default settings
    path = SETTINGS.joinpath('spectrogram.json')
    default = Settings.from_file(path)

    signal = Signal(recording)

    signal.filter(
        default.butter_lowcut,
        default.butter_highcut
    )

    settings = Settings.from_file(settings)

    onset = subset.onset
    offset = subset.offset

    segment = signal.segment(onset, offset)

    # # Mel filterbank
    # path = PICKLE.joinpath('matrix.npy')
    # matrix = np.load(path, allow_pickle=True)

    strategy = Mel(segment, default)

    spectrogram = Spectrogram()
    spectrogram.strategy = strategy
    spectrogram = spectrogram.generate()

    # spectrogram = resize(spectrogram)

    # padding = 50
    # spectrogram = pad(spectrogram, padding)

    c_spectrogram = (spectrogram * 255).astype('uint8')
    s_spectrogram = (s_spectrogram * 255).astype('uint8')

    equal = np.array_equal(s_spectrogram, c_spectrogram)
    print(equal)

    with open('sainburg.txt', 'w+') as handle:
        s_spectrogram = flatten(s_spectrogram.tolist())
        s_spectrogram = ', '.join(map(str, s_spectrogram))

        handle.write(s_spectrogram)

    with open('carlson.txt', 'w+') as handle:
        c_spectrogram = flatten(c_spectrogram.tolist())
        c_spectrogram = ', '.join(map(str, c_spectrogram))

        handle.write(c_spectrogram)


if __name__ == '__main__':
    main()
