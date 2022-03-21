import collections
import json
import pandas as pd
import noisereduce as nr
import numpy as np

from avgn.signalprocessing.filtering import (
    butter_bandpass_filter,
    prepare_mel_matrix
)
from avgn.signalprocessing.spectrogramming import spectrogram
from avgn.utils.audio import int16_to_float32, load_wav
from collections import OrderedDict
from dataclass.signal import Signal
from joblib import delayed, Parallel
from parameters import Parameters
from path import PARAMETERS
from PIL import Image
from tqdm import tqdm


def flatten_spectrograms(specs):
    return np.reshape(
        specs,
        (
            np.shape(specs)[0],
            np.prod(np.shape(specs)[1:])
        )
    )


def subset_syllables(datafile, indv, unit='notes', include_labels=True):
    """
    Grab syllables from wav data
    """

    if type(indv) == list:
        indv = indv[0]

    if type(datafile) != OrderedDict:
        datafile = json.load(
            open(datafile),
            object_pairs_hook=OrderedDict
        )

    start_times = datafile['indvs'][indv][unit]['start_times']
    end_times = datafile['indvs'][indv][unit]['end_times']

    if include_labels:
        labels = datafile['indvs'][indv][unit]['labels']
    else:
        labels = None

    wav = datafile['wav_loc']
    parameters = datafile['parameters']
    parameters = Parameters(parameters)

    signal = Signal(wav)

    # convert data if needed
    if np.issubdtype(type(signal.data[0]), np.integer):
        signal.data = int16_to_float32(signal.data)

    signal.filter(
        parameters.butter_lowcut,
        parameters.butter_highcut,
    )

    # reduce noise
    if parameters.reduce_noise:
        data = nr.reduce_noise(
            y=signal.data,
            sr=signal.rate,
            **parameters.noise_reduce_kwargs
        )

    syllables = [
        data[int(st * signal.rate): int(et * signal.rate)]
        for st, et in zip(start_times, end_times)
    ]

    return syllables, signal.rate, labels, datafile['parameters']


def norm(x):
    minimum = np.min(x)
    maximum = np.max(x)

    return (x - minimum) / (maximum - minimum)


def make_spec(
    syll_wav,
    fs,
    parameters,
    use_tensorflow=False,
    use_mel=True,
    return_tensor=False,
    norm_uint8=False,
):
    if use_tensorflow:
        import tensorflow as tf
        from avgn.signalprocessing.spectrogramming_tf import spectrogram_tensorflow

    # convert to float
    if type(syll_wav[0]) == int:
        syll_wav = int16_to_float32(syll_wav)

    parameters = Parameters(parameters)
    mel_matrix = prepare_mel_matrix(parameters, fs)

    # create spec
    if use_tensorflow:
        spec = spectrogram_tensorflow(syll_wav, fs, parameters)
        if use_mel:
            # spec = tf.transpose(tf.tensordot(spec, mel_matrix, 1))
            if not return_tensor:
                spec = spec.numpy()
    else:
        spec = spectrogram(syll_wav, fs, parameters)

        if use_mel:
            spec = np.dot(spec.T, mel_matrix).T

    if norm_uint8:
        spec = (norm(spec) * 255).astype('uint8')

    return spec


def log_resize_spec(spec, scaling_factor=10):
    resize_shape = [
        int(np.log(np.shape(spec)[1]) * scaling_factor),
        np.shape(spec)[0]
    ]

    resize_spec = np.array(
        Image.fromarray(spec).resize(
            resize_shape,
            Image.ANTIALIAS
        )
    )

    return resize_spec


def pad_spectrogram(spectrogram, pad_length):
    """
    Pads a spectrogram to being a certain length
    """

    excess_needed = pad_length - np.shape(spectrogram)[1]
    pad_left = np.floor(float(excess_needed) / 2).astype('int')
    pad_right = np.ceil(float(excess_needed) / 2).astype('int')

    return np.pad(
        spectrogram,
        [
            (0, 0),
            (pad_left, pad_right)
        ],
        'constant',
        constant_values=0
    )


def list_match(_list, list_of_lists):
    # Using Counter
    return [
        collections.Counter(elem) == collections.Counter(_list)
        for elem in list_of_lists
    ]


def mask_spec(spec, spec_thresh=0.9, offset=1e-10):
    """
    mask threshold a spectrogram to be above some % of the maximum power
    """

    mask = spec >= (spec.max(axis=0, keepdims=1) * spec_thresh + offset)
    return spec * mask


def create_syllable_df(
    dataset,
    indv,
    unit='notes',
    log_scaling_factor=10,
    log_scale_time=True,
    pad_syllables=True,
    include_labels=True,
):
    """
    from a DataSet object, get all of the syllables from an individual as a spectrogram
    """

    with tqdm(total=4) as pbar:
        # get waveform of syllables
        pbar.set_description('getting syllables')

        with Parallel(n_jobs=-1, verbose=1) as parallel:
            syllables = parallel(
                delayed(subset_syllables)(
                    json_file,
                    indv=indv,
                    unit=unit,
                    include_labels=include_labels,
                )
                for json_file in tqdm(
                    np.array(dataset.metadata)[list_match([indv], dataset.individuals)],
                    desc='Getting syllable wavs',
                    leave=False,
                )
            )

            for i in syllables:
                print(i)

            # list syllables waveforms
            syllables_wav = [
                item for sublist in [i[0] for i in syllables]
                for item in sublist
            ]

            # repeat rate for each wav
            syllables_rate = np.concatenate(
                [np.repeat(i[1], len(i[0])) for i in syllables]
            )

            # list syllable labels
            if syllables[0][2] is not None:
                syllables_labels = np.concatenate([i[2] for i in syllables])
            else:
                syllables_labels = None

            syllables_parameters = np.concatenate([i[3] for i in syllables])

            pbar.update(1)
            pbar.set_description('Creating spectrograms')

            # create spectrograms
            syllables_spec = parallel(
                delayed(make_spec)(
                    syllable,
                    rate,
                    parameter,
                    use_mel=True,
                    use_tensorflow=False,
                )
                for syllable, rate, parameter in tqdm(
                    zip(syllables_wav, syllables_rate, syllables_parameters),
                    total=len(syllables_rate),
                    desc='Getting syllable spectrograms',
                    leave=False,
                )
            )

            # Mask spectrograms
            syllables_spec = parallel(
                delayed(mask_spec)(syllable)
                for syllable in tqdm(
                    syllables_spec,
                    total=len(syllables_rate),
                    desc='masking spectrograms',
                    leave=False,
                )
            )

            pbar.update(1)
            pbar.set_description('Rescaling syllables')

            # log resize spectrograms
            if log_scale_time:
                syllables_spec = parallel(
                    delayed(log_resize_spec)(spec, scaling_factor=log_scaling_factor)
                    for spec in tqdm(
                        syllables_spec,
                        desc='Scaling spectrograms',
                        leave=False
                    )
                )

            pbar.update(1)
            pbar.set_description('Padding syllables')

            # determine padding
            syll_lens = [np.shape(i)[1] for i in syllables_spec]
            pad_length = np.max(syll_lens)

            # pad syllables
            if pad_syllables:
                syllables_spec = parallel(
                    delayed(pad_spectrogram)(spec, pad_length)
                    for spec in tqdm(
                        syllables_spec,
                        desc='Padding spectrograms',
                        leave=False
                    )
                )
            pbar.update(1)

            syllable_df = pd.DataFrame(
                {
                    'syllables_wav': syllables_wav,
                    'syllables_rate': syllables_rate,
                    'syllables_parameters': syllables_parameters,
                    'syllables_labels': syllables_labels,
                    'syllables_spec': syllables_spec,
                }
            )

        return syllable_df


def prepare_wav(wav_loc, parameters=None):
    """
    load wav and convert to correct format
    """

    # get rate and date
    signal = Signal(wav_loc)

    # convert data if needed
    if np.issubdtype(type(signal.data[0]), np.integer):
        signal.data = int16_to_float32(signal.data)

    signal.filter(
        parameters.butter_lowcut,
        parameters.butter_highcut
    )

    if parameters.reduce_noise:
        signal.data = nr.reduce_noise(
            y=signal.data,
            r=signal.data,
            **parameters.noise_reduce_kwargs
        )

    return signal.rate, signal.data


def create_label_df(
    json_dict,
    hparams=None,
    labels_to_retain=[],
    unit='notes',
    dict_features_to_retain=[],
    key=None,
):
    """
    create a dataframe from json dictionary of time events and labels
    """

    syllable_dfs = []

    # loop through individuals
    for indvi, indv in enumerate(json_dict["indvs"].keys()):
        if unit not in json_dict["indvs"][indv].keys():
            continue

        indv_dict = {}
        indv_dict["start_time"] = json_dict["indvs"][indv][unit]["start_times"]
        indv_dict["end_time"] = json_dict["indvs"][indv][unit]["end_times"]

        # get data for individual
        for label in labels_to_retain:
            indv_dict[label] = json_dict["indvs"][indv][unit][label]
            if len(indv_dict[label]) < len(indv_dict["start_time"]):
                indv_dict[label] = np.repeat(
                    indv_dict[label], len(indv_dict["start_time"])
                )

        # create dataframe
        indv_df = pd.DataFrame(indv_dict)
        indv_df["indv"] = indv
        indv_df["indvi"] = indvi
        syllable_dfs.append(indv_df)

    syllable_df = pd.concat(syllable_dfs)

    for feat in dict_features_to_retain:
        syllable_df[feat] = json_dict[feat]

    # associate current syllables with key
    syllable_df["key"] = key

    return syllable_df


def get_row_audio(syllable_df, wav_loc, hparams):
    """
    load audio and grab individual syllables
    """

    # load audio
    signal = Signal(wav_loc)
    rate, data = prepare_wav(wav_loc, hparams)
    signal.data = signal.data.astype('float32')

    # get audio for each syllable
    syllable_df["audio"] = [
        signal.data[int(st * signal.rate): int(et * signal.rate)]
        for st, et in zip(syllable_df.start_time.values, syllable_df.end_time.values)
    ]

    syllable_df["rate"] = rate
    return syllable_df
