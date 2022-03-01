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
from joblib import Parallel, delayed
from parameters import BASELINE, Parameters
from path import IGNORE, PARAMETER, PARAMETERS
from PIL import Image
from tqdm.autonotebook import tqdm


class Dataset(object):
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

        self._get_wav()

        self._get_metadata()

        self._load()

        self._get_unique()

    def _get_wav(self):
        wav = []

        for individual in self.dataset_path:
            wav.extend(
                [
                    file for file in individual.glob('wav/*.wav')
                    if file.stem not in IGNORE
                ]
            )

        self.wav = sorted(wav)

    def _get_metadata(self):
        metadata = []

        for individual in self.dataset_path:
            metadata.extend(
                [
                    file for file in individual.glob('json/*.json')
                    if file.stem not in IGNORE
                ]
            )

        self.metadata = sorted(metadata)

    def _get_unique(self):
        self.individuals = np.array(
            [
                value.data["indvs"].keys()

                for key, value in tqdm(
                    self.datafiles.items(),
                    desc="Getting individuals...",
                    leave=False,
                )
            ]
        )

        self._unique = np.unique(self.individuals)

    def _load(self):
        with Parallel(n_jobs=-1, verbose=1) as parallel:
            df = parallel(
                delayed(Datafile)(i)
                for i in tqdm(
                    self.metadata,
                    desc="Loading metadata..."
                )
            )

            self.datafiles = {
                file.stem: df
                for file, df in zip(self.metadata, df)
            }


class Datafile(object):
    def __init__(self, metadata):
        self.data = json.load(
            open(metadata),
            object_pairs_hook=OrderedDict
        )

        self.indvs = [
            file for file in self.data["indvs"].keys()
        ]


def get_parameters(individual):
    if individual in PARAMETERS:
        # Load custom parameters
        file = PARAMETER.joinpath(individual + '.json')
        parameters = Parameters(file)
    else:
        # Load baseline parameters
        parameters = BASELINE

    return parameters


def norm(x):
    minimum = np.min(x)
    maximum = np.max(x)

    return (x - minimum) / (maximum - minimum)


def flatten_spectrograms(specs):
    return np.reshape(
        specs,
        (np.shape(specs)[0], np.prod(np.shape(specs)[1:]))
    )


def subset_syllables(datafile, indv, unit="notes", include_labels=True):
    """ Grab syllables from wav data
    """
    if type(indv) == list:
        indv = indv[0]

    if type(datafile) != OrderedDict:
        datafile = json.load(
            open(datafile),
            object_pairs_hook=OrderedDict
        )

    start_times = datafile["indvs"][indv][unit]["start_times"]
    end_times = datafile["indvs"][indv][unit]["end_times"]

    if include_labels:
        labels = datafile["indvs"][indv][unit]["labels"]
    else:
        labels = None

    # get rate and date
    rate, data = load_wav(datafile["wav_loc"])

    # convert data if needed
    if np.issubdtype(type(data[0]), np.integer):
        data = int16_to_float32(data)

    filename = datafile["indvs"][indv][unit]["filename"]
    parameters = get_parameters(filename)

    data = butter_bandpass_filter(
        data,
        parameters.butter_lowcut,
        parameters.butter_highcut,
        rate,
        order=5
    )

    # reduce noise
    if parameters.reduce_noise:
        data = nr.reduce_noise(
            y=data,
            sr=rate,
            **parameters.noise_reduce_kwargs
        )

    syllables = [
        data[int(st * rate): int(et * rate)]
        for st, et in zip(start_times, end_times)
    ]

    return syllables, rate, labels


def create_label_df(
    datafile,
    labels_to_retain=[],
    unit="syllables",
    dict_features_to_retain=[],
    key=None,
):
    """ create a dataframe from json dictionary of time events and labels
    """

    syllable_dfs = []

    # loop through individuals
    for indvi, indv in enumerate(datafile["indvs"].keys()):
        if unit not in datafile["indvs"][indv].keys():
            continue

        indv_dict = {}
        indv_dict["start_time"] = datafile["indvs"][indv][unit]["start_times"]
        indv_dict["end_time"] = datafile["indvs"][indv][unit]["end_times"]

        # get data for individual
        for label in labels_to_retain:
            indv_dict[label] = datafile["indvs"][indv][unit][label]

            if len(indv_dict[label]) < len(indv_dict["start_time"]):
                indv_dict[label] = np.repeat(
                    indv_dict[label],
                    len(indv_dict["start_time"])
                )

        # create dataframe
        indv_df = pd.DataFrame(indv_dict)
        indv_df["indv"] = indv
        indv_df["indvi"] = indvi
        syllable_dfs.append(indv_df)

    syllable_df = pd.concat(syllable_dfs)

    for feat in dict_features_to_retain:
        syllable_df[feat] = datafile[feat]

    # associate current syllables with key
    syllable_df["key"] = key

    return syllable_df


def make_spec(
    syll_wav,
    fs,
    key,
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

    parameters = get_parameters(key)
    mel_matrix = prepare_mel_matrix(parameters, 44100)

    # create spec
    if use_tensorflow:
        spec = spectrogram_tensorflow(syll_wav, fs, parameters)

        if use_mel:
            spec = tf.transpose(
                tf.tensordot(spec, mel_matrix, 1)
            )

            if not return_tensor:
                spec = spec.numpy()
    else:
        spec = spectrogram(syll_wav, fs, parameters)

        if use_mel:
            spec = np.dot(spec.T, mel_matrix).T

    if norm_uint8:
        spec = (norm(spec) * 255).astype("uint8")

    return spec


def log_resize_spec(spec, scaling_factor=10):
    resize_shape = [
        int(np.log(np.shape(spec)[1]) *
            scaling_factor), np.shape(spec)[0]
    ]

    resize_spec = np.array(
        Image.fromarray(spec).resize(
                resize_shape,
                Image.ANTIALIAS
            )
        )

    return resize_spec


def pad_spectrogram(spectrogram, pad_length):
    """ Pads a spectrogram to being a certain length
    """
    excess_needed = pad_length - np.shape(spectrogram)[1]
    pad_left = np.floor(float(excess_needed) / 2).astype("int")
    pad_right = np.ceil(float(excess_needed) / 2).astype("int")

    return np.pad(
        spectrogram,
        [
            (0, 0),
            (pad_left, pad_right)
        ],
        "constant",
        constant_values=0
    )


def list_match(_list, list_of_lists):
    # Using Counter
    return [
        collections.Counter(elem) == collections.Counter(_list)
        for elem in list_of_lists
    ]


def mask_spec(spec, spec_thresh=0.9, offset=1e-10):
    """ mask threshold a spectrogram to be above some % of the maximum power
    """
    mask = spec >= (spec.max(axis=0, keepdims=1) * spec_thresh + offset)
    return spec * mask


def prepare_wav(wav_loc, parameters=None):
    # get rate and date
    rate, data = load_wav(wav_loc)

    # convert data if needed
    if np.issubdtype(type(data[0]), np.integer):
        data = int16_to_float32(data)

    if parameters is not None:
        if parameters.bandpass_filter:
            data = butter_bandpass_filter(
                data,
                parameters.butter_lowcut,
                parameters.butter_highcut,
                rate,
                order=5
            )

        if parameters.reduce_noise:
            data = nr.reduce_noise(
                y=data,
                sr=rate,
                **parameters.noise_reduce_kwargs
            )

    return rate, data


def get_row_audio(syllable_df, wav_loc, key):
    """ load audio and grab individual syllables
    """

    parameters = get_parameters(key)

    # load audio
    rate, data = prepare_wav(wav_loc, parameters)
    data = data.astype('float32')

    # get audio for each syllable
    syllable_df["audio"] = [
        data[int(st * rate): int(et * rate)]
        for st, et in zip(
            syllable_df.start_time.values,
            syllable_df.end_time.values
        )
    ]

    syllable_df["rate"] = rate
    return syllable_df
