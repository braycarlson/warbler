import json
import librosa
import logging
import matplotlib.pyplot as plt
import numpy as np

from parameters import PARAMETERS
from logger import logger
from path import bootstrap, INDIVIDUALS
from scipy.io import wavfile
from vocalseg.dynamic_thresholding import (
    dynamic_threshold_segmentation
)
from vocalseg.utils import (
    butter_bandpass_filter,
    int16tofloat32,
    plot_spec,
    spectrogram
)


log = logging.getLogger(__name__)


# See: parameters.py for more information
n_fft = PARAMETERS.get('n_fft')
hop_length_ms = PARAMETERS.get('hop_length_ms')
win_length_ms = PARAMETERS.get('win_length_ms')
ref_level_db = PARAMETERS.get('ref_level_db')
pre = PARAMETERS.get('pre')
min_level_db = PARAMETERS.get('min_level_db')
silence_threshold = PARAMETERS.get('silence_threshold')
min_silence_for_spec = PARAMETERS.get('min_silence_for_spec')
max_vocal_for_spec = PARAMETERS.get('max_vocal_for_spec')
min_syllable_length_s = PARAMETERS.get('min_syllable_length_s')
spectral_range = PARAMETERS.get('spectral_range')

low, high = spectral_range


def get_notes(wav):
    rate, data = wavfile.read(wav)

    data = butter_bandpass_filter(
        int16tofloat32(data),
        low,
        high,
        rate
    )

    results = dynamic_threshold_segmentation(
        data,
        rate,
        n_fft=n_fft,
        hop_length_ms=hop_length_ms,
        win_length_ms=win_length_ms,
        ref_level_db=ref_level_db,
        pre=pre,
        min_level_db=min_level_db,
        silence_threshold=silence_threshold,
        verbose=False,
        spectral_range=spectral_range,
        min_syllable_length_s=min_syllable_length_s
    )

    if results is not None:
        time = (results['onsets'], results['offsets'])
        return time

    threshold = wav.parent.parent.joinpath('threshold')
    bad = threshold.joinpath('bad')

    path = wav.as_posix()

    figsize = (20, 3)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(data)

    fig.savefig(
        bad.joinpath(wav.stem + '_wave')
    )

    plt.close(fig)

    spec = spectrogram(
        data,
        rate,
        n_fft=n_fft,
        hop_length_ms=hop_length_ms,
        win_length_ms=win_length_ms,
        ref_level_db=ref_level_db,
        pre=pre,
        min_level_db=min_level_db,
    )

    np.shape(spec)

    fig, ax = plt.subplots(figsize=figsize)
    plot_spec(spec, fig, ax)
    fig.savefig(
        bad.joinpath(wav.stem + '_spectrogram')
    )
    plt.close(fig)

    log.warning(f"Inspect: {path}")
    return None


@bootstrap
def main():
    for individual in INDIVIDUALS:
        name = individual.stem

        # Get a list of .wav files
        wavs = [file for file in individual.glob('wav/*.wav')]
        wavs = list(wavs)

        for index, wav in enumerate(wavs, 1):
            # Pad zero
            index = str(index).zfill(2)

            # Name of the individual
            directory = wav.parent.parent.stem

            if directory != name:
                continue

            label = wav.parent.parent.stem
            label = label.replace('_STE2017', '')

            # Get the .wav sample rate and duration
            posix = wav.as_posix()
            samplerate = librosa.get_samplerate(posix)
            duration = librosa.get_duration(filename=posix)

            bird = {}

            # Species
            bird['species'] = 'Setophaga adelaidae'
            bird['common_name'] = 'Adelaide\'s warbler'
            bird['wav_loc'] = posix

            # Sample rate and duration
            #   - Sample rate in Hz
            #   - Duration in seconds
            bird['samplerate_hz'] = samplerate
            bird['length_s'] = duration

            bird["indvs"] = {
                name: {
                    "notes": {
                        "start_times": [],
                        "end_times": [],
                        "labels": [],
                        "sequence_num": [],
                        "files": []
                    }
                }
            }

            # Get note start/end time
            notes = get_notes(wav)

            if notes is not None:
                start, end = notes

                bird['indvs'][name]['notes']['start_times'] = start.tolist()
                bird['indvs'][name]['notes']['end_times'] = end.tolist()

                bird['indvs'][name]['notes']['labels'] = [label] * len(start)
                bird['indvs'][name]['notes']['sequence_num'] = [index] * len(start)

            # Name of the individual
            filename = wav.stem + '.json'

            with open(individual.joinpath('json', filename), 'w+') as file:
                text = json.dumps(bird, indent=2)
                file.write(text)


if __name__ == '__main__':
    with logger():
        main()
