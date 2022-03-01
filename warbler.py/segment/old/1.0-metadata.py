import json
import librosa
import logging
import matplotlib.pyplot as plt
import numpy as np

from multiprocessing import cpu_count, Pool
from logger import logger
from parameters import Parameters
from path import bootstrap, INDIVIDUALS
from pathlib import Path
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


# Parameters
file = Path('parameters.json')
parameters = Parameters(file)


def get_notes(wav):
    rate, data = wavfile.read(wav)
    low, high = parameters.spectral_range

    data = butter_bandpass_filter(
        int16tofloat32(data),
        low,
        high,
        rate
    )

    results = dynamic_threshold_segmentation(
        data,
        rate,
        n_fft=parameters.n_fft,
        hop_length_ms=parameters.hop_length_ms,
        win_length_ms=parameters.win_length_ms,
        ref_level_db=parameters.ref_level_db,
        pre=parameters.pre,
        min_level_db=parameters.min_level_db,
        silence_threshold=parameters.silence_threshold,
        verbose=parameters.verbosity,
        spectral_range=parameters.spectral_range,
        min_syllable_length_s=parameters.min_syllable_length_s
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
        n_fft=parameters.n_fft,
        hop_length_ms=parameters.hop_length_ms,
        win_length_ms=parameters.win_length_ms,
        ref_level_db=parameters.ref_level_db,
        pre=parameters.pre,
        min_level_db=parameters.min_level_db,
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


def create_metadata(individual):
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
                    "filename": wav.stem,
                    "start_times": [],
                    "end_times": [],
                    "labels": [],
                    "files": []
                }
            }
        }

        # Get note start/end time
        notes = get_notes(wav)

        if notes is not None:
            start, end = notes
            start = start.tolist()
            end = end.tolist()

            bird['indvs'][name]['notes']['start_times'] = start
            bird['indvs'][name]['notes']['end_times'] = end

            bird['indvs'][name]['notes']['labels'] = [label] * len(start)

        # Name of the individual
        filename = wav.stem + '.json'

        with open(individual.joinpath('json', filename), 'w+') as file:
            text = json.dumps(bird, indent=2)
            file.write(text)


@bootstrap
def main():
    processes = cpu_count()

    tasks = []

    processes = int(cpu_count() / 2)
    maxtasksperchild = 50

    with Pool(processes=processes, maxtasksperchild=maxtasksperchild) as pool:
        for individual in INDIVIDUALS:
            tasks.append(
                pool.apply_async(
                    create_metadata,
                    args=(individual,)
                )
            )

        pool.close()
        pool.join()

    with Pool(processes=processes, maxtasksperchild=maxtasksperchild) as pool:
        for task in tasks:
            task.get(10)

        pool.close()
        pool.join()


if __name__ == '__main__':
    with logger():
        main()
