import json
import logging
import pickle

from collections import OrderedDict
from dataclass.signal import Signal
from multiprocessing import cpu_count, Pool
from logger import logger
from parameters import BASELINE, Parameters
from path import (
    bootstrap,
    IGNORE,
    INDIVIDUALS,
    PARAMETER,
    PARAMETERS,
    PICKLE
)
from vocalseg.dynamic_thresholding import (
    dynamic_threshold_segmentation
)


log = logging.getLogger(__name__)


def get_notes(signal, parameters):
    results = dynamic_threshold_segmentation(
        signal.data,
        signal.rate,
        n_fft=parameters.n_fft,
        hop_length_ms=parameters.hop_length_ms,
        win_length_ms=parameters.win_length_ms,
        ref_level_db=parameters.ref_level_db,
        pre=parameters.preemphasis,
        min_level_db=parameters.min_level_db,
        silence_threshold=parameters.silence_threshold,
        spectral_range=parameters.spectral_range,
        min_syllable_length_s=parameters.min_syllable_length_s
    )

    if results is not None:
        time = (
            results['onsets'],
            results['offsets']
        )

        return time

    return None


def create_metadata(individual):
    name = individual.stem
    label = name.replace('_STE2017', '')

    wavs = [
        file for file in individual.glob('wav/*.wav')
        if file.stem not in IGNORE
    ]

    wavs = sorted(wavs)

    errors = []

    for index, wav in enumerate(wavs, 1):
        parameters = BASELINE

        if wav.stem in PARAMETERS:
            # Load custom parameters
            file = PARAMETER.joinpath(wav.stem + '.json')
            parameters = Parameters(file)

        signal = Signal(wav)

        signal.filter(
            parameters.butter_lowcut,
            parameters.butter_highcut
        )

        # Padding index
        index = str(index).zfill(2)

        bird = OrderedDict()

        bird['species'] = 'Setophaga adelaidae'
        bird['common_name'] = 'Adelaide\'s warbler'
        bird['wav_loc'] = wav.as_posix()
        bird['samplerate_hz'] = signal.rate
        bird['length_s'] = signal.duration

        bird['indvs'] = {
            name: {
                'notes': {
                    'filename': wav.stem,
                    'start_times': [],
                    'end_times': [],
                    'labels': [],
                    'files': [],
                    'parameters': parameters.path.as_posix()
                }
            }
        }

        # Get note start/end time
        notes = get_notes(signal, parameters)

        if notes is None:
            log.warning(f"Error: {wav.stem}")
            errors.append(wav.stem)
            continue

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

    return errors


@bootstrap
def main():
    processes = int(cpu_count() / 2)
    maxtasksperchild = 100

    tasks = []

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

    errors = []

    with Pool(processes=processes, maxtasksperchild=maxtasksperchild) as pool:
        for task in tasks:
            error = task.get(10)
            errors.extend(error)

        pool.close()
        pool.join()

    e = PICKLE.joinpath('error.pkl')

    if not e.is_file():
        e.touch()

    handle = open(e, 'wb')
    pickle.dump(errors, handle)
    handle.close()


if __name__ == '__main__':
    with logger():
        main()
