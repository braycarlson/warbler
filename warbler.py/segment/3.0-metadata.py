import json
import logging
import pickle

from bootstrap import bootstrap
from collections import OrderedDict
from dataclass.signal import Signal
from logger import logger
from multiprocessing import cpu_count, Pool
from natsort import os_sorted
from parameters import Parameters
from path import (
    IGNORE,
    INDIVIDUALS,
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

    wav = os_sorted([
        file for file in individual.glob('wav/*.wav')
        if file.is_file() and file.stem not in IGNORE
    ])

    parameter = os_sorted([
        file for file in individual.glob('parameter/*.json')
        if file.is_file() and file.stem not in IGNORE
    ])

    metadata = os_sorted([
        file for file in individual.glob('json/*.json')
        if file.is_file() and file.stem not in IGNORE
    ])

    errors = []

    for i, (w, p, m) in enumerate(zip(wav, parameter, metadata), 1):
        if w.stem == p.stem == m.stem:
            signal = Signal(w)
            parameters = Parameters(p)

            signal.filter(
                parameters.butter_lowcut,
                parameters.butter_highcut
            )

            bird = OrderedDict()

            bird['species'] = 'Setophaga adelaidae'
            bird['common_name'] = 'Adelaide\'s warbler'
            bird['wav_loc'] = w.as_posix()
            bird['parameters'] = p.as_posix()
            bird['samplerate_hz'] = signal.rate
            bird['length_s'] = signal.duration

            bird['indvs'] = {
                name: {
                    'notes': {
                        'filename': w.stem,
                        'start_times': [],
                        'end_times': [],
                        'exclude': [],
                        'labels': [],
                        'files': [],
                    }
                }
            }

            # Get note start/end time
            notes = get_notes(signal, parameters)

            if notes is None:
                log.warning(f"Error: {w.stem}")
                errors.append(w.stem)
                continue

            start, end = notes
            start = start.tolist()
            end = end.tolist()

            if parameters.exclude:
                # Remove the notes at the largest indices to preserve order
                parameters.exclude.sort(reverse=True)

                for index in parameters.exclude:
                    del start[index]
                    del end[index]

            bird['indvs'][name]['notes']['start_times'] = start
            bird['indvs'][name]['notes']['end_times'] = end

            bird['indvs'][name]['notes']['labels'] = [label] * len(start)
            bird['indvs'][name]['notes']['exclude'] = parameters.exclude

            with open(m, 'w+') as file:
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

    handle = open(PICKLE.joinpath('error.pkl'), 'wb')
    pickle.dump(errors, handle)
    handle.close()


if __name__ == '__main__':
    with logger():
        main()
