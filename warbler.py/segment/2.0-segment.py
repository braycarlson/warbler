import logging

from bootstrap import bootstrap
from constant import CWD, PICKLE
from datatype.file import File
from datatype.settings import Settings
from logger import logger
from vocalseg.dynamic_thresholding import (
    dynamic_threshold_segmentation
)


log = logging.getLogger(__name__)


def get_onset_offset(signal, settings):
    results = dynamic_threshold_segmentation(
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
        min_silence_for_spec=settings.min_silence_for_spec,
        max_vocal_for_spec=settings.max_vocal_for_spec,
        min_syllable_length_s=settings.min_syllable_length_s,
    )

    if results is None:
        return None

    return (
        results['onsets'],
        results['offsets']
    )


def update(file):
    recordings = file.load()

    errors = []

    for recording in recordings:
        filename = recording.get('filename')
        print(f"Processing: {filename}")

        signal = recording.get('signal')

        path = CWD.joinpath(
            recording.get('segmentation')
        )

        settings = Settings.from_file(path)

        if settings.bandpass_filter:
            signal.filter(
                settings.butter_lowcut,
                settings.butter_highcut
            )

        if settings.reduce_noise:
            signal.reduce()

        recording['samplerate'] = signal.rate
        recording['duration'] = signal.duration
        recording['onset'] = []
        recording['offset'] = []
        recording['exclude'] = []

        # Get each recording's onset/offset
        time = get_onset_offset(signal, settings)

        if time is None:
            errors.append(recording)
            log.warning(f"The algorithm was unable to process {filename}")
            continue

        onset, offset = time
        onset = onset.tolist()
        offset = offset.tolist()

        if len(onset) == len(offset):
            length = len(onset)
        else:
            errors.append(recording)
            log.warning(f"The segmentation was not equal for {filename}")

        for index in settings.exclude:
            if index > (length - 1):
                errors.append(recording)
                log.warning(f"{index} is out of range for {filename}")

        recording['onset'].extend(onset)
        recording['offset'].extend(offset)
        recording['exclude'].extend(settings.exclude)

    file.save()
    return errors


@bootstrap
def main():
    errors = []

    pkl = [
        file
        for file in PICKLE.glob('*.xz')
        if file.stem not in ['aggregate', 'error']
    ]

    for p in pkl:
        file = File(p)
        error = update(file)
        errors.extend(error)

    if errors:
        file = File('error.xz')
        file.extend(errors)
        file.sort()
        file.save()

    # Aggregate
    file = File('aggregate.xz')

    for p in pkl:
        f = File(p.name)
        data = f.load()
        file.extend(data)

    file.sort()
    file.save()


if __name__ == '__main__':
    with logger():
        main()
