import platform
import shutil
import subprocess

from path import (
    CWD,
    DATA,
    INDIVIDUALS,
    LOGS,
    NOTES,
    PICKLE,
    SPECTROGRAM
)


def remove():
    LOGS.joinpath('warbler.log').unlink(missing_ok=True)

    DATA.joinpath('df.pkl').unlink(missing_ok=True)
    DATA.joinpath('df_umap.pkl').unlink(missing_ok=True)
    DATA.joinpath('image_data.pkl').unlink(missing_ok=True)
    DATA.joinpath('notes.csv').unlink(missing_ok=True)

    if NOTES.exists():
        shutil.rmtree(NOTES)

    if PICKLE.exists():
        shutil.rmtree(PICKLE)

    if SPECTROGRAM.exists():
        shutil.rmtree(SPECTROGRAM)

    for individual in INDIVIDUALS:
        try:
            json = individual.joinpath('json')
            notes = individual.joinpath('notes')
            threshold = individual.joinpath('threshold')

            if json.exists():
                shutil.rmtree(json)

            if notes.exists():
                shutil.rmtree(notes)

            if threshold.exists():
                shutil.rmtree(threshold)
        except OSError as e:
            print(f"Error: {e}")


def execute():
    if platform.system() == 'Windows':
        venv = CWD.joinpath('venv/Scripts/python')
    else:
        venv = CWD.joinpath('venv/bin/python')

    warbler = CWD.joinpath('warbler.py')
    segment = warbler.joinpath('segment')

    spreadsheet = segment.joinpath('1.0-spreadsheet.py')
    metadata = segment.joinpath('2.0-metadata.py')
    segment = segment.joinpath('3.0-segment.py')
    copy = segment.joinpath('4.0-copy.py')
    csv = segment.joinpath('5.0-csv.py')

    subprocess.call([venv, spreadsheet])
    subprocess.call([venv, metadata])
    subprocess.call([venv, segment])
    subprocess.call([venv, copy])
    subprocess.call([venv, csv])


def main():
    remove()
    # execute()

    print('Done')


if __name__ == '__main__':
    main()
