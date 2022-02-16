import platform
import shutil
import subprocess

from path import CWD, DATA, INDIVIDUALS, LOGS, NOTES


def remove():
    logs = LOGS.joinpath('warbler.log')
    dataframe = DATA.joinpath('df.pkl')
    umap = DATA.joinpath('df_umap.pkl')
    csv = DATA.joinpath('info_file.csv')

    logs.unlink(missing_ok=True)
    dataframe.unlink(missing_ok=True)
    umap.unlink(missing_ok=True)
    csv.unlink(missing_ok=True)

    if NOTES.exists():
        shutil.rmtree(NOTES)

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

    metadata = segment.joinpath('1.0-metadata.py')
    segment = segment.joinpath('2.0-segment.py')
    split = segment.joinpath('3.0-split.py')
    csv = segment.joinpath('4.0-csv.py')

    subprocess.call([venv, metadata])
    subprocess.call([venv, segment])
    subprocess.call([venv, split])
    subprocess.call([venv, csv])


def main():
    # remove()
    execute()

    print('Done')


if __name__ == '__main__':
    main()
