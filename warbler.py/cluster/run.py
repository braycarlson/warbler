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
    cluster = warbler.joinpath('cluster')

    dataframe = cluster.joinpath('1.0-dataframe.py')
    umap = cluster.joinpath('2.0-umap.py')
    clustering = cluster.joinpath('3.0-clustering.py')

    subprocess.call([venv, dataframe])
    subprocess.call([venv, umap])
    subprocess.call([venv, clustering])


def main():
    # remove()
    execute()

    print('Done')


if __name__ == '__main__':
    main()
