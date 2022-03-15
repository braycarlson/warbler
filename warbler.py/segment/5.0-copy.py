import logging
import shutil

from bootstrap import bootstrap
from logger import logger
from path import INDIVIDUALS, NOTES


log = logging.getLogger(__name__)


@bootstrap
def main():
    for individual in INDIVIDUALS:
        notes = [
            file for file in individual.glob('notes/*.wav')
            if file.is_file()
        ]

        for note in notes:
            try:
                shutil.copy(
                    note,
                    NOTES
                )
            except OSError as e:
                print(f"Error: {e}")

    print('Done')


if __name__ == '__main__':
    with logger():
        main()
