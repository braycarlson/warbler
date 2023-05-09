"""
Global
------

"""

from constant import SEGMENTATIONS
from datatype.settings import Settings


def main():
    for path in SEGMENTATIONS:
        settings = Settings.from_file(path)

        # settings.ref_level_db = 15
        # settings.silence_threshold = 0.0001
        # settings.mask_spec = False
        # settings.reduce_noise = False
        settings.exclude = []

        settings.save(path)


if __name__ == '__main__':
    main()
