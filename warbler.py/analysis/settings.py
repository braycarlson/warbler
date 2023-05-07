from constant import SEGMENTATIONS
from datatype.settings import Settings


def insert() -> None:
    """Insert an attribute into the settings and saves the updated settings.

    Args:
        None.

    Returns:
        None.

    """

    for segmentation in SEGMENTATIONS:
        settings = Settings.from_file(segmentation)

        srk, srv = 'sample_rate', 44100

        keys = list(
            settings.__dict__.keys()
        )

        index = keys.index('butter_highcut')
        keys.insert(index + 1, srk)

        settings.__dict__[srk] = srv

        update = {
            key: settings.__dict__[key]
            for key in keys
            if key in settings.__dict__
        }

        settings = Settings.from_dict(update)
        settings.save(segmentation)


def update():
    """Update an existing parameter in the settings and save the settings.

    Args:
        None.

    Returns:
        None.

    """

    for segmentation in SEGMENTATIONS:
        settings = Settings.from_file(segmentation)
        settings.exclude = []

        settings.save(segmentation)


def main():
    insert()


if __name__ == '__main__':
    main()
