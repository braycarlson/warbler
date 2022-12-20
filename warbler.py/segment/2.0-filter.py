import logging
import swifter

from bootstrap import bootstrap
from constant import SETTINGS
from datatype.dataset import Dataset
from datatype.settings import resolve, Settings
from logger import logger


log = logging.getLogger(__name__)


@bootstrap
def main():
    swifter.set_defaults(
        npartitions=None,
        dask_threshold=1,
        scheduler='processes',
        progress_bar=False,
        progress_bar_desc=False,
        allow_dask_on_strings=True,
        force_parallel=True,
    )

    dataset = Dataset('signal')
    dataframe = dataset.load()

    path = SETTINGS.joinpath('dereverberate.json')
    dereverberate = Settings.from_file(path)

    dataframe['settings'] = (
        dataframe['segmentation']
        .swifter
        .apply(
            lambda x: resolve(x)
        )
    )

    # Bandpass filter
    column = ['signal', 'settings']

    dataframe[column].swifter.apply(
        lambda x: x.signal.filter(
            x.settings.butter_lowcut,
            x.settings.butter_highcut
        ),
        axis=1
    )

    # Normalize
    dataframe['signal'].swifter.apply(
        lambda x: x.normalize()
    )

    # Dereverberate
    dataframe['signal'].swifter.apply(
        lambda x: x.dereverberate(dereverberate)
    )

    # # Reduce noise
    # dataframe['signal'].swifter.apply(
    #     lambda x: x.reduce()
    # )

    dataset.save(dataframe)


if __name__ == '__main__':
    with logger():
        main()
