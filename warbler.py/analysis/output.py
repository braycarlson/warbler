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
    dataset = Dataset('segment')
    dataframe = dataset.load()

    dataframe.to_html('dataframe.html')


if __name__ == '__main__':
    with logger():
        main()
