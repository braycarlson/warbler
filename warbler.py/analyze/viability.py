import logging
import pandas as pd

from bootstrap import bootstrap
from constant import SPREADSHEET
from logger import logger

log = logging.getLogger(__name__)


@bootstrap
def main():
    spreadsheet = SPREADSHEET.joinpath('2017.xlsx')

    dataframe = pd.read_excel(
        spreadsheet,
        sheet_name='2017_recording_viability',
        engine='openpyxl'
    )

    mask = (dataframe['recording_viability'] == 'N')

    dataframe = dataframe[mask]
    bad = dataframe.filename.to_numpy()

    print(bad)


if __name__ == '__main__':
    with logger():
        main()
