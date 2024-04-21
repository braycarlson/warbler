from __future__ import annotations

from datatype.dataset import Dataset
from datatype.signal import Signal


def migrate(old: Signal) -> Signal:
    new = Signal(old._path)

    new.strategy = old._strategy
    new.rate = old._rate
    new.data = old._data

    return new


def signal() -> None:
    dataset = Dataset('signal')
    dataframe = dataset.load()

    dataframe['signal'] = dataframe['signal'].apply(
        lambda x: migrate(x)
    )

    dataset.save(dataframe)


def segment() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    dataframe['segment'] = dataframe['segment'].apply(
        lambda x: migrate(x)
    )

    mean = dataframe['segment'].apply(
        lambda x: x.mean()
    )

    dataframe.insert(8, 'mean', mean)

    dataset.save(dataframe)


def main() -> None:
    signal()

    segment()


if __name__ == '__main__':
    main()
