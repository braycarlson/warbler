"""
Segment
-------

"""

from __future__ import annotations

from datatype.dataset import Dataset
from datatype.segmentation import DynamicThresholdSegmentation


def main() -> None:
    dataset = Dataset('signal')
    dataframe = dataset.load()

    mask = dataframe['filename'] == 'STE02_2_YYLb2017'
    row = dataframe.loc[mask]
    row = row.reset_index()
    row = row.copy()

    signal = row.loc[0, 'signal']
    settings = row.loc[0, 'settings']

    algorithm = DynamicThresholdSegmentation()
    algorithm.signal = signal
    algorithm.settings = settings
    algorithm.start()

    onset = algorithm.component['onset']
    offset = algorithm.component['offset']
    spectrogram = algorithm.component['spectrogram']
    vocal_envelope = algorithm.component['vocal_envelope']

    print(onset)
    print(offset)
    print(spectrogram)
    print(vocal_envelope)


if __name__ == '__main__':
    main()
