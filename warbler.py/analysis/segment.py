"""
Segment
-------

"""

from datatype.dataset import Dataset
from datatype.segmentation import DynamicThresholdSegmentation


def main():
    dataset = Dataset('signal')
    dataframe = dataset.load()

    mask = dataframe['filename'] == 'STE02_2_YYLb2017'
    row = dataframe.loc[mask]
    row.reset_index(inplace=True)

    signal = row.at[0, 'signal']
    settings = row.at[0, 'settings']

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
