from datatype.dataset import Dataset
from datatype.segmentation import dynamic_threshold_segmentation


def main():
    dataset = Dataset('signal')
    dataframe = dataset.load()

    mask = dataframe['filename'] == 'STE02_2_YYLb2017'
    row = dataframe.loc[mask]
    row.reset_index(inplace=True)

    signal = row.at[0, 'signal']
    settings = row.at[0, 'settings']

    result = dynamic_threshold_segmentation(signal, settings)
    print(result)


if __name__ == '__main__':
    main()
