import numpy as np

from datatype.dataset import Dataset
from datatype.scatter import (
    Builder,
    ScatterFCM,
    ScatterHDBSCAN
)


def main():
    dataset = Dataset('segment')
    dataframe = dataset.load()

    unique = dataframe.hdbscan_label_2d.unique()

    # Mask the "noise"
    dataframe = (
        dataframe[dataframe.hdbscan_label_2d > -1]
        .reset_index(drop=True)
    )

    coordinates = [
        dataframe.umap_x_2d,
        dataframe.umap_y_2d
    ]

    embedding = np.column_stack(coordinates)
    label = dataframe.hdbscan_label_2d.to_numpy()

    settings = {
        'scatter': {
            'alpha': 0.50,
            's': 10
        },
        'name': 'Adelaide\'s warbler',
    }

    scatter = ScatterHDBSCAN()
    # scatter = ScatterFCM()

    scatter.builder = Builder()
    scatter.embedding = embedding
    scatter.label = label
    scatter.settings = settings
    scatter.unique = unique

    scatter.construct()
    scatter.show()

    # scatter.save('adelaideswarbler.png')

    # unique = dataframe.folder.unique()

    # for folder in unique:
    #     print(f"Processing: {folder}")

    #     settings['scatter']['alpha'] = 0.75
    #     settings['scatter']['s'] = 25
    #     settings['name'] = folder

    #     individual = dataframe[dataframe.folder == folder]

    #     coordinates = [
    #         individual.umap_x_2d,
    #         individual.umap_y_2d
    #     ]

    #     embedding = np.column_stack(coordinates)
    #     label = individual.fcm_label_2d.to_numpy()

    #     scatter.embedding = embedding
    #     scatter.label = label
    #     scatter.settings = settings
    #     scatter.unique = unique

    #     scatter.construct()
    #     # scatter.show()
    #     scatter.save(folder + '.png')


if __name__ == '__main__':
    main()
