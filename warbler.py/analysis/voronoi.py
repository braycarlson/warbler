import numpy as np

from datatype.dataset import Dataset
from datatype.voronoi import Builder, Voronoi


def main():
    dataset = Dataset('segment')
    dataframe = dataset.load()

    unique = dataframe.hdbscan_label_2d.unique()

    # Mask the "noise"
    dataframe = (
        dataframe[dataframe.hdbscan_label_2d > -1]
        .reset_index(drop=True)
    )

    by = ['duration']

    ascending = [False]

    dataframe.sort_values(
        ascending=ascending,
        by=by,
        inplace=True
    )

    coordinates = [
        dataframe.umap_x_2d,
        dataframe.umap_y_2d
    ]

    embedding = np.column_stack(coordinates)

    spectrogram = dataframe.original_array.to_numpy()
    label = dataframe.hdbscan_label_2d.to_numpy()

    voronoi = Voronoi()

    voronoi.builder = Builder()
    voronoi.embedding = embedding
    voronoi.label = label
    voronoi.spectrogram = ~spectrogram
    voronoi.unique = unique

    voronoi.construct()
    voronoi.show()


if __name__ == '__main__':
    main()
