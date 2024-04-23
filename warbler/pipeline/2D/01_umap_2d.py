from __future__ import annotations

import pickle

from umap import UMAP
from warbler.datatype.dataset import Dataset
from warbler.datatype.spectrogram import flatten


def main() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    array = dataframe['filter_array'].apply(
        lambda x: pickle.loads(
            bytes.fromhex(x)
        )
    )

    spectrogram = array.tolist()
    flattened = flatten(spectrogram)

    um = UMAP(
        low_memory=True,
        metric='euclidean',
        min_dist=0.0,
        n_neighbors=5,
        n_components=2,
        n_jobs=-1
    )

    embedding = um.fit_transform(flattened)

    column = [
        'umap_x_2d',
        'umap_y_2d'
    ]

    dataframe[column] = embedding

    dataset.save(dataframe)


if __name__ == '__main__':
    main()
