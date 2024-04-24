from __future__ import annotations

from umap import UMAP
from warbler.datatype.dataset import Dataset
from warbler.datatype.spectrogram import flatten


def main() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    spectrogram = dataframe['filter_array'].tolist()
    flattened = flatten(spectrogram)

    um = UMAP(
        low_memory=True,
        metric='euclidean',
        min_dist=0.0,
        n_neighbors=5,
        n_components=3,
        n_jobs=-1,
        random_state=42
    )

    embedding = um.fit_transform(flattened)

    column = [
        'umap_x_3d',
        'umap_y_3d',
        'umap_z_3d'
    ]

    dataframe[column] = embedding

    dataset.save(dataframe)


if __name__ == '__main__':
    main()
