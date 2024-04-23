from __future__ import annotations

import numpy as np

from fcmeans import FCM
from warbler.datatype.dataset import Dataset


def main() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    x = np.array(
        [
            dataframe.umap_x_2d,
            dataframe.umap_y_2d
        ]
    ).transpose()

    fcm = FCM(
        m=1.5,
        max_iter=150,
        n_clusters=19
    )

    fcm.fit(x)

    labels = fcm.predict(x)

    dataframe['fcm_label_2d'] = labels

    dataset.save(dataframe)


if __name__ == '__main__':
    main()
