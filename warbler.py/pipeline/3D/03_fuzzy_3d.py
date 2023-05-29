"""
FCM
---

"""

from __future__ import annotations

import numpy as np

from datatype.dataset import Dataset
from fcmeans import FCM


def main() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    x = np.array(
        [
            dataframe.umap_x_3d,
            dataframe.umap_y_3d,
            dataframe.umap_z_3d
        ]
    ).transpose()

    fcm = FCM(
        m=1.5,
        max_iter=150,
        n_clusters=19
    )

    fcm.fit(x)

    labels = fcm.predict(x)

    dataframe['fcm_label_3d'] = labels

    dataset.save(dataframe)


if __name__ == '__main__':
    main()
