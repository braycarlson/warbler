"""
FCM
---

"""

import numpy as np

from datatype.dataset import Dataset
from fcmeans import FCM


def main() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    x = np.concatenate(
        (
            [dataframe.umap_x_2d],
            [dataframe.umap_y_2d]
        )
    )

    x = x.transpose()

    fcm = FCM(n_clusters=6)
    fcm.fit(x)

    labels = fcm.predict(x)

    dataframe['fcm_label_2d'] = labels

    dataset.save(dataframe)


if __name__ == '__main__':
    main()
