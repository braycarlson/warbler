import numpy as np

from datatype.dataset import Dataset
from fcmeans import FCM


def main():
    dataset = Dataset('segment')
    dataframe = dataset.load()

    X = np.concatenate(
        (
            [dataframe.umap_x_3d],
            [dataframe.umap_y_3d],
            [dataframe.umap_z_3d],
        )
    )

    X = X.transpose()

    fcm = FCM(n_clusters=6)
    fcm.fit(X)

    labels = fcm.predict(X)

    dataframe['fcm_label_3d'] = labels

    dataset.save(dataframe)


if __name__ == '__main__':
    main()
