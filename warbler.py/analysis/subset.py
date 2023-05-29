from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from constant import PICKLE
from datatype.dataset import Dataset
from fcmeans import FCM
from textwrap import dedent

pd.set_option('display.max_colwidth', None)


def main() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    print(dataframe[['filename', 'fcm_label_2d']])

    unique = dataframe.fcm_label_2d.unique()
    unique.sort()

    # print(unique)

    # size = [
    #     len(dataframe[dataframe.fcm_label_2d == label])
    #     for label in unique
    # ]

    # minimum = min(size)
    # maximum = max(size)

    # print(minimum, maximum)

    for index in range(1, 11):
        print(f"[Trial {index}]")

        n = 400
        small = []

        for label in unique:
            subset = dataframe[dataframe.fcm_label_2d == label]
            length = len(subset)

            # print(f"{n} samples from a total of {length} for cluster {label}")

            sample = subset.sample(n=n)
            small.append(sample)

        subset = pd.concat(small)

        columns = {'fcm_label_2d': 'fcm_label_2d_original'}
        subset = subset.rename(columns=columns)

        x = np.array(
            [
                subset.umap_x_2d,
                subset.umap_y_2d
            ]
        ).transpose()

        fcm = FCM(
            m=1.5,
            max_iter=150,
            n_clusters=19
        )

        fcm.fit(x)

        labels = fcm.predict(x)

        subset['fcm_label_2d_validation'] = labels

        dataset = Dataset('subset')
        dataset.save(subset)

        print(subset[['filename', 'fcm_label_2d_original', 'fcm_label_2d_validation']])


        comparison = (
            subset['fcm_label_2d_original'] ==
            subset['fcm_label_2d_validation']
        )

        length = len(comparison)

        percentage = (comparison.sum() / length) * 100
        percentage = round(percentage, 2)

        print(f"{percentage}% \n")

        # figsize = (10, 8)
        # fig, ax = plt.subplots(figsize=figsize)

        # subset['fcm_label_2d_original'].value_counts().sort_index().plot(ax=ax, kind='bar')

        # plt.xlabel('FCM Label')
        # plt.ylabel('Count')
        # plt.title('FCM Label 2D: Original')

        # plt.show()

        # figsize = (10, 8)
        # fig, ax = plt.subplots(figsize=figsize)

        # subset['fcm_label_2d_validation'].value_counts().sort_index().plot(ax=ax, kind='bar')

        # plt.xlabel('FCM Label')
        # plt.ylabel('Count')
        # plt.title('FCM Label 2D: Validation')

        # plt.show()




if __name__ == '__main__':
    main()
