from __future__ import annotations

import numpy as np

from datatype.dataset import Dataset
from datatype.validation import Validate


def main() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    x = np.array(
        [
            dataframe.umap_x_2d,
            dataframe.umap_y_2d
        ]
    ).transpose()

    validate = Validate()
    stability = validate.bootstrap(x)

    score = stability.score.to_numpy()

    mean = round(np.mean(score) * 100, 2)

    print(stability)
    print(f"Mean similarity score: {mean:.2f}%")


if __name__ == '__main__':
    main()
