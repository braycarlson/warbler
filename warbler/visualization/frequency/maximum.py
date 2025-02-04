from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from warbler.constant import PROJECTION
from warbler.datatype.dataset import Dataset
from warbler.datatype.partition import Partition


def main() -> None:
    save = False

    plt.style.use('science')

    dataset = Dataset('segment')
    dataframe = dataset.load()

    figsize = (12, 9)
    plt.figure(figsize=figsize)

    partition = Partition()
    partition.dataframe = dataframe
    partition.column = 'maximum'

    partitions, sizes = partition.divide()
    legend = partition.palette(sizes)

    for name, p in zip(sizes, partitions):
        color = legend[name]

        plt.scatter(
            p.umap_x_2d,
            p.umap_y_2d,
            c=np.array(color).reshape(1, -1),
            label=str(name),
            s=3
        )

    plt.legend(
        bbox_to_anchor=(0.80, 1.0080),
        loc='upper left'
    )

    plt.title('Adelaide\'s Warbler Notes by Maximum Frequency')

    plt.axis('off')
    plt.tight_layout()

    if save:
        path = PROJECTION.joinpath('maximum_frequency.png')

        plt.savefig(
            path,
            bbox_inches='tight',
            dpi=300,
            format='png'
        )
    else:
        plt.show()

    plt.close()


if __name__ == '__main__':
    main()
