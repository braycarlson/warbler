import matplotlib.pyplot as plt
import numpy as np

from constant import PROJECTION
from datatype.dataset import Dataset
from datatype.transition import draw_projection_transitions


def main():
    dataset = Dataset('segment')
    dataframe = dataset.load()

    unique = dataframe.folder.unique()

    for folder in unique:
        print(f"Processing: {folder}")

        individual = dataframe[dataframe.folder == folder]

        coordinates = [
            individual.umap_x_2d,
            individual.umap_y_2d
        ]

        embedding = np.column_stack(coordinates)
        sequence = individual.sequence.tolist()

        ax = draw_projection_transitions(
            embedding,
            sequence,
            alpha=0.25,
            linewidth=6
        )

        title = f"Projection Transition for {folder}"

        ax.set_title(
            title,
            fontsize=18,
            pad=25
        )

        plt.axis('off')

        # plt.show()

        PROJECTION.mkdir(parents=True, exist_ok=True)

        filename = f"transition_{folder}.png"
        path = PROJECTION.joinpath(filename)

        plt.savefig(
            path,
            bbox_inches='tight',
            dpi=300,
            format='png'
        )


if __name__ == '__main__':
    main()
