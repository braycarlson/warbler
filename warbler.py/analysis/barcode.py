import matplotlib.pyplot as plt
import seaborn as sns

from constant import PROJECTION
from datatype.barcode import (
    palplot,
    plot_sorted_barcodes,
    song_barcode
)
from datatype.dataset import Dataset


def main():
    dataset = Dataset('segment')
    dataframe = dataset.load()

    # Mask the "noise"
    dataframe = (
        dataframe[dataframe.hdbscan_label_2d > -1]
        .reset_index(drop=True)
    )

    unique = dataframe.folder.unique()

    hdbscan = dataframe.hdbscan_label_2d.unique()
    hdbscan.sort()

    # Song palette
    label_pal = sns.color_palette(
        'tab20',
        len(hdbscan)
    )

    for folder in unique:
        print(f"Processing: {folder}")

        individual = dataframe[dataframe.folder == folder]

        label_dict = {
            lab: str(i).zfill(3)
            for i, lab in enumerate(hdbscan)
        }

        label_pal_dict = {
            label_dict[lab]: color
            for lab, color in zip(hdbscan, label_pal)
        }

        figsize = (25, 3)
        fig, ax = plt.subplots(2, figsize=figsize)

        palplot(
            list(
                label_pal_dict.values()
            ),
            ax=ax[0]
        )

        for i, label in enumerate(hdbscan):
            ax[0].text(
                i,
                0,
                label,
                horizontalalignment='center',
                verticalalignment='center'
            )

        # Get list of syllables by time
        colors = []
        transitions = []

        for filename in individual.filename.unique():
            song = individual[individual['filename'] == filename]

            label = song.hdbscan_label_2d.tolist()
            onset = song.onset.tolist()
            offset = song.offset.tolist()

            transition, color = song_barcode(
                onset,
                offset,
                label,
                label_dict,
                label_pal_dict,
                resolution=0.01,
            )

            colors.append(color)
            transitions.append(transition)

        plot_sorted_barcodes(
            colors,
            transitions,
            max_list_len=600,
            seq_len=100,
            nex=600,
            figsize=(20, 4),
            ax=ax[1]
        )

        # plt.show()

        title = f"Barcode for {folder}"

        ax[0].set_title(
            title,
            fontsize=18,
            pad=25
        )

        PROJECTION.mkdir(parents=True, exist_ok=True)

        filename = f"barcode_{folder}.png"
        path = PROJECTION.joinpath(filename)

        plt.savefig(
            path,
            bbox_inches='tight',
            dpi=300,
            format='png'
        )


if __name__ == '__main__':
    main()
