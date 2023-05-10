"""
Barcode
-------

"""

import matplotlib.pyplot as plt
import seaborn as sns

from datatype.barcode import (
    BarcodeHDBSCAN,
    Builder,
    palplot,
    SongBarcode
)
from datatype.dataset import Dataset


def main() -> None:
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

    settings = {
        'figure': {
            'figsize': (20, 4),

        },
        'length': 100,
        'maximum': 600,
        'name': 'Adelaide\'s warbler',
        'nex': 600
    }

    for folder in unique:
        settings['name'] = folder

        individual = dataframe[dataframe.folder == folder]

        mapping = {
            lab: str(i).zfill(3)
            for i, lab in enumerate(hdbscan)
        }

        palette = {
            mapping[lab]: color
            for lab, color in zip(hdbscan, label_pal, strict=True)
        }

        figsize = (25, 3)
        figure, ax = plt.subplots(2, figsize=figsize)

        palplot(
            list(
                palette.values()
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
            subset = individual[individual['filename'] == filename]

            label = subset.hdbscan_label_2d.tolist()
            onset = subset.onset.tolist()
            offset = subset.offset.tolist()

            song = SongBarcode()
            song.onset = onset
            song.offset = offset
            song.label = label
            song.mapping = mapping
            song.palette = palette

            transition, color = song.build()

            colors.append(color)
            transitions.append(transition)

        builder = Builder()
        builder.ax = ax[1]
        builder.color = colors
        builder.transition = transitions

        barcode = BarcodeHDBSCAN()
        barcode.builder = builder
        barcode.settings = settings

        barcode.build()

        # barcode.show()

        filename = f"barcode_{folder}.png"

        barcode.save(
            figure=figure,
            filename=filename,
        )


if __name__ == '__main__':
    main()
