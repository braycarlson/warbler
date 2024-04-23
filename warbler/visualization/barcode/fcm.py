from __future__ import annotations

import matplotlib.pyplot as plt
import scienceplots
import seaborn as sns

from warbler.constant import SETTINGS
from warbler.datatype.barcode import (
    BarcodeFCM,
    Builder,
    palplot,
    SongBarcode
)
from warbler.datatype.dataset import Dataset
from warbler.datatype.settings import Settings


def main() -> None:
    save = False

    plt.style.use('science')

    dataset = Dataset('segment')
    dataframe = dataset.load()

    # Load default settings
    path = SETTINGS.joinpath('barcode.json')
    settings = Settings.from_file(path)

    unique = dataframe.folder.unique()

    fcm = dataframe.fcm_label_2d.unique()
    fcm.sort()

    # Song palette
    label_pal = sns.color_palette(
        'tab20',
        len(fcm)
    )

    for folder in unique:
        settings['name'] = folder

        individual = dataframe[dataframe.folder == folder]

        mapping = {
            lab: str(i).zfill(3)
            for i, lab in enumerate(fcm)
        }

        palette = {
            mapping[lab]: color
            for lab, color in zip(fcm, label_pal)
        }

        figsize = (25, 3)
        figure, ax = plt.subplots(2, figsize=figsize)

        palplot(
            list(
                palette.values()
            ),
            ax=ax[0]
        )

        for i, label in enumerate(fcm):
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

            label = subset.fcm_label_2d.tolist()
            onset = subset.onset.tolist()
            offset = subset.offset.tolist()

            song = SongBarcode(
                onset=onset,
                offset=offset,
                label=label,
                mapping=mapping,
                palette=palette
            )

            transition, color = song.build()

            colors.append(color)
            transitions.append(transition)

        builder = Builder(
            ax=ax[1],
            color=colors,
            transition=transitions,
        )

        barcode = BarcodeFCM(
            builder=builder,
            settings=settings
        )

        barcode.build()

        if save:
            filename = f"barcode_{folder}.png"

            barcode.save(
                figure=figure,
                filename=filename,
            )
        else:
            barcode.show()


if __name__ == '__main__':
    main()
