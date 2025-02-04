from __future__ import annotations

import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from matplotlib.patches import Rectangle
from tqdm import tqdm
from warbler.constant import MASTER, OUTPUT
from warbler.datatype.axes import LinearAxes
from warbler.datatype.spectrogram import Linear, Spectrogram
from warbler.datatype.dataset import Dataset


def plot(iterable: zip) -> None:
    black = mcolors.to_rgba('black', alpha=1.0)

    iterable = list(iterable)
    total = len(iterable)

    progress = tqdm(iterable, total=total)

    for (folder, filename, signal, setting, onset, offset, label, sequence) in progress:
        fig = plt.figure(
            constrained_layout=True,
            figsize=(20, 4)
        )

        projection = LinearAxes(
            fig,
            [0, 0, 0, 0]
        )

        ax = fig.add_subplot(projection=projection)

        spectrogram = Spectrogram(strategy=Linear(signal, setting)).generate()
        extent = [0, signal.duration, 0, signal.rate / 2]

        ax.matshow(
            spectrogram,
            aspect='auto',
            cmap='Greys',
            extent=extent,
            interpolation=None,
            origin='lower'
        )

        ax.initialize()
        ax._x_lim(signal.duration)
        ax._y_lim(signal.rate / 2)
        ax._x_step(signal.duration)

        ylmin, ylmax = ax.get_ylim()
        ysize = (ylmax - ylmin) * 0.1
        ymin = ylmax - ysize

        palette = np.array(
            [MASTER[x] for x in label],
            dtype='float'
        )

        for index, (start, end) in enumerate(zip(onset, offset), 0):
            color, text = (
                (black, 'x')
                if index not in sequence
                else (
                    palette[sequence.index(index)],
                    label[sequence.index(index)]
                )
            )

            ax.axvline(
                start,
                color=color,
                ls='dashed',
                lw=1,
                alpha=1.0
            )

            ax.axvline(
                end,
                color=color,
                ls='dashed',
                lw=1,
                alpha=1.0
            )

            rectangle = Rectangle(
                xy=(start, ymin + 46),
                width=end - start,
                height=1000,
                alpha=1.0,
                color=color,
                label=str(index)
            )

            rx, ry = rectangle.get_xy()
            cx = rx + rectangle.get_width() / 2.0
            cy = ry + rectangle.get_height() / 2.0

            ax.annotate(
                text,
                (cx, cy),
                color='white',
                weight=600,
                fontfamily='Arial',
                fontsize=10,
                ha='center',
                va='center'
            )

            ax.add_patch(rectangle)

        plt.title(
            f"Fuzzy C-Means Clustering for {filename}",
            fontsize=20,
            fontweight=400,
            pad=25
        )

        path = OUTPUT.joinpath(folder)
        path.mkdir(exist_ok=True, parents=True)

        path = path.joinpath(filename + '.png')

        plt.savefig(
            path,
            bbox_inches='tight',
            dpi=300,
            format='png',
        )

        plt.close('all')


def main() -> None:
    plt.style.use('science')

    mpl.use("Agg")

    segment = Dataset('segment')
    signal = Dataset('signal')

    agg = {
        column: list
        for column in ['fcm_label_2d', 'sequence']
    }

    temporary = (
        segment
        .load()
        .groupby('filename', sort=False)
        .agg(agg)
    )

    temporary = temporary.reset_index()

    dataframe = signal.load()
    dataframe['label'] = temporary.fcm_label_2d.tolist()
    dataframe['sequence'] = temporary.sequence.tolist()

    del segment
    del temporary

    iterable = zip(
        dataframe.folder.tolist(),
        dataframe.filename.tolist(),
        dataframe.signal.tolist(),
        dataframe.settings.tolist(),
        dataframe.onset.tolist(),
        dataframe.offset.tolist(),
        dataframe.label.tolist(),
        dataframe.sequence.tolist()
    )

    plot(iterable)


if __name__ == '__main__':
    main()
