from __future__ import annotations

import matplotlib as mpl
mpl.use('Agg')

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import scienceplots

from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from pathlib import Path
from typing import TYPE_CHECKING
from warbler.datatype.builder import Base, Plot
from warbler.constant import SETTINGS
from warbler.datatype.dataset import Dataset
from warbler.datatype.settings import Settings

if TYPE_CHECKING:
    import numpy.typing as npt
    import pandas as pd

    from matplotlib.image import AxesImage
    from typing_extensions import Any, Self


class Builder(Base):
    def __init__(
        self,
        component: dict[Any, Any] | None = None,
        dataframe: pd.DataFrame | None = None,
        embedding: npt.NDArray | None = None,
        label: npt.NDArray | None = None,
        sequence: npt.NDArray | None = None,
        settings: Settings | None = None,
        spectrogram: npt.NDArray | None = None,
        unique: npt.NDArray | None = None,
    ):
        super().__init__(
            component,
            embedding,
            label,
            sequence,
            settings,
            spectrogram,
            unique
        )

        self.dataframe = dataframe

        self.unit = {
            'onset': 's',
            'offset': 's',
            'duration': 's',
            'minimum': 'Hz',
            'mean': 'Hz',
            'maximum': 'Hz'
        }

    def ax(self) -> Self:
        figsize = (14, 16)
        figure = plt.figure(figsize=figsize)

        gs = GridSpec(
            3,
            2,
            height_ratios=[1, 3, 2],
            width_ratios=[1, 3]
        )

        # Projection
        ax1 = plt.subplot(gs[1:, :])

        # Spectrogram
        ax2 = plt.subplot(gs[0, 0])

        ax2.set_title(
            'Image',
            fontsize=24,
            pad=90
        )

        ax2.axis('off')

        # Table
        ax3 = plt.subplot(gs[0, 1])

        plt.subplots_adjust(
            hspace=1.0,
            wspace=1.0
        )

        self.component['ax'] = ax1
        self.component['ax2'] = ax2
        self.component['ax3'] = ax3
        self.component['figure'] = figure

        return self

    def legend(self) -> Self:
        condition = (
            not self.settings.is_legend or
            not self.settings.is_color
        )

        if condition:
            return self

        ax = self.component.get('ax')
        handles = self.component.get('handles')

        ax.legend(
            handles=handles,
            fontsize=16,
            **self.settings.legend
        )

        return self

    def line(self) -> Self:
        condition = (
            not self.settings.is_legend or
            not self.settings.is_color
        )

        if condition:
            return self

        label = self.component.get('label')

        label = {
            chr(k + 65): v
            for k, v in label.items()
        }

        handles = [
            Line2D(
                [0],
                [0],
                color=color,
                label=label,
                **self.settings.line
            )
            for label, color in label.items()
        ]

        self.component['handles'] = handles

        return self

    def scatter(self) -> Self:
        ax = self.component.get('ax')

        if self.settings.is_color:
            color = self.component.get('color')
            self.settings.scatter['color'] = color

        if self.settings.is_legend:
            label = self.component.get('label')
            self.settings.scatter['label'] = label

        scatter_plot = ax.scatter(
            self.embedding[:, 0],
            self.embedding[:, 1],
            **self.settings.scatter
        )

        ax.set(
            xlabel=None,
            ylabel=None,
            xticklabels=[],
            yticklabels=[]
        )

        ax.tick_params(
            bottom=False,
            left=False
        )

        self.component['scatter_plot'] = scatter_plot

        spectrogram_plot = self.component['ax2'].matshow(
            self.spectrogram[0],
            aspect='auto',
            cmap='Greys'
        )

        self.component['spectrogram_plot'] = spectrogram_plot

        return self

    def table(self, i: int) -> Self:
        ax = self.component.get('ax3')

        ax.cla()

        columns = [
            'folder',
            'filename',
            'sequence',
            'onset',
            'offset',
            'duration',
            'minimum',
            'mean',
            'maximum'
        ]

        copy = self.dataframe.copy()
        features = ['onset', 'offset', 'duration', 'minimum', 'mean', 'maximum']

        copy[features] = copy[features].round(2)

        information = copy.loc[i, columns]

        text = []

        for column, value in zip(columns, information):
            if column in features:
                text.append(
                    (column, str(value) + self.unit[column])
                )
            else:
                text.append(
                    (column, value)
                )

        table = ax.table(
            cellLoc='center',
            cellText=text,
            loc='center'
        )

        table.auto_set_font_size(value=False)
        table.set_fontsize(16)
        table.scale(1.5, 2.0)

        for (row, _), cell in table.get_celld().items():
            if row % 2 == 0:
                cell.set_facecolor('white')
            else:
                cell.set_facecolor(self.settings.row)

        ax.axis('tight')

        ax.set_title(
            'Acoustic Features',
            fontsize=24,
            pad=90
        )

        ax.set(
            xlabel=None,
            ylabel=None,
            xticklabels=[],
            yticklabels=[]
        )

        ax.tick_params(
            bottom=False,
            left=False
        )

        ax.axis('off')

        return self

    def update(self, i: int) -> tuple[Line2D, AxesImage]:
        print(f"Processing: frame {i}")

        embedding = self.embedding[:i]
        spectrogram = self.spectrogram[i]

        self.component['scatter_plot'].set_offsets(embedding)
        self.component['spectrogram_plot'].set_data(spectrogram)

        self.table(i)

        return (
            self.component['scatter_plot'],
            self.component['spectrogram_plot']
        )


    def title(self) -> Self:
        ax = self.component.get('ax')

        name = self.settings.name
        cluster = self.settings.cluster

        title = f"{cluster} Clustering for {name}"

        ax.set_title(
            title,
            fontsize=24,
            pad=25
        )

        return self


class ScatterHDBSCAN(Plot):
    def __init__(self, builder: Any = None):
        super().__init__(builder)

        self.builder = builder

    def animate(self) -> Any:
        cluster = {'cluster': 'HDBSCAN'}
        self.builder.settings.update(cluster)

        length = len(self.builder.embedding)
        frames = range(length)

        for i in frames:
            plot = (
                self
                .builder
                .ax()
                .title()
                .palette()
                .line()
                .scatter()
                .legend()
                .get()
            )

            figure = plot.get('figure')

            self.builder.update(i)

            plt.subplots_adjust(
                left=0.05,
                right=0.85,
                bottom=0.025
            )

            i = str(i).zfill(2)

            figure.savefig(
                f"frames/hdbscan/frame_{i}.png",
                pad_inches=0.5
            )

            plt.close()


def main() -> None:
    # plt.style.use('science')

    dataset = Dataset('segment')
    dataframe = dataset.load()

    drop = [
        'hdbscan_label_3d',
        'filter_bytes',
        'fcm_label_2d',
        'fcm_label_3d',
        'original_bytes',
        'scale',
        'segment',
        'spectrogram',
        'umap_z_3d'
    ]

    dataframe = dataframe.drop(drop, axis=1)

    # Mask the "noise"
    dataframe = (
        dataframe[dataframe.hdbscan_label_2d > -1]
        .reset_index(drop=True)
    )

    order = [2, 1, 12, 13, 11, 5, 15, 14, 4, 9, 10, 17, 18, 16, 3, 6, 7, 8]

    order_dict = {
        k: i
        for i, k in enumerate(order)
    }

    dataframe['hdbscan_label_2d_order'] = dataframe['hdbscan_label_2d'].map(order_dict)

    dataframe = dataframe.sort_values(by='hdbscan_label_2d_order')

    dataframe = dataframe.groupby('hdbscan_label_2d_order').apply(
        lambda x: x.sort_values('mean')
    ).reset_index(drop=True)

    dataframe = dataframe.drop('hdbscan_label_2d_order', axis=1)
    dataframe = dataframe.reset_index(drop=True)

    # Load default settings
    path = SETTINGS.joinpath('scatter.json')
    settings = Settings.from_file(path)

    # Change the default to accommodate a larger dataset
    color = 'gainsboro'
    opacity = 0.5
    color = mcolors.to_rgba(color)
    row = color[:3] + (opacity,)

    settings['scatter']['alpha'] = 0.25
    settings['scatter']['s'] = 50
    settings['row'] = row

    frames = Path.cwd().joinpath('frames/hdbscan')
    frames.mkdir(exist_ok=True, parents=True)

    unique = dataframe.hdbscan_label_2d.unique()

    coordinates = [
        dataframe.umap_x_2d,
        dataframe.umap_y_2d
    ]

    embedding = np.column_stack(coordinates)
    label = dataframe.hdbscan_label_2d.to_numpy()
    spectrogram = dataframe.filter_array.to_numpy()

    builder = Builder(
        dataframe=dataframe,
        embedding=embedding,
        label=label,
        settings=settings,
        spectrogram=~spectrogram,
        unique=unique,
    )

    scatter = ScatterHDBSCAN(builder=builder)
    scatter.builder.component['color'] = label

    scatter.animate()


if __name__ == '__main__':
    main()
