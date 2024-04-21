from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from matplotlib.lines import Line2D
from constant import OUTPUT
from datatype.dataset import Dataset
from itertools import combinations
from tqdm import tqdm


def main() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    drop = [
        'duration',
        'fcm_label_2d',
        'fcm_label_3d',
        'folder',
        'filter_array',
        'filter_bytes',
        'hdbscan_label_2d',
        'hdbscan_label_3d',
        'maximum',
        'mean',
        'minimum',
        'offset',
        'onset',
        'original',
        'original_array',
        'original_bytes',
        'scale',
        'segment',
        'settings',
        'spectrogram',
        'umap_x_3d',
        'umap_y_3d',
        'umap_z_3d'
    ]

    dataframe = dataframe.drop(drop, axis=1)

    path = OUTPUT.joinpath('songtype.csv')
    songtype = pd.read_csv(path)

    group = songtype.groupby('songtype').filename.unique()

    unique = dataframe.sequence.unique()
    unique = sorted(unique)

    master = sns.color_palette(
        'deep',
        len(unique)
    )

    alpha = 0.75
    figsize = (20, 15)

    empty = Line2D(
        [0],
        [0],
        color='white',
        linestyle='None',
        marker='o',
        markerfacecolor='white',
        markersize=10
    )

    for songtype, songs in group.items():
        combination = combinations(songs, 2)
        pairs = list(combination)

        total = len(pairs)

        for pair in tqdm(pairs, total=total):
            subset = dataframe[dataframe.filename.isin(pair)]

            first, second = subset.filename.unique()
            title = f"Note Sequences for {first} and {second} \n"

            fig, ax = plt.subplots(figsize=figsize)

            x_min, x_max = (
                subset.umap_x_2d.min(),
                subset.umap_x_2d.max()
            )

            y_min, y_max = (
                subset.umap_y_2d.min(),
                subset.umap_y_2d.max()
            )

            x_pad = (x_max - x_min) * 0.05
            y_pad = (y_max - y_min) * 0.05

            palette = dict(
                zip(unique, master)
            )

            f_dataframe = subset[subset.filename == first]
            s_dataframe = subset[subset.filename == second]

            x_sequence = sorted(
                f_dataframe.sequence.unique()
            )

            y_sequence = sorted(
                s_dataframe.sequence.unique()
            )

            x_handles = [
                Line2D(
                    [0],
                    [0],
                    marker='o',
                    color=palette[sequence],
                    label=str(sequence),
                    linestyle='None',
                    markerfacecolor=palette[sequence],
                    markersize=10
                )
                for sequence in x_sequence
            ]

            x_handles.insert(
                0,
                Line2D(
                    [0],
                    [0],
                    color='blue',
                    label=first,
                    marker='o',
                    markeredgecolor='k',
                    markerfacecolor='k',
                    markersize=10
                )
            )

            x_handles.insert(1, empty)

            y_handles = [
                Line2D(
                    [0],
                    [0],
                    marker='x',
                    color=palette[sequence],
                    label=str(sequence),
                    linestyle='None',
                    markeredgewidth=4,
                    markerfacecolor=palette[sequence],
                    markersize=10
                )
                for sequence in y_sequence
            ]

            y_handles.insert(
                0,
                Line2D(
                    [0],
                    [0],
                    color='green',
                    label=second,
                    marker='x',
                    markeredgecolor='k',
                    markeredgewidth=4,
                    markerfacecolor='k',
                    markersize=10,
                )
            )

            y_handles.insert(1, empty)

            x_line, = ax.plot(
                f_dataframe.umap_x_2d,
                f_dataframe.umap_y_2d,
                'o-',
                color='blue'
            )

            y_line, = ax.plot(
                s_dataframe.umap_x_2d,
                s_dataframe.umap_y_2d,
                'x--',
                color='green'
            )

            sns.scatterplot(
                alpha=alpha,
                ax=ax,
                data=subset,
                hue='sequence',
                legend=False,
                linewidth=0.75,
                palette=palette,
                s=800,
                style='filename',
                x='umap_x_2d',
                y='umap_y_2d',
            )

            xlim = (x_min - x_pad, x_max + x_pad)
            ylim = (y_min - y_pad, y_max + y_pad)

            ax.set(
                xlabel=None,
                ylabel=None,
                xlim=xlim,
                ylim=ylim,
                xticklabels=[],
                yticklabels=[]
            )

            ax.set_title(title, fontsize=24)

            ax.tick_params(
                bottom=False,
                left=False
            )

            x_legend = ax.legend(
                bbox_to_anchor=(-0.01, 1.0065),
                handles=x_handles,
                loc='upper right',
                ncol=1
            )

            y_legend = ax.legend(
                bbox_to_anchor=(1.01, 1.0065),
                handles=y_handles,
                loc='upper left',
                ncol=1
            )

            ax.add_artist(x_legend)
            ax.add_artist(y_legend)

            for handle in x_legend.legendHandles:
                handle.set_alpha(alpha)

            for handle in y_legend.legendHandles:
                handle.set_alpha(alpha)

            plt.subplots_adjust(
                left=0.15,
                right=0.85
            )

            filename = f"static_tracing_{songtype}_{first}_{second}.png"
            path = OUTPUT.joinpath(filename)

            plt.savefig(
                path,
                dpi=300,
                format='png'
            )

            plt.close()


if __name__ == '__main__':
    main()
