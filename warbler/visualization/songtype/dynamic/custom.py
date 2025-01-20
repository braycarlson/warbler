from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from itertools import combinations
from tqdm import tqdm
from typing_extensions import TYPE_CHECKING
from warbler.constant import OUTPUT
from warbler.datatype.dataset import Dataset

if TYPE_CHECKING:
    import pandas as pd


def determine_group(row: pd.Series) -> str | np.nan:
    group_a = [
        'STE02a_DbWY2017',
        'STE05_2_DgDgY2017',
        '1_STE01_LbLgLb2017',
        'STE11_LgLLb2017',
        'STE19_LLbLg2017',
        'STE08_2_OPR2017',
        '1_STE06_RbRY2017',
        'STE06_RbTbL2017',
        'STE05_RRO2017',
        'STE15_1_TbWLb2017',
        'STE02_YO2017',
        'STE04_YWW2017',
        'STE02_2_YYLb2017'
    ]

    group_b = [
        'STE12_DbWY2017',
        'STE08_DgDgY2017',
        'STE04_1_OPR2017',
        '1_STE10_RbRY2017',
        'STE05_YO2017',
        'STE11_2_YYLb2017'
    ]

    group_c = [
        'STE23_DgDgY2017',
        '1_STE23_LbLgLb2017',
        'STE03_1_LgLLb2017',
        'STE02_LLbLg2017',
        'STE11_OPR2017',
        'STE07_RbRY2017',
        'STE13_1_YYLb2017'
    ]

    if row['filename'] in group_a:
        return 'a'
    elif row['filename'] in group_b:
        return 'b'
    elif row['filename'] in group_c:
        return 'c'
    else:
        return np.nan


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

    dataframe['group'] = dataframe.apply(determine_group, axis=1)

    group = dataframe.groupby('group').filename.unique()

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
        grouping = OUTPUT.joinpath(songtype)
        grouping.mkdir(exist_ok=True, parents=True)

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
                [],
                [],
                'o-',
                color='blue'
            )

            y_line, = ax.plot(
                [],
                [],
                'x--',
                color='green'
            )

            def animate(i: int) -> None:
                xdf = f_dataframe.umap_x_2d[:i]
                ydf = f_dataframe.umap_y_2d[:i]

                xds = s_dataframe.umap_x_2d[:i]
                yds = s_dataframe.umap_y_2d[:i]

                x_line.set_data(xdf, ydf)
                y_line.set_data(xds, yds)

                if len(ax.collections) > 0:
                    ax.collections[0].remove()

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

            animation = FuncAnimation(
                fig,
                animate,
                frames=len(subset),
                interval=200
            )

            plt.subplots_adjust(
                left=0.15,
                right=0.85
            )

            filename = f"dynamic_tracing_{songtype}_{first}_{second}.gif"
            path = grouping.joinpath(filename)

            animation.save(
                path,
                writer='pillow',
                fps=5
            )

            plt.close()


if __name__ == '__main__':
    main()
