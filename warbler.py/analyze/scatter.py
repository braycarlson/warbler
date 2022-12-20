import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from constant import OUTPUT
from datatype.dataset import Dataset
from io import BytesIO
from matplotlib import lines
from matplotlib import gridspec
from PIL import Image
from scipy.spatial import cKDTree


def scatter_projections(
    projection=None,
    labels=None,
    ax=None,
    figsize=(10, 10),
    alpha=0.1,
    s=1,
    color='k',
    color_palette='tab20',
    categorical_labels=True,
    show_legend=True,
    tick_pos='bottom',
    tick_size=16,
    cbar_orientation='vertical',
    log_x=False,
    log_y=False,
    grey_unlabelled=True,
    fig=None,
    colornorm=False,
    rasterized=True,
):
    # color labels
    if labels is not None:
        if categorical_labels:
            if (color_palette == 'tab20') & (len(np.unique(labels)) < 20):
                pal = sns.color_palette(color_palette, n_colors=20)

                pal = np.array(pal)[
                    np.linspace(0, 19, len(np.unique(labels))).astype('int')
                ]
            else:
                pal = sns.color_palette(
                    color_palette,
                    n_colors=len(np.unique(labels))
                )

            lab_dict = {
                lab: pal[i]
                for i, lab in enumerate(np.unique(labels))
            }

            if grey_unlabelled:
                if -1 in lab_dict.keys():
                    lab_dict[-1] = [0.95, 0.95, 0.95, 1.0]

            colors = np.array([lab_dict[i] for i in labels], dtype='object')
    else:
        colors = color

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

        # plot
    if colornorm:
        norm = norm = matplotlib.colors.LogNorm()
    else:
        norm = None

    if categorical_labels or labels is None:
        ax.scatter(
            projection[:, 0],
            projection[:, 1],
            rasterized=rasterized,
            alpha=alpha,
            s=s,
            color=colors,
            norm=norm,
        )

    else:
        cmin = np.quantile(labels, 0.01)
        cmax = np.quantile(labels, 0.99)

        sct = ax.scatter(
            projection[:, 0],
            projection[:, 1],
            vmin=cmin,
            vmax=cmax,
            cmap=plt.get_cmap(color_palette),
            rasterized=rasterized,
            alpha=alpha,
            s=s,
            c=labels,
        )

    if log_x:
        ax.set_xscale('log')

    if log_y:
        ax.set_yscale('log')

    return ax


def scatter_spec(
    z,
    specs,
    column_size=15,
    pal_color='hls',
    scatter_kwargs={'alpha': 0.5, 's': 1},
    line_kwargs={'lw': 1, 'ls': 'dashed', 'alpha': 1},
    color_points=True,
    figsize=(10, 10),
    range_pad=0.1,
    x_range=None,
    y_range=None,
    enlarge_points=0,
    draw_lines=True,
    n_subset=-1,
    ax=None,
    show_scatter=True,
    border_line_width=1,
):
    n_columns = column_size * 4 - 4
    pal = sns.color_palette(pal_color, n_colors=n_columns)

    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(column_size, column_size)

    if x_range is None and y_range is None:
        xmin, xmax = np.sort(np.vstack(z)[:, 0])[
            np.array([int(len(z) * 0.01), int(len(z) * 0.99)])
        ]
        ymin, ymax = np.sort(np.vstack(z)[:, 1])[
            np.array([int(len(z) * 0.01), int(len(z) * 0.99)])
        ]

        xmin -= (xmax - xmin) * range_pad
        xmax += (xmax - xmin) * range_pad
        ymin -= (ymax - ymin) * range_pad
        ymax += (ymax - ymin) * range_pad
    else:
        xmin, xmax = x_range
        ymin, ymax = y_range

    x_block = (xmax - xmin) / column_size
    y_block = (ymax - ymin) / column_size

    # Ignore segments outside of range
    z = np.array(z)

    mask = np.array(
        [
            (z[:, 0] > xmin) & (z[:, 1] > ymin) &
            (z[:, 0] < xmax) & (z[:, 1] < ymax)
        ]
    )[0]

    if 'labels' in scatter_kwargs:
        scatter_kwargs['labels'] = np.array(scatter_kwargs['labels'])[mask]

    specs = np.array(specs)[mask]
    z = z[mask]

    # prepare the main axis
    main_ax = fig.add_subplot(
        gs[1:column_size - 1, 1:column_size - 1]
    )

    if show_scatter:
        scatter_projections(
            projection=z,
            ax=main_ax,
            fig=fig,
            **scatter_kwargs
        )

    # Loop through example columns
    axs = {}

    for column in range(n_columns):
        # Get example column location
        if column < column_size:
            row = 0
            col = column

        elif (column >= column_size) & (column < (column_size * 2) - 1):
            row = column - column_size + 1
            col = column_size - 1

        elif (column >= ((column_size * 2) - 1)) & (column < (column_size * 3 - 2)):
            row = column_size - 1
            col = column_size - 3 - (column - column_size * 2)
        elif column >= column_size * 3 - 3:
            row = n_columns - column
            col = 0

        axs[column] = {
            'ax': fig.add_subplot(gs[row, col]),
            'col': col,
            'row': row
        }

        # Sample a point in z based upon the row and column
        xpos = xmin + x_block * col + x_block / 2
        ypos = ymax - y_block * row - y_block / 2

        axs[column]['xpos'] = xpos
        axs[column]['ypos'] = ypos

    main_ax.set_xlim([xmin, xmax])
    main_ax.set_ylim([ymin, ymax])

    # Create a voronoi diagram over the x and y pos points
    points = [
        [
            axs[i]['xpos'],
            axs[i]['ypos']
        ]
        for i in axs.keys()
    ]

    voronoi_kdtree = cKDTree(points)

    # Find where each point lies in the voronoi diagram
    z = z[:n_subset]

    point_dist, point_regions = voronoi_kdtree.query(
        list(z)
    )

    lines_list = []

    # Loop through regions and select a point
    for key in axs.keys():
        # Sample a point in (or near) voronoi region
        nearest_points = np.argsort(np.abs(point_regions - key))
        possible_points = np.where(point_regions == point_regions[nearest_points][0])[0]
        selection = np.random.choice(a=possible_points, size=1)[0]
        point_regions[selection] = 1e4

        # Plot point
        if enlarge_points > 0:
            if color_points:
                color = pal[key]
            else:
                color = 'k'

            main_ax.scatter(
                [z[selection, 0]],
                [z[selection, 1]],
                color=color,
                s=enlarge_points,
            )

        # Draw spectrogram
        axs[key]['ax'].matshow(
            ~specs[selection],
            origin='lower',
            interpolation='bicubic',
            aspect='auto',
            cmap=plt.cm.Greys
        )

        axs[key]['ax'].set_xticks([])
        axs[key]['ax'].set_yticks([])

        if color_points:
            plt.setp(axs[key]['ax'].spines.values(), color=pal[key])

        for i in axs[key]['ax'].spines.values():
            i.set_linewidth(border_line_width)

        # Draw a line between point and image
        if draw_lines:
            mytrans = (
                axs[key]['ax'].transAxes + axs[key]['ax'].figure.transFigure.inverted()
            )

            line_end_pos = [0.5, 0.5]

            if axs[key]['row'] == 0:
                line_end_pos[1] = 0

            if axs[key]['row'] == column_size - 1:
                line_end_pos[1] = 1

            if axs[key]['col'] == 0:
                line_end_pos[0] = 1

            if axs[key]['col'] == column_size - 1:
                line_end_pos[0] = 0

            infig_position = mytrans.transform(line_end_pos)

            xpos, ypos = main_ax.transLimits.transform(
                (z[selection, 0], z[selection, 1])
            )

            mytrans2 = main_ax.transAxes + main_ax.figure.transFigure.inverted()
            infig_position_start = mytrans2.transform([xpos, ypos])

            color = pal[key] if color_points else 'k'

            lines_list.append(
                lines.Line2D(
                    [infig_position_start[0], infig_position[0]],
                    [infig_position_start[1], infig_position[1]],
                    color=color,
                    transform=fig.transFigure,
                    **line_kwargs,
                )
            )

    if draw_lines:
        for l in lines_list:
            fig.lines.append(l)

    gs.update(wspace=0, hspace=0)

    fig = plt.gcf()
    return fig, axs, main_ax, [xmin, xmax, ymin, ymax]


def to_numpy(image):
    buffer = BytesIO(image)
    image = Image.open(buffer)

    return np.array(image).astype('uint8')


def main():
    dataset = Dataset('segment')
    dataframe = dataset.load()

    spectrogram = (
        dataframe['resize']
        .apply(
            lambda x: to_numpy(x)
        )
    ).to_numpy()

    # spectrogram = dataframe['denoise'].to_numpy()

    # labels = dataframe.hdbscan_label_2d.to_numpy()
    # labels = dataframe.fcm_label_2d.to_numpy()

    coordinates = [
        dataframe.umap_x_2d,
        dataframe.umap_y_2d
    ]

    embedding = np.column_stack(coordinates)

    _, _, ax, _ = scatter_spec(
        embedding,
        spectrogram,
        column_size=15,
        pal_color='hls',
        color_points=True,
        enlarge_points=15,
        figsize=(10, 10),
        scatter_kwargs={
            # 'labels': labels,
            'alpha': 0.50,
            's': 1
        },
        line_kwargs={
            'lw': 1,
            'ls': 'solid',
            'alpha': 0.75,
        },
        draw_lines=True
    )

    title = f"Adelaide's warbler"

    ax.set_title(
        title,
        fontsize=24,
        pad=75
    )

    # plt.show()

    projection = OUTPUT.joinpath('projection')
    projection.mkdir(parents=True, exist_ok=True)

    filename = 'aw.png'
    path = projection.joinpath(filename)

    plt.savefig(
        path,
        dpi=300,
        format='png'
    )

    # dataset = Dataset('segment')
    # dataframe = dataset.load()

    # unique = dataframe.folder.unique()

    # for folder in unique:
    #     individual = dataframe[dataframe.folder == folder]

    #     spectrogram = (
    #         individual['resize']
    #         .apply(
    #             lambda x: to_numpy(x)
    #         )
    #     ).to_numpy()

    #     # spectrogram = individual['denoise'].to_numpy()

    #     labels = individual.hdbscan_label_2d.to_numpy()
    #     # labels = individual.fcm_label_2d.to_numpy()

    #     coordinates = [
    #         individual.umap_x_2d,
    #         individual.umap_y_2d
    #     ]

    #     embedding = np.column_stack(coordinates)

    #     _, _, ax, _ = scatter_spec(
    #         embedding,
    #         spectrogram,
    #         column_size=15,
    #         pal_color='hls',
    #         color_points=False,
    #         enlarge_points=10,
    #         figsize=(10, 10),
    #         scatter_kwargs={
    #             'labels': labels,
    #             'alpha': 0.75,
    #             's': 50
    #         },
    #         line_kwargs={
    #             'lw': 1,
    #             'ls': 'solid',
    #             'alpha': 0.50,
    #         },
    #         draw_lines=True
    #     )

    #     title = f"{folder}\nFuzzy Clustering"

    #     ax.set_title(
    #         title,
    #         fontsize=18,
    #         pad=75
    #     )

    #     # plt.show()

    #     projection = OUTPUT.joinpath('projection')
    #     projection.mkdir(parents=True, exist_ok=True)

    #     filename = f"{folder}.png"
    #     path = projection.joinpath(filename)

    #     plt.savefig(
    #         path,
    #         dpi=300,
    #         format='png'
    #     )


if __name__ == '__main__':
    main()
