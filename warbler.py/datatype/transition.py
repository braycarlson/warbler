import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import numpy as np


def draw_projection_transitions(
    projections,
    sequences,
    ax=None,
    nseq=-1,
    cmap=plt.get_cmap('cubehelix'),
    alpha=0.05,
    linewidth=3,
    range_pad=0.1,
):
    if ax is None:
        figsize = (10, 10)
        fig, ax = plt.subplots(figsize=figsize)

    for sequence in np.unique(sequences):
        mask = sequences == sequence
        projection_seq = projections[mask]

        colorline(
            projection_seq[:, 0],
            projection_seq[:, 1],
            ax,
            np.linspace(0, 1, len(projection_seq)),
            cmap=cmap,
            linewidth=linewidth,
            alpha=alpha,
        )

    xmin, xmax = np.sort(np.vstack(projections)[:, 0])[
        np.array(
            [
                int(len(projections) * 0.01),
                int(len(projections) * 0.99)
            ]
        )
    ]

    ymin, ymax = np.sort(np.vstack(projections)[:, 1])[
        np.array(
            [
                int(len(projections) * 0.01),
                int(len(projections) * 0.99)
            ]
        )
    ]

    xmin -= (xmax - xmin) * range_pad
    xmax += (xmax - xmin) * range_pad
    ymin -= (ymax - ymin) * range_pad
    ymax += (ymax - ymin) * range_pad

    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))

    return ax


def colorline(
    x,
    y,
    ax,
    z=None,
    cmap=plt.get_cmap('copper'),
    norm=plt.Normalize(0.0, 1.0),
    linewidth=1,
    alpha=1.0,
):
    # Default colors equally spaced on [0, 1]:
    if z is None:
        z = np.linspace(
            0.0,
            1.0,
            len(x)
        )

    # Special case if a single number:
    # to check for numerical input -- this is a hack
    if not hasattr(z, '__iter__'):
        z = np.array([z])

    z = np.asarray(z)

    line = [x, y]
    points = np.array(line).T.reshape(-1, 1, 2)

    segments = np.concatenate(
        [
            points[:-1],
            points[1:]
        ],
        axis=1
    )

    collection = mcoll.LineCollection(
        segments,
        array=z,
        cmap=cmap,
        norm=norm,
        linewidth=linewidth,
        alpha=alpha
    )

    ax.add_collection(collection)

    return collection
