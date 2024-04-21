from __future__ import annotations

import json
import matplotlib.pyplot as plt
import numpy as np

from constant import SETTINGS
from skimage import color
from matplotlib.colors import LinearSegmentedColormap


def main() -> None:
    tab20 = plt.cm.get_cmap('tab20', 20)

    shape = 3

    tab20_rgb = [
        tab20(i)[:shape]
        for i in range(20)
    ]

    tab20_lab = color.rgb2lab([tab20_rgb])

    new_colors_lab = []

    for i in range(20):
        new_colors_lab.append(
            tab20_lab[0, i, :]
        )

    for j in range(1, 5):
        for i in range(20):
            color_variation = np.copy(tab20_lab[0, i, :])

            color_variation[0] = color_variation[0] + (j * 10) % 100
            color_variation[0] = np.clip(color_variation[0], 0, 100)

            new_colors_lab.append(color_variation)

    new_colors = [
        color.lab2rgb([color_lab])
        for color_lab in new_colors_lab
    ]

    new_colors = [
        color[0]
        for color in new_colors
    ]

    colors_list = [
        list(map(float, color[:3]))
        for color in new_colors
    ]

    color_dict = {
        i: color
        for i, color in enumerate(colors_list)
    }

    path = SETTINGS.joinpath('palette.json')

    with open(path, 'w+') as handle:
        palette = json.dumps(color_dict, indent=4)
        handle.write(palette)

    cmap = LinearSegmentedColormap.from_list(
        'custom',
        colors_list,
        N=100
    )

    plt.imshow(
        [np.linspace(0, 1, 100)],
        aspect='auto',
        cmap=cmap
    )

    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
