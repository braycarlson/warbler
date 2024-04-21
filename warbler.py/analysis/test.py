from __future__ import annotations

import numpy as np

from constant import PROJECTION
from datatype.dataset import Dataset
from PIL import Image, ImageDraw, ImageFont
from skimage import data


def main() -> None:
    dataset = Dataset('segment')
    dataframe = dataset.load()

    filenames = dataframe.filename.unique()

    for filename in filenames:
        subset = dataframe.loc[dataframe.filename == filename]

        folder = subset.iloc[0].folder
        directory = PROJECTION.joinpath(folder)
        directory.mkdir(exist_ok=True, parents=True)

        ob = [
            Image.fromarray(o)
            for o in subset.original_array.tolist()
        ]

        fb = [
            Image.fromarray(f)
            for f in subset.filter_array.tolist()
        ]

        ob_horizontal = np.hstack(
            [np.array(note) for note in ob]
        )

        ob_horizontal = Image.fromarray(ob_horizontal)

        fb_horizontal = np.hstack(
            [np.array(note) for note in fb]
        )

        fb_horizontal = Image.fromarray(fb_horizontal)

        vertical = np.vstack(
            [
                np.array(ob_horizontal),
                np.array(fb_horizontal)
            ]
        )

        vertical = Image.fromarray(vertical)

        factor = 2

        width = vertical.width * factor
        height = vertical.height * factor
        size = (width, height)

        vertical = vertical.resize(size)

        font = ImageFont.truetype('arial.ttf', 48)

        bbox = font.getbbox(filename)

        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]

        padding = 50

        image = Image.new(
            'L',
            (vertical.width, vertical.height + height + padding),
            color=(255)
        )

        image.paste(
            vertical,
            (0, height + padding)
        )

        draw = ImageDraw.Draw(image)

        position = (
            (image.width - width) // 2,
            padding - (padding / 2)
        )

        draw.text(
            position,
            filename,
            font=font,
            fill=(0)
        )

        path = directory.joinpath(filename + '.png')
        image.save(path, format='png')


if __name__ == '__main__':
    main()
