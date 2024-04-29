from __future__ import annotations

import os
import subprocess

from natsort import os_sorted
from pathlib import Path


def main() -> None:
    cluster = 'hdbscan'

    if os.name == 'posix':
        folder = Path.cwd().joinpath('ffmpeg')

        current = os.environ.get('PATH', '')
        os.environ['PATH'] = f"{folder}:{current}"

    frames = Path.cwd().joinpath('frames', cluster)
    frames.mkdir(exist_ok=True, parents=True)

    files = [
        file
        for file in frames.glob('*.png')
        if file.is_file()
    ]

    files = os_sorted(files)

    filelist = frames.joinpath('frames.txt')

    with open(filelist, 'w') as handle:
        for file in files:
            file = file.as_posix()
            handle.write(f"file '{file}'\n")
            handle.write('duration 0.1\n')

    # .mp4
    subprocess.run(
        [
            'ffmpeg',
            '-y',
            '-r',
            '15',
            '-f',
            'concat',
            '-safe',
            '0',
            '-i',
            f"{filelist}",
            'dataset.mp4'
        ]
    )


if __name__ == '__main__':
    main()
