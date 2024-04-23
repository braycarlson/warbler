from __future__ import annotations

import contextlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings

from abc import abstractmethod
from typing import TYPE_CHECKING
from warbler.constant import PROJECTION

with contextlib.suppress(ModuleNotFoundError):
    from matplotlib.backends._backend_tk import FigureManagerTk

if TYPE_CHECKING:
    import numpy.typing as npt

    from matplotlib.figure import Figure
    from typing_extensions  import Any, Self
    from warbler.datatype.settings import Settings


warnings.simplefilter(
    'ignore',
    UserWarning
)


class Base:
    def __init__(
        self,
        component: dict[Any, Any] | None = None,
        embedding: npt.NDArray | None = None,
        label: npt.NDArray | None = None,
        sequence: npt.NDArray | None = None,
        settings: Settings | None = None,
        spectrogram: npt.NDArray | None = None,
        unique: npt.NDArray | None = None,
    ):
        if component is None:
            component = {}

        self.component = component
        self.embedding = embedding
        self.label = label
        self.sequence = sequence
        self.settings = settings
        self.spectrogram = spectrogram
        self.unique = unique

    def get(self) -> dict[Any, Any] | None:
        return self.component

    def palette(self) -> Self:
        if self.label is None:
            return self

        if self.unique is None:
            self.unique = np.unique(self.label)

        if self.unique.dtype == np.int64:
            self.unique.sort()

        n_colors = len(self.unique)

        # palette =  [
        #     viridis(i / (n_colors - 1))
        #     for i in range(n_colors)
        # ]

        palette = sns.color_palette(
            self.settings.palette,
            n_colors=n_colors
        )

        master = dict(
            zip(self.unique, palette)
        )

        label = {
            label: master[label]
            for label in self.label
        }

        if self.label.dtype == np.int64:
            label = dict(
                sorted(
                    label.items()
                )
            )

        if -1 in label:
            label[-1] = (0.85, 0.85, 0.85)

        color = np.array(
            [label[i] for i in self.label],
            dtype='float'
        )

        self.component['color'] = color
        self.component['label'] = label
        self.component['palette'] = palette

        return self


class Plot:
    def __init__(self, builder: Any = None):
        self.builder = builder

    @property
    def settings(self) -> dict[Any, Any] | Settings | None:
        return self.builder.settings

    @settings.setter
    def settings(self, settings: dict[Any, Any] | Settings) -> None:
        self.builder.settings = settings

    @property
    def embedding(self) -> npt.NDArray | None:
        return self.builder.embedding

    @embedding.setter
    def embedding(self, embedding: npt.NDArray) -> None:
        self.builder.embedding = embedding

    @property
    def label(self) -> npt.NDArray | None:
        return self.builder.label

    @label.setter
    def label(self, label: npt.NDArray) -> None:
        self.builder.label = label

    @property
    def sequence(self) -> npt.NDArray | None:
        return self.builder.sequence

    @sequence.setter
    def sequence(self, sequence: npt.NDArray) -> None:
        self.builder.sequence = sequence

    @property
    def spectrogram(self) -> npt.NDArray | None:
        return self.builder.spectrogram

    @spectrogram.setter
    def spectrogram(self, spectrogram: npt.NDArray) -> None:
        self.builder.spectrogram = spectrogram

    @property
    def unique(self) -> npt.NDArray | None:
        return self.builder.unique

    @unique.setter
    def unique(self, unique: npt.NDArray) -> None:
        self.builder.unique = unique

    @abstractmethod
    def build(self) -> dict[Any, Any]:
        raise NotImplementedError

    def save(
        self,
        figure: Figure | None = None,
        filename: str | None = None,
        transparent: bool = True
    ) -> None:
        if filename is None:
            filename = 'plot.png'

        PROJECTION.mkdir(
            parents=True,
            exist_ok=True
        )

        path = PROJECTION.joinpath(filename)

        if figure is None:
            plt.savefig(
                path,
                bbox_inches='tight',
                dpi=300,
                format='png',
                transparent=transparent
            )
        else:
            figure.savefig(
                path,
                bbox_inches='tight',
                dpi=300,
                format='png',
                transparent=transparent
            )

        plt.close()

    def show(self) -> None:
        manager = plt.get_current_fig_manager()

        if isinstance(manager, FigureManagerTk):
            w, h = self.builder.settings.figure.get('figsize')
            w = w * 96
            h = h * 96

            width = manager.window.winfo_screenwidth()
            height = manager.window.winfo_screenheight()

            x = (width / 2) - (w / 2)
            x = int(x)

            y = (height / 2) - (h / 2)
            y = int(y)

            geometry = f"{w}x{h}+{x}+{y}"
            manager.window.wm_geometry(geometry)

        plt.tight_layout()
        plt.show()
        plt.close()
