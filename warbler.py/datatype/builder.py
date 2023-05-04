import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import warnings

from abc import abstractmethod
from constant import PROJECTION
from matplotlib.backends._backend_tk import FigureManagerTk

warnings.simplefilter('ignore', UserWarning)


class Component():
    def __init__(self):
        self.collection = {}


class Base:
    def __init__(self, settings=None):
        self._component = Component()
        self._embedding = None
        self._label = None
        self._sequence = None
        self._settings = settings
        self._spectrogram = None
        self._unique = None

    @property
    def component(self):
        return self._component

    @property
    def embedding(self):
        return self._embedding

    @property
    def label(self):
        return self._label

    @property
    def sequence(self):
        return self._sequence

    @property
    def settings(self):
        return self._settings

    @property
    def spectrogram(self):
        return self._spectrogram

    @property
    def unique(self):
        return self._unique

    @embedding.setter
    def embedding(self, embedding):
        self._embedding = embedding

    @label.setter
    def label(self, label):
        self._label = label

    @sequence.setter
    def sequence(self, sequence):
        self._sequence = sequence

    @settings.setter
    def settings(self, settings):
        self._settings = settings

    @spectrogram.setter
    def spectrogram(self, spectrogram):
        self._spectrogram = spectrogram

    @unique.setter
    def unique(self, unique):
        self._unique = unique

    def get(self):
        return self.component

    def palette(self):
        if self.unique is None:
            self.unique = np.unique(self.label)

        self.unique.sort()
        n_colors = len(self.unique)

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

        self.component.collection['color'] = color
        self.component.collection['label'] = label
        self.component.collection['palette'] = palette

        return self


class Plot:
    def __init__(self):
        self._builder = None
        self._embedding = None
        self._label = None
        self._sequence = None
        self._spectrogram = None
        self._settings = None

    @property
    def builder(self):
        return self._builder

    @property
    def settings(self):
        return self.builder.settings

    @builder.setter
    def builder(self, builder):
        self._builder = builder

    @Base.embedding.setter
    def embedding(self, embedding):
        self.builder.embedding = embedding

    @Base.label.setter
    def label(self, label):
        self.builder.label = label

    @Base.sequence.setter
    def sequence(self, sequence):
        self.builder.sequence = sequence

    @settings.setter
    def settings(self, settings):
        self.builder.settings = settings

    @Base.spectrogram.setter
    def spectrogram(self, spectrogram):
        self.builder.spectrogram = spectrogram

    @Base.unique.setter
    def unique(self, unique):
        self.builder.unique = unique

    @abstractmethod
    def construct(self):
        pass

    def save(self, filename):
        PROJECTION.mkdir(
            parents=True,
            exist_ok=True
        )

        path = PROJECTION.joinpath(filename)

        plt.savefig(
            path,
            bbox_inches='tight',
            dpi=300,
            format='png'
        )

        plt.close()

    def show(self):
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
