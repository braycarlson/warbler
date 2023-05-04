import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from collections import defaultdict
from datatype.builder import Base, Plot
from datatype.settings import Settings
from matplotlib.lines import Line2D


class Builder(Base):
    def ax(self):
        figsize = self.settings.figure.get('figsize')
        fig, ax = plt.subplots(figsize=figsize)

        ax.xaxis.set_visible(self.settings.is_axis)
        ax.yaxis.set_visible(self.settings.is_axis)

        self.component.collection['ax'] = ax

        return self

    def decorate(self):
        default = {
            'figure': {
                'figsize': (9, 8)
            },
            'legend': {
                'borderaxespad': 0,
                'bbox_to_anchor': (1.15, 1.00),
            },
            'line': {
                'marker': 'o',
                'rasterized': False
            },
            'scatter': {
                'alpha': 0.50,
                'color': 'black',
                'label': None,
                'rasterized': False,
                's': 10
            },
            'cluster': 'HDBSCAN',
            'is_axis': False,
            'is_cluster': True,
            'is_color': True,
            'is_legend': True,
            'is_title': True,
            'name': 'Adelaide\'s warbler',
            'palette': 'tab20',
        }

        if self.settings is not None:
            merge = defaultdict(dict)
            merge.update(default)

            for key, value in self.settings.items():
                if isinstance(value, dict):
                    merge[key].update(value)
                else:
                    merge[key] = value

            default = dict(merge)

        self.settings = Settings.from_dict(default)

        return self

    def legend(self):
        condition = (
            not self.settings.is_legend or
            not self.settings.is_color
        )

        if condition:
            return self

        ax = self.component.collection.get('ax')
        handles = self.component.collection.get('handles')

        ax.legend(
            handles=handles,
            **self.settings.legend
        )

        return self

    def line(self):
        condition = (
            not self.settings.is_legend or
            not self.settings.is_color
        )

        if condition:
            return self

        label = self.component.collection.get('label')

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

        self.component.collection['handles'] = handles

        return self

    def scatter(self):
        ax = self.component.collection.get('ax')

        if self.settings.is_color:
            color = self.component.collection.get('color')
            self.settings.scatter['color'] = color

        if self.settings.is_legend:
            label = self.component.collection.get('label')
            self.settings.scatter['label'] = label

        ax.scatter(
            self.embedding[:, 0],
            self.embedding[:, 1],
            **self.settings.scatter
        )

        return self

    def title(self):
        ax = self.component.collection.get('ax')

        name = self.settings.name
        cluster = self.settings.cluster

        title = f"{cluster} Clustering of {name}"
        ax.set_title(title)

        return self


class ScatterHDBSCAN(Plot):
    def construct(self):
        self.builder.settings['cluster'] = 'HDBSCAN'

        return (
            self.builder
            .decorate()
            .ax()
            .title()
            .palette()
            .line()
            .scatter()
            .legend()
            .get()
        )


class ScatterFCM(Plot):
    def construct(self):
        self.builder.settings['cluster'] = 'Fuzzy C-Means'

        return (
            self.builder
            .decorate()
            .ax()
            .title()
            .palette()
            .line()
            .scatter()
            .legend()
            .get()
        )
