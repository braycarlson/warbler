import seaborn as sns

from abc import ABC, abstractmethod
from IPython.display import Audio, display
from ipywidgets import (
    HBox,
    HTML,
    Image,
    interactive,
    Layout,
    Output,
    VBox
)
from plotly import graph_objs as go


class Visitor(ABC):
    @abstractmethod
    def two(self, dataframe):
        pass

    @abstractmethod
    def three(self, dataframe):
        pass


class SingleTrace(Visitor):
    def two(self, dataframe):
        return go.Scattergl(
            x=dataframe.umap_x_2d,
            y=dataframe.umap_y_2d,
            marker=dict(size=3),
            mode='markers',
            name=str(dataframe.index),
            showlegend=False
        )

    def three(self, dataframe):
        return go.Scatter3d(
            x=dataframe.umap_x_3d,
            y=dataframe.umap_y_3d,
            z=dataframe.umap_z_3d,
            marker=dict(size=3),
            mode='markers',
            name=str(dataframe.index),
            showlegend=False
        )


class MultipleTrace(Visitor):
    def two(self, dataframe):
        labels = dataframe.hdbscan_label_2d.unique()
        length = len(labels)

        palette = sns.color_palette('Paired', length).as_hex()

        iterable = zip(labels, palette)
        legend = dict(iterable)

        traces = []

        for label in labels:
            cluster = dataframe[dataframe.hdbscan_label_2d == label]
            color = legend[label]

            trace = go.Scattergl(
                x=cluster.umap_x_2d,
                y=cluster.umap_y_2d,
                mode='markers',
                marker=dict(
                    color=color,
                    size=3,
                ),
                name=str(label),
                showlegend=True
            )

            traces.append(trace)

        return traces

    def three(self, dataframe):
        labels = dataframe.hdbscan_label_3d.unique()
        labels.sort()

        length = len(labels)

        palette = sns.color_palette('Paired', length).as_hex()

        iterable = zip(labels, palette)
        legend = dict(iterable)

        traces = []

        for label in labels:
            cluster = dataframe[dataframe.hdbscan_label_3d == label]
            color = legend[label]

            trace = go.Scatter3d(
                x=cluster.umap_x_3d,
                y=cluster.umap_y_3d,
                z=cluster.umap_z_3d,
                mode='markers',
                marker=dict(
                    color=color,
                    size=3,
                ),
                name=str(label),
                showlegend=True
            )

            traces.append(trace)

        return traces


class DimensionalStrategy(ABC):
    @abstractmethod
    def create_scatter(self):
        pass

    @abstractmethod
    def create_scene(self, data):
        pass


class ThreeDimensional(DimensionalStrategy):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self._trace = None

    @property
    def trace(self):
        return self._trace

    @trace.setter
    def trace(self, trace):
        self._trace = trace

    def create_scatter(self):
        return self.trace.three(self.dataframe)

    def create_scene(self):
        return go.layout.Scene(
            xaxis=go.layout.scene.XAxis(title='x'),
            yaxis=go.layout.scene.YAxis(title='y'),
            zaxis=go.layout.scene.ZAxis(title='z')
        )


class TwoDimensional(DimensionalStrategy):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self._trace = None

    @property
    def trace(self):
        return self._trace

    @trace.setter
    def trace(self, trace):
        self._trace = trace

    def create_scatter(self):
        return self.trace.two(self.dataframe)

    def create_scene(self):
        return go.layout.Scene(
            xaxis=go.layout.scene.XAxis(title='x'),
            yaxis=go.layout.scene.YAxis(title='y')
        )


class Scatter:
    def __init__(self, dataframe):
        self.dataframe = dataframe

        self.audio = None
        self.description = None
        self.figure = None
        self.html = None
        self.layout = None
        self.player = None
        self.scatter = None
        self.scene = None
        self.widget = None

        self.height = 800
        self.width = 1000

        self._strategy = None

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy):
        self._strategy = strategy

    def create(self):
        self.create_scatter()
        self.create_scene()
        self.create_layout()
        self.create_figure()
        self.create_widget()
        self.create_description()
        self.create_html()
        self.create_audio()

        self.set_css()
        self.set_event_listener()

    def create_scatter(self):
        self.scatter = self.strategy.create_scatter()

    def create_scene(self):
        self.scene = self.strategy.create_scene()

    def create_audio(self):
        self.player = Output()
        self.player.layout.visibility = 'hidden'

        with self.player:
            self.audio = Audio(url='')
            display(self.audio)

    def create_description(self):
        value = (
            self.dataframe
            .loc[
                self.dataframe.index[0],
                ['resize']
            ]
            .values[0]
        )

        self.description = Image(value=value)

    def create_figure(self):
        self.figure = go.Figure(
            data=self.scatter,
            layout=self.layout
        )

    def create_html(self):
        column = [
            'folder',
            'filename',
            'sequence',
            'onset',
            'offset',
            'duration'
        ]

        value = (
            self.dataframe
            .loc[self.dataframe.index[0], column]
            .transpose()
            .to_frame()
            .to_html(
                classes='description',
                index=True,
                justify='center'
            )
        )

        self.html = HTML(value=value)

    def create_layout(self):
        self.layout = go.Layout(
            scene=self.scene,
            height=self.height,
            width=self.width,
            template='plotly_white',
            showlegend=True,
            legend=dict(
                itemsizing='constant',
                x=0.01,
                xanchor='right',
                y=0.99,
                yanchor='top'
            )
        )

    # def create_slider(self):
    #     self.slider = interactive(
    #         self.on_slide,
    #         opacity=(0.0, 1.0, 0.01),
    #         size=(1, 10, 0.25)
    #     )

    #     # Opacity
    #     self.slider.children[0].default_value = 0.75
    #     self.slider.children[0].value = 0.75
    #     self.slider.children[0].layout.width = str(self.width / 2) + 'px'
    #     self.slider.children[0].description = 'Opacity'

    #     # Size
    #     self.slider.children[1].default_value = 2.50
    #     self.slider.children[1].value = 2.50
    #     self.slider.children[1].layout.width = str(self.width / 2) + 'px'
    #     self.slider.children[1].description = 'Size'

    def create_widget(self):
        self.widget = go.FigureWidget(self.figure)

    def get_plot(self):
        left = VBox(
            [self.description, self.html],
            layout=Layout(
                align_items='center',
                display='flex',
                flex_flow='column',
                align_content='stretch',
                justify_content='center'
            )
        )

        left.add_class('image')

        right = VBox(
            [self.widget],
            layout=Layout(
                align_items='center',
                display='flex',
                width='100%'
            )
        )

        box = VBox(
            [
                HBox(
                    [left, right]
                ),
                self.player
            ],
            layout=Layout(
                align_items='center',
                flex='none',
                width='100%'
            )
        )

        return box

    def on_click(self, trace, points, selector):
        index = points.point_inds

        if index:
            index = index[0]
            self.play(index)

    def on_hover(self, trace, points, state):
        index = points.point_inds

        if index:
            column = [
                'folder',
                'filename',
                'sequence',
                'onset',
                'offset',
                'duration'
            ]

            self.html.value = (
                self.dataframe
                .loc[self.dataframe.index[index], column]
                .transpose()
                .to_html(
                    classes='description',
                    index=True,
                    justify='center'
                )
            )

            spectrogram = (
                self.dataframe
                .loc[
                    self.dataframe.index[index],
                    ['resize']
                ]
                .values[0]
            )

            self.description.value = spectrogram[0]

    # def on_slide(self, opacity, size):
    #     self.widget.data[0].marker.opacity = opacity
    #     self.widget.data[0].marker.size = size

    def play(self, index):
        segment = self.dataframe.loc[
            self.dataframe.index[index],
            'segment'
        ]

        with self.player:
            self.player.clear_output(True)

            self.audio = Audio(
                data=segment.data,
                rate=segment.rate,
                autoplay=True
            )

            display(self.audio)

    def set_css(self):
        css = '''
            <style>
                .image > .widget-image {
                    width: 99%; !important;
                }

                .description {
                    background-color: transparent !important;
                    border: 1px solid black; !important;
                    width: 100% !important;
                }

                .description td, .description th {
                    border: none; !important;
                    padding: 5px 10px; !important;
                    text-align: center !important;
                }

                .js-plotly-plot .plotly .cursor-crosshair,
                .js-plotly-plot .plotly .cursor-move {
                    cursor: default !important;
                }
            </style>
        '''

        html = HTML(css)
        display(html)

    def set_event_listener(self):
        for trace in self.widget.data:
            trace.on_hover(self.on_hover)
            trace.on_click(self.on_click)
