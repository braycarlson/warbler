from abc import ABC, abstractmethod
from IPython.display import Audio, display
from ipywidgets import (
    HBox,
    HTML,
    Image,
    Layout,
    Output,
    VBox
)
from plotly import graph_objs as go


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

        if not index:
            return

        zero = 0
        index = index[zero]
        indices = trace.customdata[zero]
        i = indices[index]

        self.play(i)

    def on_hover(self, trace, points, state):
        index = points.point_inds

        if not index:
            return

        column = [
            'folder',
            'filename',
            'sequence',
            'onset',
            'offset',
            'duration'
        ]

        zero = 0
        index = index[zero]
        indices = trace.customdata[zero]
        i = indices[index]

        self.html.value = (
            self.dataframe
            .loc[self.dataframe.index[[i]], column]
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
                self.dataframe.index[[i]],
                ['resize']
            ]
            .values[zero]
        )

        self.description.value = spectrogram[zero]

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
