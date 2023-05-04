from abc import ABC, abstractmethod
from datatype.trace import AnimationTrace
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
    def __init__(self, dataframe=None, trace=None):
        self._dataframe = dataframe
        self._trace = trace

    @property
    def dataframe(self):
        return self._dataframe

    @dataframe.setter
    def dataframe(self, dataframe):
        self._dataframe = dataframe

    @property
    def trace(self):
        return self._trace

    @trace.setter
    def trace(self, trace):
        self._trace = trace

    @abstractmethod
    def scatter(self):
        pass

    @abstractmethod
    def scene(self, data):
        pass


class ThreeDimensional(DimensionalStrategy):
    def scatter(self):
        return self.trace.three(self.dataframe)

    def scene(self):
        return go.layout.Scene(
            xaxis=go.layout.scene.XAxis(title='x'),
            yaxis=go.layout.scene.YAxis(title='y'),
            zaxis=go.layout.scene.ZAxis(title='z')
        )


class TwoDimensional(DimensionalStrategy):
    def scatter(self):
        return self.trace.two(self.dataframe)

    def scene(self):
        return go.layout.Scene(
            xaxis=go.layout.scene.XAxis(title='x'),
            yaxis=go.layout.scene.YAxis(title='y')
        )


class Builder:
    def __init__(self, dataframe=None, strategy=None):
        self._dataframe = dataframe
        self._strategy = strategy

        self.plot = Plot()

        self.height = 800
        self.width = 1000

    @property
    def dataframe(self):
        return self._dataframe

    @dataframe.setter
    def dataframe(self, dataframe):
        self._dataframe = dataframe

    @property
    def strategy(self):
        return self._strategy

    @strategy.setter
    def strategy(self, strategy):
        self._strategy = strategy

    def animation(self):
        if not isinstance(self.strategy.trace, AnimationTrace):
            return self

        figure = self.plot.component.get('figure')

        self.strategy.trace.animate()
        figure.frames = self.strategy.trace.frames

        figure.update_layout(
            scene_camera=dict(
                eye=dict(
                    x=self.strategy.trace.x_eye,
                    y=self.strategy.trace.y_eye,
                    z=self.strategy.trace.z_eye
                )
            ),
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=True,
                    y=1,
                    x=0.1,
                    xanchor='left',
                    yanchor='top',
                    pad=dict(t=45, r=10),
                    buttons=[
                        dict(
                            label='Rotate',
                            method='animate',
                            args=[
                                None,
                                dict(
                                    frame=dict(
                                        duration=0.5,
                                        redraw=True
                                    ),
                                    transition=dict(
                                        duration=0,
                                        easing='linear'
                                    ),
                                    fromcurrent=True,
                                    mode='immediate'
                                )
                            ]
                        )
                    ],
                )
            ]
        )

        return self

    def scatter(self):
        scatter = self.strategy.scatter()
        self.plot.component['scatter'] = scatter

        return self

    def scene(self):
        scene = self.strategy.scene()
        self.plot.component['scene'] = scene

        return self

    def audio(self):
        if isinstance(self.strategy.trace, AnimationTrace):
            return self

        player = Output()
        player.layout.visibility = 'hidden'

        with player:
            audio = Audio(url='')
            display(audio)

        self.plot.component['audio'] = audio
        self.plot.component['player'] = player

        return self

    def description(self):
        if isinstance(self.strategy.trace, AnimationTrace):
            return self

        value = (
            self.dataframe
            .loc[
                self.dataframe.index[0],
                ['filter_bytes']
            ]
            .values[0]
        )

        description = Image(value=value)
        self.plot.component['description'] = description

        return self

    def figure(self):
        data = self.plot.component.get('scatter')
        layout = self.plot.component.get('layout')

        figure = go.Figure(
            data=data,
            layout=layout
        )

        if isinstance(self.strategy, TwoDimensional):
            figure.update_xaxes(fixedrange=True)
            figure.update_yaxes(fixedrange=True)

        self.plot.component['figure'] = figure

        return self

    def html(self):
        if isinstance(self.strategy.trace, AnimationTrace):
            return self

        mask = (self.dataframe.folder.str.len() > 20)

        self.dataframe.loc[mask, 'folder'] = (
            self.dataframe
            .loc[mask]
            .folder
            .str
            .slice(0, 20) + '...'
        )

        mask = (self.dataframe.filename.str.len() > 20)

        self.dataframe.loc[mask, 'filename'] = (
            self.dataframe
            .loc[mask]
            .filename
            .str
            .slice(0, 20) + '...'
        )

        column = [
            'folder',
            'filename',
            'sequence',
            'onset',
            'offset',
            'duration',
            'minimum',
            'maximum'
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

        html = HTML(value=value)
        self.plot.component['html'] = html

        return self

    def layout(self):
        scene = self.plot.component.get('scene')

        layout = go.Layout(
            scene=scene,
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

        self.plot.component['layout'] = layout

        return self

    def widget(self):
        figure = self.plot.component.get('figure')

        if isinstance(self.strategy.trace, AnimationTrace):
            widget = go.Figure(figure)
        else:
            widget = go.FigureWidget(figure)

        self.plot.component['widget'] = widget

        return self

    def get(self):
        widget = self.plot.component.get('widget')

        if isinstance(self.strategy.trace, AnimationTrace):
            return widget

        description = self.plot.component.get('description')
        html = self.plot.component.get('html')

        left = VBox(
            [description, html],
            layout=Layout(
                align_items='center',
                display='flex',
                flex_flow='column',
                justify_content='center'
            )
        )

        left.add_class('left')

        right = VBox(
            [widget],
            layout=Layout(
                align_items='center',
                display='flex',
                width='100%'
            )
        )

        right.add_class('right')

        player = self.plot.component.get('player')

        box = VBox(
            [
                HBox(
                    [left, right]
                ),
                player
            ],
            layout=Layout(
                align_items='center',
                flex='none',
                width='100%'
            )
        )

        box.add_class('projection')

        if isinstance(self.strategy, AnimationTrace):
            self.strategy.animate()

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
            'duration',
            'minimum',
            'maximum'
        ]

        zero = 0
        index = index[zero]
        indices = trace.customdata[zero]
        i = indices[index]

        html = self.plot.component.get('html')

        html.value = (
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
                ['filter_bytes']
            ]
            .values[zero]
        )

        description = self.plot.component.get('description')
        description.value = spectrogram[zero]

    def play(self, index):
        segment = self.dataframe.loc[
            self.dataframe.index[index],
            'segment'
        ]

        audio = self.plot.component.get('audio')
        player = self.plot.component.get('player')

        with player:
            player.clear_output(True)

            audio = Audio(
                data=segment.data,
                rate=segment.rate,
                autoplay=True
            )

            display(audio)

    def css(self):
        if isinstance(self.strategy.trace, AnimationTrace):
            return self

        css = '''
            <style>
                .left {
                    width: 30% !important;
                    margin-left: 50px;
                    margin-right: 100px;
                }

                .left > .widget-image {
                    width: 50% !important;
                }

                .right {
                    width: 90% !important;
                    overflow: hidden !important;
                }

                .description {
                    background-color: transparent !important;
                    border: 1px solid black; !important;
                    width: 300px; !important;
                }

                .description td, .description th {
                    border: none !important;
                    padding: 5px 10px !important;
                    text-align: center !important;
                }

                .js-plotly-plot .plotly .cursor-crosshair,
                .js-plotly-plot .plotly .cursor-move {
                    cursor: default !important;
                }

                .projection {
                    width: 100% !important;
                }

                .button {
                    margin-top: 50px;
                }

                .widget-button {
                    margin-left: 50px;
                    margin-right: 50px;
                }
            </style>
        '''

        html = HTML(css)
        display(html)

        return self

    def listener(self):
        if isinstance(self.strategy.trace, AnimationTrace):
            return self

        widget = self.plot.component.get('widget')

        for trace in widget.data:
            trace.on_hover(self.on_hover)
            trace.on_click(self.on_click)

        return self


class Plot():
    def __init__(self):
        self.component = {}


class Interactive:
    def __init__(self):
        self.builder = Builder()

    @Builder.dataframe.setter
    def dataframe(self, dataframe):
        self.builder.dataframe = dataframe

    @Builder.strategy.setter
    def strategy(self, strategy):
        self.builder.strategy = strategy

    def create(self):
        return (
            self.builder
            .scatter()
            .scene()
            .layout()
            .figure()
            .animation()
            .widget()
            .description()
            .html()
            .audio()
            .css()
            .listener()
            .get()
        )
