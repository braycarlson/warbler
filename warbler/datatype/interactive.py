from __future__ import annotations

import numpy as np

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
from typing_extensions import TYPE_CHECKING
from warbler.constant import SETTINGS

if TYPE_CHECKING:
    import pandas as pd

    from typing_extensions import Any, Self
    from warbler.datatype.trace import Visitor


class DimensionalStrategy(ABC):
    def __init__(
        self,
        dataframe: pd.DataFrame | None = None,
        trace: Visitor | None = None
    ):
        self.dataframe = dataframe
        self.trace = trace

    @abstractmethod
    def scatter(self) -> Any:
        raise NotImplementedError

    @abstractmethod
    def scene(self) -> go.layout.Scene:
        raise NotImplementedError


class ThreeDimensional(DimensionalStrategy):
    def __init__(
        self,
        dataframe: pd.DataFrame | None = None,
        trace: Visitor | None = None
    ):
        super().__init__(dataframe, trace)

    def scatter(self) -> Any:
        return self.trace.three(self.dataframe)

    def scene(self) -> go.layout.Scene:
        return go.layout.Scene(
            xaxis=go.layout.scene.XAxis(title='x'),
            yaxis=go.layout.scene.YAxis(title='y'),
            zaxis=go.layout.scene.ZAxis(title='z')
        )


class TwoDimensional(DimensionalStrategy):
    def __init__(
        self,
        dataframe: pd.DataFrame | None = None,
        trace: Visitor | None = None
    ):
        super().__init__(dataframe, trace)

    def scatter(self) -> Any:
        return self.trace.two(self.dataframe)

    def scene(self) -> go.layout.Scene:
        return go.layout.Scene(
            xaxis=go.layout.scene.XAxis(title='x'),
            yaxis=go.layout.scene.YAxis(title='y')
        )


class Builder:
    def __init__(
        self,
        dataframe: pd.DataFrame = None,
        strategy: Visitor = None
    ):
        self.dataframe = dataframe
        self.strategy = strategy

        self.plot = Plot()

        self.height = 800
        self.width = 1000

        self.unit = {
            'onset': 's',
            'offset': 's',
            'duration': 's',
            'minimum': 'Hz',
            'mean': 'Hz',
            'maximum': 'Hz'
        }

    def animation(self) -> Self:
        if str(self.strategy.trace) != 'animation':
            return self

        figure = self.plot.component.get('figure')

        self.strategy.trace.animate()
        figure.frames = self.strategy.trace.frames

        figure.update_layout(
            scene_camera={
                'eye': {
                    'x': self.strategy.trace.x_eye,
                    'y': self.strategy.trace.y_eye,
                    'z': self.strategy.trace.z_eye
                }
            },
            updatemenus=[
                {
                    'type': 'buttons',
                    'showactive': True,
                    'y': 1,
                    'x': 0.1,
                    'xanchor': 'left',
                    'yanchor': 'top',
                    'pad': {'t': 45, 'r': 10},
                    'buttons': [
                        {
                            'label': 'Rotate',
                            'method': 'animate',
                            'args': [
                                None,
                                {
                                    'frame': {
                                        'duration': 0.5,
                                        'redraw': True
                                    },
                                    'transition': {
                                        'duration': 0,
                                        'easing': 'linear'
                                    },
                                    'fromcurrent': True,
                                    'mode': 'immediate'
                                }
                            ]
                        }
                    ],
                }
            ]
        )

        return self

    def scatter(self) -> Self:
        scatter = self.strategy.scatter()
        self.plot.component['scatter'] = scatter

        return self

    def scene(self) -> Self:
        scene = self.strategy.scene()
        self.plot.component['scene'] = scene

        return self

    def audio(self) -> Self:
        if str(self.strategy.trace) == 'animation':
            return self

        player = Output()
        player.layout.visibility = 'hidden'

        with player:
            audio = Audio(url='')
            display(audio)

        self.plot.component['audio'] = audio
        self.plot.component['player'] = player

        return self

    def description(self) -> Self:
        if str(self.strategy.trace) == 'animation':
            return self

        value = (
            self.dataframe
            .loc[
                self.dataframe.index[0],
                ['filter_bytes']
            ]
            .tolist()[0]
        )

        description = Image(value=value)
        self.plot.component['description'] = description

        return self

    def figure(self) -> Self:
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

    def html(self) -> Self:
        if str(self.strategy.trace) == 'animation':
            return self

        for column, unit in self.unit.items():
            self.dataframe[column] = (
                self.dataframe[column].apply(
                    lambda x: f"{x:.2f}"
                    ) + unit
            )

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
            'mean',
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

    def layout(self) -> Self:
        scene = self.plot.component.get('scene')

        layout = go.Layout(
            scene=scene,
            height=self.height,
            width=self.width,
            template='plotly_white',
            showlegend=True,
            legend={
                'itemsizing': 'constant',
                'x': 0.01,
                'xanchor': 'right',
                'y': 0.99,
                'yanchor': 'top'
            }
        )

        self.plot.component['layout'] = layout

        return self

    def widget(self) -> Self:
        figure = self.plot.component.get('figure')

        if str(self.strategy.trace) == 'animation':
            widget = go.Figure(figure)
        else:
            widget = go.FigureWidget(figure)

        self.plot.component['widget'] = widget

        return self

    def get(self) -> VBox:
        widget = self.plot.component.get('widget')

        if str(self.strategy.trace) == 'animation':
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

        if str(self.strategy.trace) == 'animation':
            self.strategy.animate()

        return box

    def on_click(
        self,
        trace: go.Scatter3d,
        points: go.callbacks.Point,
        _ : None
    ) -> None:
        index = points.point_inds

        if not index:
            return

        zero = 0
        index = index[zero]
        indices = trace.customdata[zero]
        i = indices[index]

        self.play(i)

    def on_hover(
        self,
        trace: go.Scatter3d,
        points: go.callbacks.Point,
        _: None
    ) -> None:
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
            'mean',
            'maximum'
        ]

        zero = 0
        index = index[zero]
        indices = trace.customdata[zero]
        i = indices[index]

        html = self.plot.component.get('html')

        selection = np.array(
            self.dataframe.index[i],
            dtype=np.int64,
            ndmin=1
        )

        html.value = (
            self.dataframe
            .loc[selection, column]
            .transpose()
            .to_html(
                classes='description',
                index=True,
                justify='center'
            )
        )

        spectrogram = [
            self
            .dataframe.loc[selection, 'filter_bytes']
            .squeeze()
        ]

        description = self.plot.component.get('description')
        description.value = spectrogram[zero]

    def play(self, index: int) -> None:
        segment = self.dataframe.loc[
            self.dataframe.index[index],
            'segment'
        ]

        audio = self.plot.component.get('audio')
        player = self.plot.component.get('player')

        with player:
            player.clear_output(wait=True)

            audio = Audio(
                data=segment.data,
                rate=segment.rate,
                autoplay=True
            )

            display(audio)

    def css(self) -> Self:
        if str(self.strategy.trace) == 'animation':
            return self

        stylesheet = SETTINGS.joinpath('stylesheet.css')

        with open(stylesheet, 'r') as handle:
            file = handle.read()

        css = f"<style>{file}</style>"

        html = HTML(css)
        display(html)

        return self

    def listener(self) -> Self:
        if str(self.strategy.trace) == 'animation':
            return self

        widget = self.plot.component.get('widget')

        for trace in widget.data:
            trace.on_hover(self.on_hover)
            trace.on_click(self.on_click)

        return self


class Plot:
    def __init__(self):
        self.component = {}


class Interactive:
    def __init__(self, builder: Builder | None = None):
        self.builder = builder

    @property
    def dataframe(self) -> pd.DataFrame:
        return self.builder.dataframe

    @dataframe.setter
    def dataframe(self, dataframe: pd.DataFrame) -> None:
        self.builder.dataframe = dataframe

    @property
    def strategy(self) -> DimensionalStrategy:
        return self.builder.strategy

    @strategy.setter
    def strategy(self, strategy: DimensionalStrategy) -> None:
        self.builder.strategy = strategy

    def create(self) -> VBox:
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
