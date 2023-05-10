"""
Interactive
-----------

"""

from __future__ import annotations

import numpy as np

from abc import ABC, abstractmethod
from datatype.trace import AnimationTrace, Visitor
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
from typing import Any, Self, TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd


class DimensionalStrategy(ABC):
    """Abstract base class for dimensional strategies.

    Args:
        dataframe: The input pandas DataFrame.
        trace: The visitor object for the strategy.

    Attributes:
        _dataframe: The input pandas DataFrame.
        _trace: The visitor object for the strategy.

    """

    def __init__(
        self,
        dataframe: pd.DataFrame | None = None,
        trace: Visitor | None = None
    ):
        self._dataframe = dataframe
        self._trace = trace

    @property
    def dataframe(self) -> pd.DataFrame:
        """Get the input pandas DataFrame."""

        return self._dataframe

    @dataframe.setter
    def dataframe(self, dataframe: pd.DataFrame) -> None:
        self._dataframe = dataframe

    @property
    def trace(self) -> Any:
        """Get the visitor object for the strategy."""

        return self._trace

    @trace.setter
    def trace(self, trace: Visitor) -> None:
        self._trace = trace

    @abstractmethod
    def scatter(self) -> Any:
        """Abstract method for generating a scatter plot."""

        raise NotImplementedError

    @abstractmethod
    def scene(self) -> go.layout.Scene:
        """Abstract method for generating a scene layout.

        Args:
            data: The scene layout data.

        """

        raise NotImplementedError


class ThreeDimensional(DimensionalStrategy):
    def scatter(self) -> Any:
        """Generate a scatter plot for a three-dimensional visualization.

        Returns:
            The visitor object with scatter data.

        """

        return self.trace.three(self.dataframe)

    def scene(self) -> go.layout.Scene:
        """Generate a scene layout for a three-dimensional visualization.

        Returns:
            The scene layout.

        """

        return go.layout.Scene(
            xaxis=go.layout.scene.XAxis(title='x'),
            yaxis=go.layout.scene.YAxis(title='y'),
            zaxis=go.layout.scene.ZAxis(title='z')
        )


class TwoDimensional(DimensionalStrategy):
    def scatter(self) -> Any:
        """Generate a scatter plot for a two-dimensional visualization.

        Returns:
            The visitor object with scatter data.

        """

        return self.trace.two(self.dataframe)

    def scene(self) -> go.layout.Scene:
        """Generate a scene layout for a two-dimensional visualization.

        Returns:
            The scene layout.

        """

        return go.layout.Scene(
            xaxis=go.layout.scene.XAxis(title='x'),
            yaxis=go.layout.scene.YAxis(title='y')
        )


class Builder:
    """
    The builder class for creating an interactive plot.

    Args:
        dataframe: The input pandas DataFrame.
        strategy: The dimensional strategy used for visualization.

    Attributes:
        _dataframe: The input pandas DataFrame.
        _strategy: The dimensional strategy used for visualization.
        plot: The Plot object for storing visualization components.
        height: The height of the visualization.
        width: The width of the visualization.

    """

    def __init__(self, dataframe: pd.DataFrame = None, strategy: Visitor = None):
        self._dataframe = dataframe
        self._strategy = strategy

        self.plot = Plot()

        self.height = 800
        self.width = 1000

    @property
    def dataframe(self) -> pd.DataFrame:
        """Get the input pandas DataFrame."""

        return self._dataframe

    @dataframe.setter
    def dataframe(self, dataframe: pd.DataFrame) -> None:
        """Set the input pandas DataFrame."""

        self._dataframe = dataframe

    @property
    def strategy(self) -> DimensionalStrategy:
        """Get the dimensional strategy used for visualization."""

        return self._strategy

    @strategy.setter
    def strategy(self, strategy: DimensionalStrategy) -> None:
        self._strategy = strategy

    def animation(self) -> Self:
        """Enable animation for the visualization.

        Returns:
            The modified Builder instance.

        """

        if not isinstance(self.strategy.trace, AnimationTrace):
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
        """Generate scatter data for the visualization.

        Returns:
            The modified Builder instance.

        """

        scatter = self.strategy.scatter()
        self.plot.component['scatter'] = scatter

        return self

    def scene(self) -> Self:
        """Generate a scene layout for the visualization.

        Returns:
            The modified Builder instance.

        """

        scene = self.strategy.scene()
        self.plot.component['scene'] = scene

        return self

    def audio(self) -> Self:
        """Enable audio component for the visualization.

        Returns:
            The modified Builder instance.

        """

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

    def description(self) -> Self:
        """Enable description component for the visualization.

        Returns:
            The modified Builder instance.

        """

        if isinstance(self.strategy.trace, AnimationTrace):
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
        """Generate the figure for the visualization.

        Returns:
            The modified Builder instance.

        """

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
        """Enable the HTML component for the visualization.

        Returns:
            The modified Builder instance.

        """

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

    def layout(self) -> Self:
        """Set the layout for the visualization.

        Returns:
            The modified Builder instance.

        """

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
        """Generate the widget for the visualization.

        Returns:
            The modified Builder instance.

        """

        figure = self.plot.component.get('figure')

        if isinstance(self.strategy.trace, AnimationTrace):
            widget = go.Figure(figure)
        else:
            widget = go.FigureWidget(figure)

        self.plot.component['widget'] = widget

        return self

    def get(self) -> VBox:
        """Get the final visualization as a VBox container.

        Returns:
            The VBox container containing the visualization.

        """

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

    def on_click(
        self,
        trace: go.Scatter3d,
        points: go.callbacks.Point,
        _ : None
    ) -> None:
        """Event handler for click events on the visualization.

        Args:
            trace: The 3D scatter trace.
            points: The clicked points.
            _ : Unused argument.

        """

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
        """Event handler for hover events on the visualization.

        Args:
            trace: The 3D scatter trace.
            points: The hovered points.
            _ : Unused argument.

        """

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
        """Play the audio segment corresponding to the index.

        Args:
            index: The index of the audio segment.

        """

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
        """Apply CSS styling to the visualization.

        Returns:
            The modified Builder instance.

        """

        if isinstance(self.strategy.trace, AnimationTrace):
            return self

        css = """
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
        """

        html = HTML(css)
        display(html)

        return self

    def listener(self) -> Self:
        """Attach event listeners to the visualization.

        Returns:
            The modified Builder instance.

        """

        if isinstance(self.strategy.trace, AnimationTrace):
            return self

        widget = self.plot.component.get('widget')

        for trace in widget.data:
            trace.on_hover(self.on_hover)
            trace.on_click(self.on_click)

        return self


class Plot():
    """A class representing a plot.

    Attributes:
        component: A dictionary containing plot components.

    """

    def __init__(self):
        self.component = {}


class Interactive:
    """A class for creating interactive visualizations.

    Attributes:
        builder: An instance of the Builder class.

    """

    def __init__(self):
        self.builder = Builder()

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
        """Create an interactive visualization.

        Returns:
            A VBox object containing the interactive visualization.

        """

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
