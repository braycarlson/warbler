"""
Axes
----

"""

from __future__ import annotations

import numpy as np

from matplotlib.axes import Axes
from matplotlib.projections import register_projection
from matplotlib import ticker
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Any


class SpectrogramAxes(Axes):
    """Axes for creating a spectrogram plot.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    """

    name = 'spectrogram'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize_x(self) -> None:
        """Initialize the x-axis of the spectrogram plot."""

        self._x_label()
        self._x_lim()
        self._x_position()
        self._x_step()
        self._x_ticks()

    def initialize_y(self) -> None:
        """Initialize the y-axis of the spectrogram plot."""

        self._y_label()
        self._y_position()
        self._y_padding()
        self._y_ticks()

    def initialize(self) -> None:
        """Initialize the spectrogram plot."""

        self.initialize_x()
        self.initialize_y()
        self.format()

    def _x_label(self, **kwargs) -> None:
        """Set the label for the x-axis.

        Args:
            **kwargs: Arbitrary keyword arguments.

        """

        kwargs.setdefault('fontfamily', 'Arial')
        kwargs.setdefault('fontsize', 12)
        kwargs.setdefault('fontweight', 400)

        self.set_xlabel(
            'Time (s)',
            **kwargs
        )

    def _y_label(self, **kwargs) -> None:
        """Set the label for the y-axis.

        Args:
            **kwargs: Arbitrary keyword arguments.

        """

        kwargs.setdefault('fontfamily', 'Arial')
        kwargs.setdefault('fontsize', 12)
        kwargs.setdefault('fontweight', 400)

        self.set_ylabel(
            'Frequency (kHz)',
            **kwargs
        )

    def _x_lim(self, maximum: int = 5) -> None:
        """Set the limits for the x-axis.

        Args:
            maximum: The maximum value for the x-axis.

        """

        self.set_xlim(0, maximum)

    def _x_step(self, maximum: int = 5) -> None:
        """Set the step size for the x-axis ticks.

        Args:
            maximum: The maximum value for the x-axis.

        """

        self.xaxis.set_ticks(
            np.arange(0, maximum, 0.5)
        )

    def _x_position(self) -> None:
        """Set the position of the x-axis ticks."""

        self.xaxis.tick_bottom()

    def _y_position(self) -> None:
        """Set the position of the y-axis ticks."""

        self.yaxis.tick_left()

    def _x_padding(self) -> None:
        """Set the padding for the x-axis label."""

        self.xaxis.labelpad = 10

    def _y_padding(self) -> None:
        """Set the padding for the y-axis label."""

        self.yaxis.labelpad = 10

    def _x_ticks(self) -> None:
        """Set the tick labels for the x-axis."""

        self.set_xticklabels(
            self.get_xticks(),
            fontfamily='Arial',
            fontsize=12,
            fontweight=400
        )

    def _y_ticks(self) -> None:
        """Set the tick labels for the y-axis."""

        self.set_yticklabels(
            self.get_yticks(),
            fontfamily='Arial',
            fontsize=12,
            fontweight=400
        )

    def _title(self, title: str = 'Spectrogram', **kwargs) -> None:
        """Set the title of the plot.

        Args:
            title: The title of the plot.
            **kwargs: Arbitrary keyword arguments.

        """

        kwargs.setdefault('fontsize', 14)
        kwargs.setdefault('weight', 600)
        kwargs.setdefault('pad', 15)

        super().set_title(title, **kwargs)

    def format(self) -> None:
        """Format the spectrogram plot."""

        self.format_coord = (
            lambda x, y: f"Time={x:1.17f}, Frequency={y:1.2f}"
        )


class LinearAxes(SpectrogramAxes):
    """Axes for creating a linear spectrogram.

    Args:
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    """

    name = 'linear'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _as_mpl_axes(self) -> tuple[type[LinearAxes], dict[str, Any]]:
        """Convert LinearAxes object to matplotlib axes.

        Returns:
            A tuple containing LinearAxes and additional keyword arguments.

        """

        return LinearAxes, {}

    def initialize_x(self) -> None:
        """Initialize the x-axis of the spectrogram plot."""

        self._x_label()
        self._x_major_tick()
        self._x_lim()
        self._x_step()
        self._x_minor_tick()
        self._x_position()
        self._x_padding()

    def initialize_y(self) -> None:
        """Initialize the y-axis of the spectrogram plot."""

        self._y_scale()
        self._y_label()
        self._y_major_tick()
        self._y_lim()
        self._y_step()
        self._y_minor_tick()
        self._y_position()
        self._y_padding()

    def initialize(self) -> None:
        """Initialize the spectrogram plot."""

        super().initialize()
        self.initialize_x()
        self.initialize_y()

    def _x_major_tick(self) -> None:
        """Set the major ticks for the x-axis."""

        self.xaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda x, _: f"{x:,.1f}"
            )
        )

    def _y_major_tick(self) -> None:
        """Set the major ticks for the y-axis."""

        self.yaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda y, _: '{:g}'.format(
                    int(y / 1e3)
                )
            )
        )

    def _x_minor_tick(self) -> None:
        """Set the minor ticks for the x-axis."""

        self.xaxis.set_minor_locator(
            ticker.MultipleLocator(0.1)
        )

    def _y_minor_tick(self) -> None:
        """Set the minor ticks for the y-axis."""

        self.minorticks_off()

    def _y_lim(self, maximum: int = 22050) -> None:
        """Set the y-axis limits.

        Args:
            maximum: Maximum value for the y-axis.

        """

        self.set_ylim(0, maximum / 2)

    def _y_scale(self, scale: str = 'linear') -> None:
        """Set the scale of the y-axis.

        Args:
            scale: Scale of the y-axis.

        """

        self.set_yscale(scale)

    def _y_step(self) -> None:
        """Set the step size for the y-axis."""

        ticks = np.arange(
            start=0,
            stop=22050,
            step=5000
        )

        self.yaxis.set_ticks(ticks)


class LogarithmicAxes(SpectrogramAxes):
    """Axes for creating a logarithmic spectrogram."""

    name = 'logarithmic'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _as_mpl_axes(self) -> tuple[type[LogarithmicAxes], dict[str, Any]]:
        """Convert LogarithmicAxes object to matplotlib axes.

        Returns:
            A tuple containing LogarithmicAxes and additional
                keyword arguments.

        """

        return LogarithmicAxes, {}

    def initialize_x(self) -> None:
        """Initialize the x-axis of the LogarithmicAxes object."""

        self._x_label()
        self._x_major_tick()
        self._x_lim()
        self._x_step()
        self._x_minor_tick()
        self._x_position()
        self._x_padding()

    def initialize_y(self) -> None:
        """Initialize the y-axis of the LogarithmicAxes object."""

        self._y_scale()
        self._y_label()
        self._y_major_tick()
        self._y_lim()
        self._y_step()
        self._y_minor_tick()
        self._y_position()
        self._y_padding()

    def initialize(self) -> None:
        """Initialize the LogarithmicAxes object."""

        super().initialize()
        self.initialize_x()
        self.initialize_y()

    def _x_major_tick(self) -> None:
        """Set the major ticks for the x-axis."""

        self.xaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda x, _: f"{x:,.1f}"
            )
        )

    def _y_major_tick(self) -> None:
        """Set the major ticks for the y-axis."""

        self.yaxis.set_major_formatter(
            ticker.ScalarFormatter()
        )

    def _x_minor_tick(self) -> None:
        """Set the minor ticks for the x-axis."""

        self.xaxis.set_minor_locator(
            ticker.MultipleLocator(0.1)
        )

    def _y_minor_tick(self) -> None:
        """Disable the minor ticks for the y-axis."""

        self.minorticks_off()

    def _y_lim(self, maximum: int = 22050) -> None:
        """Set the y-axis limits.

        Args:
            maximum: Maximum value for the y-axis.

        """

        self.set_ylim(20, maximum)

    def _y_scale(self, scale: str = 'log') -> None:
        """Set the scale of the y-axis.

        Args:
            scale: Scale of the y-axis.

        """

        self.set_yscale(scale)

    def _y_step(self) -> None:
        """Set the step size for the y-axis."""

        ticks = self.get_yticks()
        self.yaxis.set_ticks(ticks)


class LusciniaAxes(LinearAxes):
    """Axes for creating a Luscinia-inspired spectrogram."""

    name = 'luscinia'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _as_mpl_axes(self) -> tuple[type[LusciniaAxes], dict[str, Any]]:
        """Convert LusciniaAxes object to matplotlib axes.

        Returns:
            A tuple containing LusciniaAxes and additional keyword arguments.

        """

        return LusciniaAxes, {}

    def initialize(self) -> None:
        """Initialize the LusciniaAxes object."""

        super().initialize()
        self._x_label()
        self._x_padding()
        self._x_ticks()
        self._y_label()
        self._y_padding()
        self._y_ticks()

    def _x_label(self, **kwargs) -> None:
        """Set the label for the x-axis.

        Args:
            **kwargs: Additional keyword arguments for the label.

        """

        kwargs.setdefault('fontfamily', 'Arial')
        kwargs.setdefault('fontsize', 16)
        kwargs.setdefault('fontweight', 600)

        self.set_xlabel(
            'Time (s)',
            **kwargs
        )

    def _y_label(self, **kwargs) -> None:
        """Set the label for the y-axis.

        Args:
            **kwargs: Additional keyword arguments for the label.

        """

        kwargs.setdefault('fontfamily', 'Arial')
        kwargs.setdefault('fontsize', 16)
        kwargs.setdefault('fontweight', 600)

        self.set_ylabel(
            'Frequency (kHz)',
            **kwargs
        )

    def _x_padding(self) -> None:
        """Set the padding for the x-axis label."""

        self.xaxis.labelpad = 20

    def _y_padding(self) -> None:
        """Set the padding for the y-axis label."""

        self.yaxis.labelpad = 20

    def _x_ticks(self) -> None:
        """Set the tick labels for the x-axis."""

        self.set_xticklabels(
            self.get_xticks(),
            fontfamily='Arial',
            fontsize=14,
            fontweight=600
        )

    def _y_ticks(self) -> None:
        """Set the tick labels for the y-axis."""

        self.set_yticklabels(
            self.get_yticks(),
            fontfamily='Arial',
            fontsize=14,
            fontweight=600
        )


register_projection(LinearAxes)
register_projection(LogarithmicAxes)
register_projection(LusciniaAxes)
