from __future__ import annotations

import numpy as np

from matplotlib.axes import Axes
from matplotlib.projections import register_projection
from matplotlib import ticker
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from typing_extensions import Any


class SpectrogramAxes(Axes):
    name = 'spectrogram'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize_x(self) -> None:
        self._x_label()
        self._x_lim()
        self._x_position()
        self._x_step()
        self._x_ticks()

    def initialize_y(self) -> None:
        self._y_label()
        self._y_position()
        self._y_padding()
        self._y_ticks()

    def initialize(self) -> None:
        self.initialize_x()
        self.initialize_y()
        self.format()

    def _x_label(self, **kwargs) -> None:
        kwargs.setdefault('fontfamily', 'Arial')
        kwargs.setdefault('fontsize', 12)
        kwargs.setdefault('fontweight', 400)

        self.set_xlabel(
            'Time (s)',
            **kwargs
        )

    def _y_label(self, **kwargs) -> None:
        kwargs.setdefault('fontfamily', 'Arial')
        kwargs.setdefault('fontsize', 12)
        kwargs.setdefault('fontweight', 400)

        self.set_ylabel(
            'Frequency (kHz)',
            **kwargs
        )

    def _x_lim(self, maximum: int = 5) -> None:
        self.set_xlim(0, maximum)

    def _x_step(self, maximum: int = 5) -> None:
        self.xaxis.set_ticks(
            np.arange(0, maximum, 0.5)
        )

    def _x_position(self) -> None:
        self.xaxis.tick_bottom()

    def _y_position(self) -> None:
        self.yaxis.tick_left()

    def _x_padding(self) -> None:
        self.xaxis.labelpad = 10

    def _y_padding(self) -> None:
        self.yaxis.labelpad = 10

    def _x_ticks(self) -> None:
        self.set_xticklabels(
            self.get_xticks(),
            fontfamily='Arial',
            fontsize=12,
            fontweight=400
        )

    def _y_ticks(self) -> None:
        self.set_yticklabels(
            self.get_yticks(),
            fontfamily='Arial',
            fontsize=12,
            fontweight=400
        )

    def _title(self, title: str = 'Spectrogram', **kwargs) -> None:
        kwargs.setdefault('fontsize', 14)
        kwargs.setdefault('weight', 600)
        kwargs.setdefault('pad', 15)

        super().set_title(title, **kwargs)

    def format(self) -> None:
        self.format_coord = (
            lambda x, y: f"Time={x:1.17f}, Frequency={y:1.2f}"
        )


class LinearAxes(SpectrogramAxes):
    name = 'linear'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _as_mpl_axes(self) -> tuple[type[LinearAxes], dict[str, Any]]:
        return LinearAxes, {}

    def initialize_x(self) -> None:
        self._x_label()
        self._x_major_tick()
        self._x_lim()
        self._x_step()
        self._x_minor_tick()
        self._x_position()
        self._x_padding()

    def initialize_y(self) -> None:
        self._y_scale()
        self._y_label()
        self._y_major_tick()
        self._y_lim()
        self._y_step()
        self._y_minor_tick()
        self._y_position()
        self._y_padding()

    def initialize(self) -> None:
        super().initialize()
        self.initialize_x()
        self.initialize_y()

    def _x_major_tick(self) -> None:
        self.xaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda x, _: f"{x:,.1f}"
            )
        )

    def _y_major_tick(self) -> None:
        self.yaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda y, _: '{:g}'.format(
                    int(y / 1e3)
                )
            )
        )

    def _x_minor_tick(self) -> None:
        self.xaxis.set_minor_locator(
            ticker.MultipleLocator(0.1)
        )

    def _y_minor_tick(self) -> None:
        self.minorticks_off()

    def _y_lim(self, maximum: int = 22050) -> None:
        self.set_ylim(0, maximum / 2)

    def _y_scale(self, scale: str = 'linear') -> None:
        self.set_yscale(scale)

    def _y_step(self) -> None:
        ticks = np.arange(
            start=0,
            stop=22050,
            step=5000
        )

        self.yaxis.set_ticks(ticks)


class LogarithmicAxes(SpectrogramAxes):
    name = 'logarithmic'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _as_mpl_axes(self) -> tuple[type[LogarithmicAxes], dict[str, Any]]:
        return LogarithmicAxes, {}

    def initialize_x(self) -> None:
        self._x_label()
        self._x_major_tick()
        self._x_lim()
        self._x_step()
        self._x_minor_tick()
        self._x_position()
        self._x_padding()

    def initialize_y(self) -> None:
        self._y_scale()
        self._y_label()
        self._y_major_tick()
        self._y_lim()
        self._y_step()
        self._y_minor_tick()
        self._y_position()
        self._y_padding()

    def initialize(self) -> None:
        super().initialize()
        self.initialize_x()
        self.initialize_y()

    def _x_major_tick(self) -> None:
        self.xaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda x, _: f"{x:,.1f}"
            )
        )

    def _y_major_tick(self) -> None:
        self.yaxis.set_major_formatter(
            ticker.ScalarFormatter()
        )

    def _x_minor_tick(self) -> None:
        self.xaxis.set_minor_locator(
            ticker.MultipleLocator(0.1)
        )

    def _y_minor_tick(self) -> None:
        self.minorticks_off()

    def _y_lim(self, maximum: int = 22050) -> None:
        self.set_ylim(20, maximum)

    def _y_scale(self, scale: str = 'log') -> None:
        self.set_yscale(scale)

    def _y_step(self) -> None:
        ticks = self.get_yticks()
        self.yaxis.set_ticks(ticks)


class LusciniaAxes(LinearAxes):
    name = 'luscinia'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _as_mpl_axes(self) -> tuple[type[LusciniaAxes], dict[str, Any]]:
        return LusciniaAxes, {}

    def initialize(self) -> None:
        super().initialize()
        self._x_label()
        self._x_padding()
        self._x_ticks()
        self._y_label()
        self._y_padding()
        self._y_ticks()

    def _x_label(self, **kwargs) -> None:
        kwargs.setdefault('fontfamily', 'Arial')
        kwargs.setdefault('fontsize', 16)
        kwargs.setdefault('fontweight', 600)

        self.set_xlabel(
            'Time (s)',
            **kwargs
        )

    def _y_label(self, **kwargs) -> None:
        kwargs.setdefault('fontfamily', 'Arial')
        kwargs.setdefault('fontsize', 16)
        kwargs.setdefault('fontweight', 600)

        self.set_ylabel(
            'Frequency (kHz)',
            **kwargs
        )

    def _x_padding(self) -> None:
        self.xaxis.labelpad = 20

    def _y_padding(self) -> None:
        self.yaxis.labelpad = 20

    def _x_ticks(self) -> None:
        self.set_xticklabels(
            self.get_xticks(),
            fontfamily='Arial',
            fontsize=14,
            fontweight=600
        )

    def _y_ticks(self) -> None:
        self.set_yticklabels(
            self.get_yticks(),
            fontfamily='Arial',
            fontsize=14,
            fontweight=600
        )


register_projection(LinearAxes)
register_projection(LogarithmicAxes)
register_projection(LusciniaAxes)
