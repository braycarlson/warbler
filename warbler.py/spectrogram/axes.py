import matplotlib.ticker as ticker
import numpy as np

from matplotlib.axes import Axes
from matplotlib.projections import register_projection


class SpectrogramAxes(Axes):
    name = 'spectrogram'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self):
        self.initialize_x()
        self.initialize_y()

    def initialize_x(self):
        self._x_label()
        self._x_major_tick()
        self._x_lim()
        self._x_step()
        self._x_minor_tick()
        self._x_position()
        self._x_padding()

    def initialize_y(self):
        self._y_label()
        self._y_major_tick()
        self._y_lim()
        self._y_step()
        self._y_minor_tick()
        self._y_position()
        self._y_padding()

    def _x_label(self):
        self.set_xlabel(
            'Time (s)',
            fontfamily='Arial',
            fontsize=12,
            fontweight=400
        )

    def _y_label(self):
        self.set_ylabel(
            'Frequency (kHz)',
            fontfamily='Arial',
            fontsize=12,
            fontweight=400
        )

    def _x_major_tick(self):
        self.xaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda x, pos: '{0:,.1f}'.format(x)
            )
        )

    def _y_major_tick(self):
        self.yaxis.set_major_formatter(
            ticker.FuncFormatter(
                lambda y, pos: '{0:g}'.format(y / 1e2)
            )
        )

    def _x_minor_tick(self):
        self.xaxis.set_minor_locator(
            ticker.MaxNLocator(30)
        )

    def _y_minor_tick(self):
        self.yaxis.set_minor_locator(
            ticker.MaxNLocator(10)
        )

    def _x_lim(self, maximum=5):
        self.set_xlim(0, maximum)

    def _y_lim(self):
        self.set_ylim(0, 1000)

    def _x_step(self, maximum=5):
        self.xaxis.set_ticks(
            np.arange(0, maximum, 0.5)
        )

    def _y_step(self):
        self.yaxis.set_ticks(
            [0, 500, 1000]
        )

    def _x_position(self):
        self.xaxis.tick_bottom()

    def _y_position(self):
        self.yaxis.tick_left()

    def _x_padding(self):
        self.xaxis.labelpad = 10

    def _y_padding(self):
        self.yaxis.labelpad = 10

    def _title(self, title=None):
        super().set_title(
            title,
            fontsize=14,
            weight=600,
            pad=15,
        )


class LusciniaAxes(SpectrogramAxes):
    name = 'luscinia'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self):
        super().initialize_x()
        super().initialize_y()

    def _x_label(self):
        self.set_xlabel(
            'Time (s)',
            fontfamily='Arial',
            fontsize=16,
            fontweight=600
        )

    def _y_label(self):
        self.set_ylabel(
            'Frequency (kHz)',
            fontfamily='Arial',
            fontsize=16,
            fontweight=600
        )

    def _x_padding(self):
        self.xaxis.labelpad = 20

    def _y_padding(self):
        self.yaxis.labelpad = 20


register_projection(SpectrogramAxes)
register_projection(LusciniaAxes)
