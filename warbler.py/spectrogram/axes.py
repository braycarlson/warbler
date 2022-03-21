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
        self.format()

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

    def format(self):
        self.format_coord = (
            lambda x, y: "Time={:1.17f}, Frequency={:1.2f}".format(x, y * 10)
        )

    def _x_label(self, **kwargs):
        kwargs.setdefault('fontfamily', 'Arial')
        kwargs.setdefault('fontsize', 12)
        kwargs.setdefault('fontweight', 400)

        self.set_xlabel(
            'Time (s)',
            **kwargs
        )

    def _y_label(self, **kwargs):
        kwargs.setdefault('fontfamily', 'Arial')
        kwargs.setdefault('fontsize', 12)
        kwargs.setdefault('fontweight', 400)

        self.set_ylabel(
            'Frequency (kHz)',
            **kwargs
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
                lambda y, pos: '{0:g}'.format(y / 1e3)
            )
        )

    def _x_minor_tick(self):
        self.xaxis.set_minor_locator(
            ticker.MultipleLocator(0.1)
        )

    def _y_minor_tick(self):
        self.yaxis.set_minor_locator(
            ticker.MaxNLocator(10)
        )

    def _x_lim(self, maximum=5):
        self.set_xlim(0, maximum)

    def _y_lim(self, maximum=10000):
        self.set_ylim(0, maximum)

    def _x_step(self, maximum=5):
        self.xaxis.set_ticks(
            np.arange(0, maximum, 0.5)
        )

    def _y_step(self):
        self.yaxis.set_ticks(
            [0, 5000, 10000]
        )

    def _x_position(self):
        self.xaxis.tick_bottom()

    def _y_position(self):
        self.yaxis.tick_left()

    def _x_padding(self):
        self.xaxis.labelpad = 10

    def _y_padding(self):
        self.yaxis.labelpad = 10

    def _title(self, title='Spectrogram', **kwargs):
        kwargs.setdefault('fontsize', 14)
        kwargs.setdefault('weight', 600)
        kwargs.setdefault('pad', 15)

        super().set_title(title, **kwargs)


class LusciniaAxes(SpectrogramAxes):
    name = 'luscinia'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def initialize(self):
        super().initialize_x()
        super().initialize_y()

    def _x_label(self, **kwargs):
        kwargs.setdefault('fontfamily', 'Arial')
        kwargs.setdefault('fontsize', 16)
        kwargs.setdefault('fontweight', 600)

        self.set_xlabel(
            'Time (s)',
            **kwargs
        )

    def _y_label(self, **kwargs):
        kwargs.setdefault('fontfamily', 'Arial')
        kwargs.setdefault('fontsize', 16)
        kwargs.setdefault('fontweight', 600)

        self.set_ylabel(
            'Frequency (kHz)',
            **kwargs
        )

    def _x_padding(self):
        self.xaxis.labelpad = 20

    def _y_padding(self):
        self.yaxis.labelpad = 20


register_projection(SpectrogramAxes)
register_projection(LusciniaAxes)
