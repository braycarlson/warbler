import numpy as np

from scipy.io import wavfile
from scipy.signal import butter, lfilter


class Signal:
    def __init__(self, path):
        self._path = path
        self._rate, self._data = self._load()

    def _load(self):
        return wavfile.read(self._path)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def rate(self):
        return self._rate

    @property
    def duration(self):
        self._duration = len(self.data) / float(self.rate)
        return self._duration

    def filter(self, lowcut, highcut, order=5):
        data = np.array(self.data / 32768).astype('float32')

        if highcut > int(self.rate / 2):
            highcut = int(self.rate / 2)

        nyquist = 0.5 * self.rate
        low = lowcut / nyquist
        high = highcut / nyquist

        numerator, denominator = butter(
            order,
            [low, high],
            btype='band'
        )

        self.data = lfilter(
            numerator,
            denominator,
            data
        )

    def save(self, path):
        wavfile.write(
            path,
            self.rate,
            self.data
        )
