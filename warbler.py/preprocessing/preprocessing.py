import librosa
import logging
import matplotlib.pyplot as plt
import numpy as np

from bootstrap import bootstrap
from constant import SETTINGS
from datatype.dataset import Dataset
from datatype.settings import Settings
from datatype.spectrogram import Linear, Mel, Spectrogram
from logger import logger
from PIL import Image, ImageEnhance, ImageFilter


log = logging.getLogger(__name__)


def zscore(spectrogram):
    mn = np.mean(spectrogram)
    std = np.std(spectrogram)

    for i in range(spectrogram.shape[0], 0):
        for j in range(spectrogram.shape[1], 0):
            spectrogram[i, j] = (spectrogram[i, j] -  mn) / std

    return spectrogram


def create_spectrogram(signal, settings):
    spectrogram = Spectrogram()
    strategy = Linear(signal, settings)
    spectrogram.strategy = strategy

    return spectrogram.generate()


def lrosa(signal, settings):
    fft_win = 0.03

    n_fft = int(fft_win * signal.rate)
    fft_hop = fft_win / 8
    hop_length = int(fft_hop * signal.rate)

    spectrogram = librosa.feature.melspectrogram(
        y=signal.data,
        sr=signal.rate,
        n_mels=settings.num_mel_bins,
        fmax=signal.rate // 2,
        fmin=settings.mel_lower_edge_hertz,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft
    )

    return librosa.power_to_db(spectrogram, ref=np.max)


@bootstrap
def main():
    dataset = Dataset('recording')
    dataframe = dataset.load()

    # sample = dataframe.sample(1)

    signal = dataframe.at[1, 'signal']

    path = SETTINGS.joinpath('spectrogram.json')
    settings = Settings.from_file(path)

    signal.filter(
        settings.butter_lowcut,
        settings.butter_highcut
    )

    signal.normalize()

    path = SETTINGS.joinpath('dereverberate.json')
    dereverberate = Settings.from_file(path)

    signal.dereverberate(dereverberate)

    spectrogram = create_spectrogram(signal, settings)



    # mask = (spectrogram > 127)
    # spectrogram[np.where(mask)] = 255

    # spectrogram = lrosa(signal, settings)
    # print(spectrogram)
    # plt.imshow(spectrogram, cmap='Greys', origin='lower')
    # plt.show()

    image = Image.fromarray(~spectrogram)
    image.show()

    # brightness = ImageEnhance.Brightness(image)
    # image = brightness.enhance(1.2)

    # contrast = ImageEnhance.Contrast(image)
    # image = contrast.enhance(1.1)

    # blur = ImageFilter.SMOOTH
    # image = image.filter(blur)

    # image = np.array(image)

    # image[
    #     np.where(image > [100])
    # ] = [255]

    # median = ImageFilter.MinFilter(size=3)
    # image = image.filter(median)

    # image.save('test2.png', format='png')
    # image.show()

    # fig, ax = plt.subplots(
    #     constrained_layout=True,
    #     figsize=(20, 4)
    # )

    # ax.imshow(
    #     image,
    #     aspect='auto',
    #     cmap=plt.cm.Greys,
    #     interpolation=None,
    #     origin='lower',
    #     vmin=0,
    #     vmax=255
    # )

    # plt.show()
    # plt.close()


if __name__ == '__main__':
    with logger():
        main()
