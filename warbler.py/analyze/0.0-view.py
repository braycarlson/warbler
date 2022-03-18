import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dataclass.signal import Signal
from dataclass.spectrogram import Spectrogram
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from parameters import Parameters
from path import BASELINE, DATA, SEGMENT
from pathlib import Path
from spectrogram.axes import SpectrogramAxes
from spectrogram.plot import (
    create_luscinia_spectrogram,
    create_spectrogram,
    plot_segmentation,
    plot_spectrogram
)
from vocalseg.dynamic_thresholding import dynamic_threshold_segmentation


def main():
    # spreadsheet = DATA.joinpath('printouts.xlsx')

    # dataframe = pd.read_excel(
    #     spreadsheet,
    #     sheet_name='2017mlViability',
    #     engine='openpyxl'
    # )

    # individual = dataframe.get('Individual')
    # filename = dataframe.get('updatedFileName')
    # viability = dataframe.get('mlViability')
    # reason = dataframe.get('mlViabilityReason')
    # notes = dataframe.get('mlNotes')

    # for i, f, v, r, n in zip(individual, filename, viability, reason, notes):
    #     # if pd.isna(r):
    #     #     print('YES')

    #     if v == 'P':
    #         print(f, v, r)
    #         continue

    path = Path(
        DATA.joinpath('DbWY_STE2017/wav/STE03_DbWY2017.wav')
    )

    parameters = Parameters(BASELINE)
    create_luscinia_spectrogram(path, parameters)
    # create_spectrogram(path, parameters)

    plt.show()
    plt.close()

    parameters.close()

    # path = Path(
    #     DATA.joinpath('DgLLb_STE2017/wav/STE01_2_DgLLb2017.wav')
    # )

    # # Note segmentation
    # parameters = BASELINE

    # parameters.n_fft = 4096
    # parameters.hop_length_ms = 1
    # parameters.win_length_ms = 5
    # parameters.ref_level_db = 50
    # parameters.preemphasis = 0.97
    # parameters.min_level_db = -50
    # parameters.min_level_db_floor = -20
    # parameters.db_delta = 5
    # parameters.silence_threshold = 0.01
    # parameters.lence_for_spec = 0.01
    # parameters.max_vocal_for_spec = 1.0
    # parameters.min_syllable_length_s = 0.01

    # signal = Signal(path)

    # signal.filter(
    #     parameters.butter_lowcut,
    #     parameters.butter_highcut
    # )

    # dts = dynamic_threshold_segmentation(
    #     signal.data,
    #     signal.rate,
    #     n_fft=parameters.n_fft,
    #     hop_length_ms=parameters.hop_length_ms,
    #     win_length_ms=parameters.win_length_ms,
    #     ref_level_db=parameters.ref_level_db,
    #     pre=parameters.preemphasis,
    #     min_level_db=parameters.min_level_db,
    #     silence_threshold=parameters.silence_threshold,
    #     # spectral_range=parameters.spectral_range,
    #     min_syllable_length_s=parameters.min_syllable_length_s,
    # )

    # spectrogram = dts.get('spec')
    # onsets = dts.get('onsets')
    # offsets = dts.get('offsets')

    # plt.figure(
    #     figsize=(19, 9)
    # )

    # gs = gridspec.GridSpec(2, 1)

    # spec = Spectrogram(signal, parameters)

    # gs.update(hspace=0.5)
    # ax0 = plt.subplot(gs[0], projection='spectrogram')
    # ax1 = plt.subplot(gs[1], projection='spectrogram')

    # ax0._title(path.stem + ': Filtered')
    # ax1._title(path.stem + ': Segmented')

    # ax0.set_aspect('auto')
    # ax1.set_aspect('auto')

    # plot_spectrogram(
    #     spectrogram,
    #     ax=ax0,
    #     signal=signal,
    #     cmap=plt.cm.Greys,
    # )

    # plot_spectrogram(
    #     spec.data,
    #     ax=ax1,
    #     signal=signal,
    #     cmap=plt.cm.Greys,
    # )

    # ylmin, ylmax = ax1.get_ylim()
    # ysize = (ylmax - ylmin) * 0.1
    # ymin = ylmax - ysize

    # patches = []

    # for index, (onset, offset) in enumerate(zip(onsets, offsets), 0):
    #     ax1.axvline(
    #         onset,
    #         color='dodgerblue',
    #         ls='dashed',
    #         lw=1,
    #         alpha=0.75
    #     )

    #     ax1.axvline(
    #         offset,
    #         color='dodgerblue',
    #         ls='dashed',
    #         lw=1,
    #         alpha=0.75
    #     )

    #     rectangle = Rectangle(
    #         xy=(onset, ymin),
    #         width=offset - onset,
    #         height=100,
    #     )

    #     rx, ry = rectangle.get_xy()
    #     cx = rx + rectangle.get_width() / 2.0
    #     cy = ry + rectangle.get_height() / 2.0

    #     ax1.annotate(
    #         index,
    #         (cx, cy),
    #         color='white',
    #         weight=600,
    #         fontfamily='Arial',
    #         fontsize=8,
    #         ha='center',
    #         va='center'
    #     )

    #     patches.append(rectangle)

    # collection = PatchCollection(
    #     patches,
    #     color='dodgerblue',
    #     alpha=0.75
    # )

    # ax1.add_collection(collection)

    # plt.show()
    # plt.close()


if __name__ == '__main__':
    main()
