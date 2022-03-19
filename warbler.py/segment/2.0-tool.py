import json
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle
import PySimpleGUI as sg
import string

from dataclass.signal import Signal
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backend_bases import MouseButton
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from parameters import Parameters
from path import BASELINE, PICKLE
from spectrogram.axes import SpectrogramAxes
from spectrogram.plot import (
    plot_spectrogram
)
from vocalseg.dynamic_thresholding import dynamic_threshold_segmentation


sg.theme('LightGrey1')


EXCLUDE = set()


def draw(canvas, figure):
    figcan = FigureCanvasTkAgg(figure, canvas)
    figcan.draw()

    figcan.get_tk_widget().pack(
        side='top',
        fill='x',
        expand=0
    )

    return figcan


def on_click(event, window, patches):
    if event.inaxes is None:
        return

    if event.button is MouseButton.LEFT:
        position = event.xdata

        for patch in patches:
            start = patch.get_x()
            end = start + patch.get_width()

            if start <= position <= end:
                label = patch.get_label()
                label = int(label)

                blue = mcolors.to_rgba('#0079d3', alpha=0.75)
                red = mcolors.to_rgba('#d1193e', alpha=0.75)

                for collection in event.inaxes.collections:
                    paths = collection.get_paths()
                    colors = collection.get_facecolors()
                    length = len(paths)

                    if len(colors) == 1 and length != 1:
                        colors = np.array([colors[0]] * length)

                    index = label * 2

                    if all(colors[label] == red):
                        color = blue
                        EXCLUDE.remove(label)
                    else:
                        color = red
                        EXCLUDE.add(label)

                    colors[label] = color
                    collection.set_facecolor(colors)
                    collection.set_edgecolor(colors)

                    event.inaxes.lines[index].set_color(color)
                    event.inaxes.lines[index + 1].set_color(color)

                    notes = ', '.join(
                        [str(note) for note in sorted(EXCLUDE)]
                    )

                    window['exclude'].update(notes)
                    window.refresh()

                event.canvas.draw()


def to_digit(data):
    exclude = []

    table = str.maketrans(
        dict.fromkeys(
            string.ascii_letters + string.punctuation
        )
    )

    translation = data.translate(table)

    if translation:
        digit = [
            int(character)
            for character in translation.strip().split(' ')
        ]

        digit = sorted(
            set(digit)
        )

        exclude.extend(digit)

    return exclude


def load(window, parameters):
    ignore = [
        'exclude',
        'power',
        'griffin_lim_iters',
        'noise_reduce_kwargs',
        'mask_spec_kwargs'
    ]

    for key in parameters.keys():
        if key in ignore:
            continue

        if key == 'spectral_range':
            low, high = parameters[key]

            window['spectral_range_low'].update(low)
            window['spectral_range_high'].update(high)
        else:
            window[key].update(parameters[key])

    exclude = parameters.get('exclude')

    EXCLUDE.clear()
    EXCLUDE.update(exclude)

    if exclude:
        notes = ', '.join([str(note) for note in exclude])
        window['exclude'].update(notes)
    else:
        window['exclude'].update('')


def get_files():
    file = PICKLE.joinpath('mediocre.pkl')

    with open(file, 'rb') as handle:
        files = pickle.load(handle)

    return files


def get_filenames():
    files = get_files()
    return [file.get('filename') for file in files]


def get_index(filename):
    files = get_filenames()

    try:
        index = files.index(filename)
    except ValueError:
        return None
    else:
        return index


def forward(filename):
    filenames = get_filenames()
    length = len(filenames)

    index = get_index(filename)

    if index == length - 1:
        index = 0
    else:
        index = index + 1

    return index


def backward(filename):
    filenames = get_filenames()
    length = len(filenames)

    index = get_index(filename)

    if index == 0:
        index = length - 1
    else:
        index = index - 1

    return index


def get_metadata(filename):
    metadata = {}

    files = get_files()

    for file in files:
        if file.get('filename') == filename:
            metadata.update(file)

    return metadata


def combobox():
    label_font = 'Arial 10 bold'
    label_size = (0, 0)

    combobox_size = (46, 1)

    filenames = get_filenames()

    return [
        sg.T('File', size=label_size, font=label_font),
        sg.Combo(
            filenames,
            size=combobox_size,
            key='file',
            pad=((4, 0), 0),
            background_color='white',
            text_color='black',
            button_arrow_color='black',
            button_background_color='white',
            enable_events=True,
            readonly=True,
        )
    ]


def parameter(name, **kwargs):
    label_font = 'Arial 10 bold'
    label_size = (20, 0)

    multi = kwargs.get('multi')

    if multi:
        input_size = (22, 1)

        return [
            sg.T(name, size=label_size, font=label_font),
            sg.I('', key=name + '_' + 'low', size=input_size),
            sg.I('', key=name + '_' + 'high', size=input_size)
        ]
    else:
        input_size = (46, 1)

        return [
            sg.T(name, size=label_size, font=label_font),
            sg.I('', key=name, size=input_size, **kwargs)
        ]


def button(name, **kwargs):
    font = 'Arial 10'
    size = (18, 1)

    return [
        sg.B(
            name,
            size=size,
            font=font,
            **kwargs
        )
    ]


def gui():
    left = [
        parameter('n_fft'),
        parameter('hop_length_ms'),
        parameter('win_length_ms'),
        parameter('ref_level_db'),
        parameter('preemphasis'),
        parameter('min_level_db'),
        parameter('min_level_db_floor'),
        parameter('db_delta'),
        parameter('silence_threshold'),
        parameter('min_silence_for_spec'),
        parameter('max_vocal_for_spec'),
    ]

    right = [
        parameter('min_syllable_length_s'),
        parameter('spectral_range', multi=True),
        parameter('num_mel_bins'),
        parameter('mel_lower_edge_hertz'),
        parameter('mel_upper_edge_hertz'),
        parameter('butter_lowcut'),
        parameter('butter_highcut'),
        parameter('bandpass_filter'),
        parameter('reduce_noise'),
        parameter('mask_spec'),
        parameter('exclude')
    ]

    return [
        combobox(),
        [
            sg.Canvas(
                key='canvas',
                size=(1600, 300),
                pad=(0, (10, 30))
            )
        ],
        [
            sg.Column(
                left,
                justification='center',
                element_justification='center',
                vertical_alignment='center',
                pad=(0, (0, 30))
            ),
            sg.Column(
                right,
                justification='center',
                element_justification='center',
                vertical_alignment='center',
                pad=(0, (0, 30))
            )
        ],
        [sg.Frame('', border_width=0, pad=(None, (20, 30)), layout=[
            button('Previous', key='previous') +
            button('Generate', key='generate', button_color='#d22245') +
            button('Next', key='next')
        ])],
        [sg.Frame('', border_width=0, pad=(None, (20, 0)), layout=[
            button('Parameters', key='parameters') +
            button('Reset to Custom', key='reset_custom') +
            button('Reset to Baseline', key='reset_baseline')
        ])],
        [sg.Frame('', border_width=0, pad=(None, (20, 0)), layout=[
            button('Play', key='play') +
            button('Copy', key='copy') +
            button('Save', key='save')
        ])]
    ]


def main():
    layout = gui()

    window = sg.Window(
        'warbler.py',
        layout,
        size=(1600, 900),
        location=(100, 50),
        element_justification='center',
        keep_on_top=False,
        finalize=True
    )

    widget = None

    while True:
        event, data = window.read()

        if event == sg.WIN_CLOSED or event == 'Cancel':
            break

        if event == 'file':
            data['exclude'] = ''

            if widget is not None:
                widget.get_tk_widget().forget()
                plt.close('all')

            item = data['file']
            metadata = get_metadata(item)

            filename = metadata.get('filename')
            parameter = metadata.get('parameter')

            with open(parameter, 'r') as handle:
                file = json.load(handle)
                load(window, file)

        if event == 'generate':
            item = data['file']

            if item == '':
                sg.Popup(
                    'Please select a file',
                    keep_on_top=True
                )

                continue

            metadata = get_metadata(item)
            song = metadata.get('song')
            parameter = metadata.get('parameter')

            parameters = Parameters(parameter)

            parameters.update(
                'n_fft',
                int(data['n_fft'])
            )

            parameters.update(
                'hop_length_ms',
                int(data['hop_length_ms'])
            )

            parameters.update(
                'win_length_ms',
                int(data['win_length_ms'])
            )

            parameters.update(
                'ref_level_db',
                int(data['ref_level_db'])
            )

            parameters.update(
                'preemphasis',
                float(data['preemphasis'])
            )

            parameters.update(
                'min_level_db',
                int(data['min_level_db'])
            )

            parameters.update(
                'min_level_db_floor',
                int(data['min_level_db_floor'])
            )

            parameters.update(
                'db_delta',
                int(data['db_delta'])
            )

            parameters.update(
                'silence_threshold',
                float(data['silence_threshold'])
            )

            parameters.update(
                'min_silence_for_spec',
                float(data['min_silence_for_spec'])
            )

            parameters.update(
                'max_vocal_for_spec',
                float(data['max_vocal_for_spec'])
            )

            parameters.update(
                'min_syllable_length_s',
                float(data['min_syllable_length_s'])
            )

            parameters.update(
                'spectral_range',
                [
                    int(data['spectral_range_low']),
                    int(data['spectral_range_high'])
                ]
            )

            parameters.update(
                'num_mel_bins',
                int(data['num_mel_bins'])
            )

            parameters.update(
                'mel_lower_edge_hertz',
                int(data['mel_lower_edge_hertz'])
            )

            parameters.update(
                'mel_upper_edge_hertz',
                int(data['mel_upper_edge_hertz'])
            )

            parameters.update(
                'butter_lowcut',
                int(data['butter_lowcut'])
            )

            parameters.update(
                'butter_highcut',
                int(data['butter_highcut'])
            )

            parameters.update(
                'bandpass_filter',
                bool(data['bandpass_filter'])
            )

            parameters.update(
                'reduce_noise',
                bool(data['reduce_noise'])
            )

            parameters.update(
                'mask_spec',
                bool(data['mask_spec'])
            )

            signal = Signal(song)

            signal.filter(
                parameters.butter_lowcut,
                parameters.butter_highcut
            )

            dts = dynamic_threshold_segmentation(
                signal.data,
                signal.rate,
                n_fft=parameters.n_fft,
                hop_length_ms=parameters.hop_length_ms,
                win_length_ms=parameters.win_length_ms,
                ref_level_db=parameters.ref_level_db,
                pre=parameters.preemphasis,
                min_level_db=parameters.min_level_db,
                silence_threshold=parameters.silence_threshold,
                min_syllable_length_s=parameters.min_syllable_length_s,
            )

            try:
                spectrogram = dts.get('spec')
                onsets = dts.get('onsets')
                offsets = dts.get('offsets')
            except AttributeError:
                sg.Popup(
                    'Please adjust the parameter(s)',
                    title='Error',
                    keep_on_top=True
                )

                return None

            fig, ax = plt.subplots(
                figsize=(18, 3),
                subplot_kw={'projection': 'spectrogram'}
            )

            fig.patch.set_facecolor('#ffffff')
            ax.patch.set_facecolor('#ffffff')

            plot_spectrogram(
                spectrogram,
                ax=ax,
                signal=signal,
                cmap=plt.cm.Greys,
            )

            ylmin, ylmax = ax.get_ylim()
            ysize = (ylmax - ylmin) * 0.1
            ymin = ylmax - ysize

            patches = []

            blue = mcolors.to_rgba('#0079d3', alpha=0.75)
            red = mcolors.to_rgba('#d1193e', alpha=0.75)

            exclude = to_digit(data['exclude'])

            for index, (onset, offset) in enumerate(zip(onsets, offsets), 0):
                if index in exclude:
                    color = red
                else:
                    color = blue

                ax.axvline(
                    onset,
                    color=color,
                    ls='dashed',
                    lw=1,
                    alpha=0.75
                )

                ax.axvline(
                    offset,
                    color=color,
                    ls='dashed',
                    lw=1,
                    alpha=0.75
                )

                rectangle = Rectangle(
                    xy=(onset, ymin),
                    width=offset - onset,
                    height=100,
                    alpha=0.75,
                    color=color,
                    label=str(index)
                )

                rx, ry = rectangle.get_xy()
                cx = rx + rectangle.get_width() / 2.0
                cy = ry + rectangle.get_height() / 2.0

                ax.annotate(
                    index,
                    (cx, cy),
                    color='white',
                    weight=600,
                    fontfamily='Arial',
                    fontsize=8,
                    ha='center',
                    va='center'
                )

                patches.append(rectangle)

            collection = PatchCollection(
                patches,
                match_original=True
            )

            ax.add_collection(collection)

            plt.tight_layout()

            if widget is not None:
                widget.get_tk_widget().forget()
                plt.close('all')

            widget = draw(window['canvas'].TKCanvas, fig)

            fig.canvas.mpl_connect(
                'button_press_event',
                lambda event: on_click(
                    event,
                    window,
                    patches
                )
            )

        if event == 'reset_custom':
            item = data['file']

            if item == '':
                sg.Popup(
                    'Please select a file',
                    keep_on_top=True
                )

                continue

            data['exclude'] = ''
            metadata = get_metadata(item)

            filename = metadata.get('filename')
            parameter = metadata.get('parameter')

            with open(parameter, 'r') as handle:
                file = json.load(handle)
                load(window, file)

        if event == 'reset_baseline':
            item = data['file']

            if item == '':
                sg.Popup(
                    'Please select a file',
                    keep_on_top=True
                )

                continue

            data['exclude'] = ''

            with open(BASELINE, 'r') as handle:
                file = json.load(handle)
                load(window, file)

        if event == 'parameters':
            item = data['file']

            if item == '':
                sg.Popup(
                    'Please select a file',
                    keep_on_top=True
                )

                continue

            metadata = get_metadata(item)

            filename = metadata.get('filename')
            parameter = metadata.get('parameter')

            os.startfile(parameter)

        if event == 'next':
            item = data['file']

            if item == '':
                index = 0
            else:
                index = forward(item)

            window['file'].update(
                set_to_index=index,
            )

            filenames = get_filenames()

            item = filenames[index]
            metadata = get_metadata(item)

            filename = metadata.get('filename')
            parameter = metadata.get('parameter')

            with open(parameter, 'r') as handle:
                file = json.load(handle)
                load(window, file)

        if event == 'previous':
            item = data['file']

            if item == '':
                index = 0
            else:
                index = backward(item)

            window['file'].update(
                set_to_index=index,
            )

            filenames = get_filenames()

            item = filenames[index]
            metadata = get_metadata(item)

            filename = metadata.get('filename')
            parameter = metadata.get('parameter')

            with open(parameter, 'r') as handle:
                file = json.load(handle)
                load(window, file)

        if event == 'play':
            item = data['file']

            if item == '':
                sg.Popup(
                    'Please select a file',
                    keep_on_top=True
                )

                continue

            metadata = get_metadata(item)

            filename = metadata.get('filename')
            song = metadata.get('song')

            os.startfile(song)

        if event == 'copy':
            item = data['file']

            if item == '':
                sg.Popup(
                    'Please select a file',
                    keep_on_top=True
                )

                continue

            df = pd.DataFrame([item])
            df.to_clipboard(index=False, header=False)

        if event == 'save':
            item = data['file']

            if item == '':
                sg.Popup(
                    'Please select a file',
                    keep_on_top=True
                )

                continue

            metadata = get_metadata(item)
            parameter = metadata.get('parameter')

            low = data.get('spectral_range_low')
            high = data.pop('spectral_range_high')
            spectral_range = [int(low), int(high)]

            data = {
                'spectral_range'
                if key == 'spectral_range_low' else key: value
                for key, value in data.items()
            }

            data['spectral_range'] = spectral_range

            ignore = [
                'canvas',
                'exclude',
                'file',
                'spectral_range'
            ]

            to_bool = [
                'bandpass_filter',
                'reduce_noise',
                'mask_spec'
            ]

            exclude = to_digit(data['exclude'])

            for key in data:
                if key in ignore:
                    continue

                if key in to_bool:
                    data[key] = bool(data[key])
                    continue

                try:
                    data[key] = int(data[key])
                except ValueError:
                    data[key] = float(data[key])
                except Exception as exception:
                    print(exception)

            data.update({
                'exclude': exclude,
                'power': 1.5,
                'griffin_lim_iters': 50,
                'noise_reduce_kwargs': {},
                'mask_spec_kwargs': {
                    'spec_thresh': 0.9,
                    'offset': 1e-10
                }
            })

            data.pop('file')
            data.pop('canvas')

            with open(parameter, 'w+') as handle:
                text = json.dumps(data, indent=4)
                handle.write(text)

    window.close()


if __name__ == '__main__':
    main()
