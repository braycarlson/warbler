import json
import matplotlib.pyplot as plt
import pickle
import PySimpleGUI as sg

from parameters import Parameters
from path import PARAMETER, PARAMETERS, PICKLE, SEGMENT
from pathlib import Path
from spectrogram.plot import (
    # create_luscinia_spectrogram,
    create_spectrogram
)


file = PICKLE.joinpath('mediocre.pkl')

with open(file, 'rb') as handle:
    files = pickle.load(handle)

# Load baseline parameters
file = SEGMENT.joinpath('parameters.json')

with open(file, 'r') as handle:
    baseline = json.load(handle)


def load(window, parameters):
    ignore = [
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


def get_metadata(filename):
    metadata = {}

    for file in files:
        if file.get('filename') == filename:
            metadata.update(file)

    return metadata


def combobox():
    label_font = 'Arial 10 bold'
    label_size = (20, 0)

    combobox_size = (44, 1)

    return [
        sg.T('File', size=label_size, font=label_font),
        sg.Combo(
            [file.get('filename') for file in files],
            size=combobox_size,
            key='file',
            pad=((4, 0), 0),
            enable_events=True,
            readonly=True,
        )
    ]


def parameter(name, multi=False):
    label_font = 'Arial 10 bold'
    label_size = (20, 0)

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
            sg.I('', key=name, size=input_size)
        ]


def button(name):
    font = 'Arial 10'
    size = (12, 0)

    return [
        sg.B(name, key=name.lower(), size=size, font=font),
    ]


def gui():
    return [
        [sg.Frame('', border_width=0, layout=[
            [sg.Frame('', border_width=0, layout=[
                combobox(),
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
            ])],
            [sg.Frame('', border_width=0, layout=[
                button('Generate') +
                button('Save')
            ])]
        ])
        ]]


def main():
    layout = gui()

    window = sg.Window(
        'warbler.py',
        layout,
        finalize=True
    )

    load(window, baseline)

    while True:
        event, data = window.read()

        if event == sg.WIN_CLOSED or event == 'Cancel':
            break

        if event == 'file':
            file = data['file']
            filename = Path(file).stem

            # Look at the existing .json file(s) to see if
            # there is an existing parameter object for the
            # selected file. If not, use the baseline.

            if filename not in PARAMETERS:
                load(window, baseline)
                continue

            path = PARAMETER.joinpath(filename + '.json')

            with open(path, 'r') as handle:
                custom = json.load(handle)

            load(window, custom)

        if event == 'generate':
            filename = data['file']

            if filename == '':
                sg.Popup(
                    'Please select a file',
                    keep_on_top=True
                )

                continue

            metadata = get_metadata(filename)
            path = metadata.get('path')

            parameters = Parameters(
                SEGMENT.joinpath('parameters.json')
            )

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

            create_spectrogram(path, parameters)

            plt.show()

            plt.close()
            parameters.close()

        if event == 'save':
            filename = data['file']

            if filename == '':
                sg.Popup(
                    'Please select a file',
                    keep_on_top=True
                )

                continue

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
                'file',
                'spectral_range'
            ]

            convert = [
                'bandpass_filter',
                'reduce_noise',
                'mask_spec'
            ]

            for key in data:
                if key in ignore:
                    continue

                if key in convert:
                    data[key] = bool(data[key])
                    continue

                try:
                    data[key] = int(data[key])
                except ValueError:
                    data[key] = float(data[key])
                except Exception as exception:
                    print(exception)

            data.update({
                'power': 1.5,
                'griffin_lim_iters': 50,
                'noise_reduce_kwargs': {},
                'mask_spec_kwargs': {
                    'spec_thresh': 0.9,
                    'offset': 1e-10
                }
            })

            name = data.pop('file')
            filename = Path(name).stem + '.json'

            with open(PARAMETER.joinpath(filename), 'w+') as handle:
                text = json.dumps(data, indent=4)
                handle.write(text)

            # sg.Popup('Saved', keep_on_top=True)

    window.close()


if __name__ == '__main__':
    main()
