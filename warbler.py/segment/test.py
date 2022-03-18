import PySimpleGUI as sg

sg.theme('BlueMono')    # Keep things interesting for your users


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

layout = [
    [
        sg.Canvas(
            key='canvas',
            size=(1800, 300),
            pad=(0, (10, 20))
        )
    ],
    [
        sg.Column(
            left,
            justification='center',
            element_justification='center',
            vertical_alignment='center',
            expand_x=True
        ),

        sg.VSeperator(),

        sg.Column(
            right,
            justification='center',
            element_justification='center',
            vertical_alignment='center',
            expand_x=True
        )
    ]
]

window = sg.Window('warbler.py', layout, size=(1200, 1200))

while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED or event == 'Exit':
        break

window.close()
