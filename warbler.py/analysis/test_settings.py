from __future__ import annotations

from constant import SETTINGS
from datatype.settings import Settings


def main() -> None:
    path = SETTINGS.joinpath('scatter.json')
    settings = Settings.from_file(path)

    default = {
        'figure': {
            'figsize': (9, 8)
        },
        'legend': {
            'borderaxespad': 0,
            'bbox_to_anchor': (1.15, 1.00),
        },
        'line': {
            'marker': 'o',
            'rasterized': False
        },
        'scatter': {
            'alpha': 0.50,
            'color': 'black',
            'label': None,
            'rasterized': False,
            's': 10
        },
        'cluster': 'HDBSCAN',
        'is_axis': False,
        'is_cluster': True,
        'is_color': True,
        'is_legend': True,
        'is_title': True,
        'name': 'Adelaide\'s warbler',
        'palette': 'tab20',
        'test': 'this is a test'
    }

    settings = settings.merge(default)

    settings['a']['b']['c'] = 'aaa'

    settings['scatter']['alpha'] = 0.75
    settings['scatter']['s'] = 25
    settings['name'] = 'test'
    settings['a']['b']['c'] = 'value'

    # if settings is not None:
    #     merge: defaultdict = defaultdict(dict)
    #     merge.update(default)

    #     for key, value in vars(settings).items():
    #         if isinstance(value, dict):
    #             merge[key].update(value)
    #         else:
    #             merge[key] = value

    #     default = dict(merge)

    # settings = Settings.from_dict(default)

    print(settings)


if __name__ == '__main__':
    main()
