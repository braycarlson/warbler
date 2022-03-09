import json
import pickle
import numpy as np
import sys

from path import INDIVIDUALS, PICKLE


np.set_printoptions(threshold=sys.maxsize)


# total = []

# for individuals in INDIVIDUALS:
#     individual = {
#         'name': None,
#         'notes': 0,
#     }

#     notes = []

#     for file in [file for file in individuals.glob('json/*.json')]:
#         name = file.parent.parent.stem

#         with open(file, 'r') as handle:
#             song = json.load(handle)

#             notes.append(
#                 len(song['indvs'][name]['notes']['start_times'])
#             )

#     individual['name'] = name
#     individual['notes'] = sum(notes)

#     total.append(individual)

# for individual in total:
#     name = individual['name']
#     notes = individual['notes']

#     print(f"{name}: {notes}")

# print('\n')
# print('\n')

path = PICKLE.joinpath('aw.pkl')

with open(path, 'rb') as handle:
    df = pickle.load(handle)
    print(df)


# with open(path, 'rb') as handle:
#     df = pickle.load(handle)

#     for key in df.keys():
#         for spec in df[key].syllables_spec:
#             print(spec)

    # for key, value in df.items():
    #     print(f"{key}: {len(value['syllables_rate'])}")
