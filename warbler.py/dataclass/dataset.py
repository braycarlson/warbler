import json
import numpy as np

from collections import OrderedDict
from joblib import Parallel, delayed
from path import IGNORE
from tqdm import tqdm


class Dataset(object):
    def __init__(self, path):
        self.path = path
        self._get_wav()
        self._get_metadata()
        self._load()
        self._get_unique()

    def _get_wav(self):
        wav = []

        for individual in self.path:
            wav.extend(
                [
                    file for file in individual.glob('wav/*.wav')
                    if file.stem not in IGNORE
                ]
            )

        self.wav = sorted(wav)

    def _get_metadata(self):
        metadata = []

        for individual in self.path:
            metadata.extend(
                [
                    file for file in individual.glob('json/*.json')
                    if file.stem not in IGNORE
                ]
            )

        self.metadata = sorted(metadata)

    def _get_unique(self):
        self.individuals = np.array(
            [
                [v for v in value.data['indvs'].keys()]
                for value in self.datafiles.values()
            ]
        )

        self._unique = np.unique(
            [list(key) for key in self.individuals]
        )

    def _load(self):
        with Parallel(n_jobs=-1, verbose=1) as parallel:
            df = parallel(
                delayed(Datafile)(i)
                for i in tqdm(
                    self.metadata,
                    desc='Loading metadata...'
                )
            )

            self.datafiles = {
                file.stem: df
                for file, df in zip(self.metadata, df)
            }


class Datafile(object):
    def __init__(self, metadata):
        self.data = json.load(
            open(metadata),
            object_pairs_hook=OrderedDict
        )

        self.indvs = [
            file for file in self.data['indvs'].keys()
        ]
