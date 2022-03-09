import pandas as pd
import numpy as np
import umap

from functions.preprocessing_functions import calc_zscore, pad_spectro
from parameters import Parameters
from path import CLUSTER, DATA, PICKLE


path = CLUSTER.joinpath('parameters.json')
parameters = Parameters(path)

path = CLUSTER.joinpath('dataframe.json')
dataframe = Parameters(path)


df = pd.read_pickle(
    PICKLE.joinpath('aw.pkl')
)

specs = df[dataframe.input_column]
specs = [calc_zscore(s) for s in specs]


maxlen = np.max(
    [spec.shape[1] for spec in specs]
)

flattened_specs = [
    pad_spectro(spec, maxlen).flatten()
    for spec in specs
]

data = np.asarray(flattened_specs)

reducer = umap.UMAP(
    n_components=parameters.dimension,
    metric=dataframe.metric,
    min_dist=0,
    random_state=2204
)

embedding = reducer.fit_transform(data)

for i in range(parameters.dimension):
    df['UMAP' + str(i + 1)] = embedding[:, i]


df.to_pickle(
    DATA.joinpath('df_umap.pkl')
)
