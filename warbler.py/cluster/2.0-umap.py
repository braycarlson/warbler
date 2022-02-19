import numpy as np
import pandas as pd
import umap

from functions.preprocessing_functions import calc_zscore, pad_spectro
from parameters import Parameters
from path import DATA
from pathlib import Path


# Dataframe
file = Path('dataframe.json')
dataframe = Parameters(file)


# Parameters
file = Path('parameters.json')
parameters = Parameters(file)


# Name of pickled dataframe with metadata and spectrograms
DF_NAME = DATA.joinpath('df.pkl')
df = pd.read_pickle(DF_NAME)

# Basic pipeline
# No time-shift allowed, spectrograms should be aligned at the start.
# All spectrograms are zero-padded to equal length

# Choose spectrogram column
specs = df[dataframe.input_column]

# z-transform each spectrogram
specs = [calc_zscore(s) for s in specs]

# Find maximal length in dataset
maxlen = np.max([spec.shape[1] for spec in specs])

# Pad all specs to maxlen, then row-wise concatenate (flatten)
flattened_specs = [pad_spectro(spec, maxlen).flatten() for spec in specs]

# data is the final input data for UMAP
data = np.asarray(flattened_specs)


reducer = umap.UMAP(
    n_components=parameters.dimension,
    # Specify parameters of UMAP reducer
    metric=dataframe.metric,
    min_dist=0,
    random_state=2204
)

# Embedding contains the new coordinates of datapoints in 3D space
embedding = reducer.fit_transform(data)

# Add UMAP coordinates to dataframe
for i in range(parameters.dimension):
    df['UMAP' + str(i + 1)] = embedding[:, i]

# Save dataframe
df_umap = DATA.joinpath('df_umap.pkl')
df.to_pickle(df_umap)

dataframe.close()
parameters.close()
