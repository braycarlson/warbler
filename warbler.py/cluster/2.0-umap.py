import numpy as np
import pandas as pd
import umap

from functions.preprocessing_functions import calc_zscore, pad_spectro
from functions.custom_dist_functions_umap import unpack_specs
from path import DATA
from pathlib import Path


# Name of pickled dataframe with metadata and spectrograms
DF_NAME = DATA.joinpath('df.pkl')

# column that is used for UMAP
# Could also choose 'denoised_spectrograms' or 'stretched_spectrograms', etc...
INPUT_COL = 'spectrograms'

# Distance metric used in UMAP. Check UMAP documentation for other options
# e.g. 'euclidean', correlation', 'cosine', 'manhattan' ...
METRIC_TYPE = 'euclidean'

# Number of dimensions desired in latent space
N_COMP = 3


df = pd.read_pickle(DF_NAME)


# Basic pipeline
# No time-shift allowed, spectrograms should be aligned at the start.
# All spectrograms are zero-padded to equal length

# Choose spectrogram column
specs = df[INPUT_COL]

# z-transform each spectrogram
specs = [calc_zscore(s) for s in specs]

# Find maximal length in dataset
maxlen = np.max([spec.shape[1] for spec in specs])

# Pad all specs to maxlen, then row-wise concatenate (flatten)
flattened_specs = [pad_spectro(spec, maxlen).flatten() for spec in specs]

# data is the final input data for UMAP
data = np.asarray(flattened_specs)


reducer = umap.UMAP(
    n_components=N_COMP,
    # Specify parameters of UMAP reducer
    metric=METRIC_TYPE,
    min_dist=0,
    random_state=2204
)

# Embedding contains the new coordinates of datapoints in 3D space
embedding = reducer.fit_transform(data)

# Add UMAP coordinates to dataframe
for i in range(N_COMP):
    df['UMAP' + str(i + 1)] = embedding[:, i]

# Save dataframe
df_umap = DATA.joinpath('df_umap.pkl')
df.to_pickle(df_umap)
