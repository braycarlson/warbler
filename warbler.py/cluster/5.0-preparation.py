import io
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from parameters import Parameters
from path import CLUSTER, DATA
from pathlib import Path


path = CLUSTER.joinpath('parameters.json')
parameters = Parameters(path)

path = CLUSTER.joinpath('dataframe.json')
dataframe = Parameters(path)


if 'filtered' in dataframe.input_column:
    parameters.update('fmin', parameters.lowcut)
    parameters.update('fmax', parameters.highcut)
    parameters.update('mel_bins', parameters.mel_bins_filtered)
    parameters.save()


df = pd.read_pickle(
    DATA.joinpath('df_umap.pkl')
)

if dataframe.call_identifier_column not in df.columns:
    print(f"No ID-Column found: ({dataframe.call_identifier_column})")

    if 'filename' in df.columns:
        print("Default ID column {dataframe.call_identifier_column} will be generated from filename.")

        df[dataframe.call_identifier_column] = [
            Path(name).stem for name in df['filename']
        ]
    else:
        raise

image_data = {}

for i, dat in enumerate(df.spectrograms):
    print('\rProcessing i:', i, '/', df.shape[0], end='')

    dat = np.asarray(df.iloc[i][dataframe.input_column])
    sr = df.iloc[i]['samplerate_hz']
    plt.figure()

    librosa.display.specshow(
        dat,
        sr=sr,
        hop_length=int(parameters.fft_hop * sr),
        fmin=parameters.fmin,
        fmax=parameters.fmax,
        y_axis='mel',
        x_axis='s',
        cmap='inferno'
    )

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    image = buffer.getvalue()
    image_data[df.iloc[i][dataframe.call_identifier_column]] = image
    plt.close()

with open(DATA.joinpath('image_data.pkl'), 'wb') as handle:
    pickle.dump(
        image_data,
        handle,
        protocol=pickle.HIGHEST_PROTOCOL
    )

df.to_pickle(
    DATA.joinpath('df_umap.pkl')
)
