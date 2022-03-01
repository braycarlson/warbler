import io
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from parameters import Parameters
from path import DATA
from pathlib import Path


# Dataframe
file = Path('dataframe.json')
dataframe = Parameters(file)


# Parameters
file = Path('parameters.json')
parameters = Parameters(file)


# Spectrogramming parameters (needed for generating the images)

# Make sure the spectrogramming parameters are correct!
# They are used to set the correct time and frequency axis
# labels for the spectrogram images.

# If you are using bandpass-filtered spectrograms...
if dataframe.filtered_input_column:
    parameters.fmin = parameters.lowcut
    parameters.fmax = parameters.highcut

DF_NAME = DATA.joinpath('df_umap.pkl')
df = pd.read_pickle(DF_NAME)


# Default callID will be the name of the wav file

if dataframe.call_identifier_column not in df.columns:
    print(f"No ID-Column found ({dataframe.call_identifier_column})")

    if 'filename' not in df.columns:
        raise

    print(f"Default ID column {dataframe.call_identifier_column} will be generated from filename.")
    df[dataframe.call_identifier_column] = [
        Path(x).stem for x in df['filename']
    ]

image_pickle = DATA.joinpath('image_data.pkl')

# https://github.com/matplotlib/matplotlib/issues/21950
matplotlib.use('Agg')

image_data = {}

for i, dat in enumerate(df.spectrograms):
    print(f"Processing: {i}/{df.shape[0]}")

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

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    byte_im = buf.getvalue()
    image_data[df.iloc[i][dataframe.call_identifier_column]] = byte_im

    plt.close()

# Store data (serialize)
with open(image_pickle, 'wb') as file:
    pickle.dump(image_data, file, protocol=pickle.HIGHEST_PROTOCOL)

df.to_pickle(DF_NAME)

dataframe.close()
parameters.close()
