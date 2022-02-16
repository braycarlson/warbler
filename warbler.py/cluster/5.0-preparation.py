import io
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pickle

from parameters import FFT_HOP, FMIN, FMAX
from path import CWD, DATA
from pathlib import Path


P_DIR = str(CWD)

DF_NAME = DATA.joinpath('df_umap.pkl')

SPEC_COL = 'spectrograms'  # column name that contains the spectrograms
ID_COL = 'callID'  # column name that contains call identifier (must be unique)

# Spectrogramming parameters (needed for generating the images)


# Make sure the spectrogramming parameters are correct!
# They are used to set the correct time and frequency axis
# labels for the spectrogram images.

# If you are using bandpass-filtered spectrograms...
# if 'filtered' in SPEC_COL:
    # ...FMIN is set to LOWCUT,
    # FMAX to HIGHCUT and
    # N_MELS to N_MELS_FILTERED

    # FMIN = LOWCUT
    # FMAX = HIGHCUT
    # N_MELS = N_MELS_FILTERED

df = pd.read_pickle(DF_NAME)


# Default callID will be the name of the wav file

if ID_COL not in df.columns:
    print(f"No ID-Column found ({ID_COL})")

    if 'filename' not in df.columns:
        raise

    print(f"Default ID column {ID_COL} will be generated from filename.")
    df[ID_COL] = [
        Path(x).stem for x in df['filename']
    ]

image_pickle = DATA.joinpath('image_data.pkl')

# https://github.com/matplotlib/matplotlib/issues/21950
matplotlib.use('Agg')

image_data = {}

for i, dat in enumerate(df.spectrograms):
    print(f"Processing i: {i}/{df.shape[0]}")

    dat = np.asarray(df.iloc[i][SPEC_COL])
    sr = df.iloc[i]['samplerate_hz']
    plt.figure()

    librosa.display.specshow(
        dat,
        sr=sr,
        hop_length=int(FFT_HOP * sr),
        fmin=FMIN,
        fmax=FMAX,
        y_axis='mel',
        x_axis='s',
        cmap='inferno'
    )

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    byte_im = buf.getvalue()
    image_data[df.iloc[i][ID_COL]] = byte_im

    plt.close()

# Store data (serialize)
with open(image_pickle, 'wb') as file:
    pickle.dump(image_data, file, protocol=pickle.HIGHEST_PROTOCOL)

df.to_pickle(DF_NAME)
