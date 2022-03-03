import io
import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle

from multiprocessing import cpu_count, Pool
from parameters import Parameters
from path import CLUSTER, DATA
from pathlib import Path
from tqdm import tqdm


path = CLUSTER.joinpath('parameters.json')
parameters = Parameters(path)

path = CLUSTER.joinpath('dataframe.json')
dataframe = Parameters(path)

df = pd.read_pickle(
    DATA.joinpath('df_umap.pkl')
)


def create_spectrogram(iterable):
    i, dat = iterable

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
    plt.close()
    return (i, image)


def main():
    if 'filtered' in dataframe.input_column:
        parameters.update('fmin', parameters.lowcut)
        parameters.update('fmax', parameters.highcut)
        parameters.update('mel_bins', parameters.mel_bins_filtered)
        parameters.save()

    if dataframe.call_identifier_column not in df.columns:
        print(f"No ID-Column found: ({dataframe.call_identifier_column})")

        if 'filename' in df.columns:
            print(f"Default ID column {dataframe.call_identifier_column} will be generated from filename.")

            df[dataframe.call_identifier_column] = [
                Path(name).stem for name in df['filename']
            ]
        else:
            raise

    # https://github.com/matplotlib/matplotlib/issues/21950
    matplotlib.use('Agg')

    processes = cpu_count()

    processes = int(cpu_count() / 2)
    maxtasksperchild = 200

    with Pool(processes=processes, maxtasksperchild=maxtasksperchild) as pool:
        image_data = {}

        iterable = [(i, dat) for i, dat in enumerate(df.spectrograms)]

        results = tqdm(
            pool.imap(
                create_spectrogram,
                iterable
            ),
            total=len(iterable)
        )

        for result in results:
            index, image = result

            image_data[
                df.iloc[index][dataframe.call_identifier_column]
            ] = image

        pool.close()
        pool.join()

    with open(DATA.joinpath('image_data.pkl'), 'wb') as handle:
        pickle.dump(
            image_data,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL
        )

    df.to_pickle(
        DATA.joinpath('df_umap.pkl')
    )


if __name__ == '__main__':
    main()
