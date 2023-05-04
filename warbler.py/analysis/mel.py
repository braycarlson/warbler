import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

from constant import DATASET


def main():
    # Load audio file
    path = DATASET.joinpath('DbWY_STE2017/recordings/STE05_DbWY2017.wav')
    y, sr = librosa.load(path, sr=44100)

    # Compute mel spectrogram
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=22050)

    # Convert power spectrogram to dB scale (logarithmic scale)
    S_db = librosa.power_to_db(S, ref=np.max)

    # Visualize spectrogram
    figsize = (10, 4)
    plt.figure(figsize=figsize)

    librosa.display.specshow(
        S_db,
        x_axis='time',
        y_axis='mel',
        sr=sr,
        fmax=11025
    )

    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel spectrogram')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
