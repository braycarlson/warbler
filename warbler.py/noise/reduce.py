import noisereduce as nr

from path import DATA
from scipy.io import wavfile


def main():
    # Load data
    path = DATA.joinpath('LgLLb_STE2017/wav/STE03a_1_LgLLb2017.wav')
    rate, data = wavfile.read(path)

    # Perform noise reduction
    reduced_noise = nr.reduce_noise(
        y=data,
        sr=rate,
        stationary=False,
        prop_decrease=1.0,
        time_constant_s=2.0,
        freq_mask_smooth_hz=500,
        time_mask_smooth_ms=50,
        thresh_n_mult_nonstationary=1,
        sigmoid_slope_nonstationary=10,
        n_std_thresh_stationary=1.5,
        chunk_size=60000,
        n_fft=4096,
        win_length=4096,
    )

    filename = path.stem + path.suffix

    wavfile.write(
        filename,
        rate,
        reduced_noise
    )


if __name__ == '__main__':
    main()
