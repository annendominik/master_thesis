# import cupy as cp
# import cupyx
import numpy as np
import scipy
import random
import time
import math
import matplotlib.pyplot as plt


def dft_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    i = np.arange(n)
    k = i.reshape((n, 1))
    f = np.exp(-2j * np.pi * k * i / n)
    return np.dot(f, x)


if __name__ == "__main__":
    exp = 13
    test_sizes = 2 ** np.arange(exp)
    time_dft = np.zeros(exp, dtype=float)
    # time_fft_np = np.zeros(exp, dtype=float)
    time_fft_sp = np.zeros(exp, dtype=float)
    # time_fftpack_sp = np.zeros(exp, dtype=float)
    for i in range(exp):
        x = np.random.random(test_sizes[i])

        start = time.time()
        dft_slow(x)
        end = time.time()
        time_dft[i] = end-start

        # start = time.time()
        # np.fft.fft(x)
        # end = time.time()
        # time_fft_np[i] = end - start

        start = time.time()
        scipy.fft.fft(x)
        end = time.time()
        time_fft_sp[i] = end - start

        # start = time.time()
        # scipy.fftpack.fft(x)
        # end = time.time()
        # time_fftpack_sp[i] = end - start

    plt.xscale('log', base=2)
    plt.plot(test_sizes, time_dft, label='DFT')
    # plt.plot(test_sizes, time_fft_np, label='FFT NumPy')
    plt.plot(test_sizes, time_fft_sp, label='FFT SciPy')
    # plt.plot(test_sizes, time_fftpack_sp, label='FFT Pack SciPy')
    plt.savefig('dft_fft_comparison')
    plt.legend()
    plt.show()
