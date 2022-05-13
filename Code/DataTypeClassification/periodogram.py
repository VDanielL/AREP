from copy import deepcopy
from statistics import stdev, mean
from scipy import signal
import numpy as np
import random

SIGMA_COEFFICIENT = 3


def aperiodic_beg(data, maxindex):
    a_beg = False
    max_val = max(data)
    for i in range(maxindex):
        a_beg = data[i] == max_val
        if a_beg:
            break
    return a_beg


def no_significant_peak(data):
    return max(data) - mean(data) <= SIGMA_COEFFICIENT * stdev(data)


def is_aperiodic(data, maxindex):
    return aperiodic_beg(data, maxindex) or no_significant_peak(data)


def periodogram(values, ax, maxindex):
    f, Pxx_den = signal.periodogram(values)

    # draw orange circle at max index
    x_max = f[np.argmax(Pxx_den)]
    y_max = np.max(Pxx_den)

    ax.semilogy(f, Pxx_den, color='#52bf5f', label='periodogram output')
    ax.axvline(f[maxindex], color='orange', label='index threshold')
    ax.semilogy(x_max, y_max, color='orange', marker='o', markersize=8, linestyle='', label='maximum')
    # ax.set_ylim([1e-33, 1e20])
    ax.set_xlabel('Frequency')  # [Hz]')
    ax.set_ylabel('Spectrum')
    ax.set_title('Periodogram')
    ax.grid()
    ax.legend()

    return not(is_aperiodic(Pxx_den, maxindex)), np.argmax(Pxx_den)
