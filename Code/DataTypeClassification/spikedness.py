import numpy as np


# change the dataset to differences of values
def differentiate(ds, normalize):
    if normalize:
        ds = ds / np.max(ds)

    diffs = list()
    for idx in range(len(ds) - 1):
        diffs.append((ds[idx + 1] - ds[idx])**2)
    std = np.std(diffs)
    avg = np.average(diffs)
    return diffs, avg, std


def check_spiked(values, coeff, perc_thd):
    time = list()
    for i in range(len(values)):
        time.append(i)

    diff_values, avg_diff, std_diff = differentiate(values, True)
    thd = list()
    for i in range(len(diff_values)):
        thd.append(avg_diff + coeff*std_diff)

    above_counter = 0
    for i in range(len(thd)):
        if diff_values[i] > thd[i]:
            above_counter += 1

    above_ratio = round(above_counter / len(thd) * 100, 2)

    # decide if the data is spiked
    return above_ratio > perc_thd, time, diff_values, thd, above_ratio


def draw_spiked(values, ax, coeff, perc_thd):
    spiked, time, diff_values, thd, above_ratio = check_spiked(values, coeff, perc_thd)

    ax.plot(time[:-1], diff_values, color='#52bf5f', label=('differentiated\u00b2'))
    ax.plot(time[:-1], thd, color='orange', label='thd: avg + ' + str(coeff) + '*std')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Difference\u00b2 of values')
    ax.set_title('Spikedness')
    ax.grid()
    ax.legend()

    return spiked, above_ratio
