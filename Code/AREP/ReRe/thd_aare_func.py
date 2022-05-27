"""
RERE ALGORITHM FUNCTIONS

These functions implement the necessary calculations of the AARE and thd values,
performed by ReRe, Alter-ReRe, etc.
"""

import numpy as np


# calculating AARE_t
def aare(self, time, predicted):
    # calculating aare
    sum_ = 0.0
    for y in range(self.window_beginning, time + 1, 1):
        # calculating coefficient for ageing
        value_coeff = self.ageing_coefficient(time, y)

        if self.values[y] != 0:
            sum_ += value_coeff * ((abs(self.values[y] - predicted[y])) / abs(self.values[y]))
        elif predicted[y] != 0:
            sum_ += 0
        else:
            sum_ += 0
    if self.DEBUG:
        print('\t\tNew AARE calculated, value: {}'.format(
            round(float((1 / (time - self.window_beginning + 1)) * sum_), 5)))
    return (1 / (time - self.window_beginning + 1)) * sum_


# calculating thd_t for detector 1
def thd_1(self, time):
    # calculating threshold value
    sum_ = 0.0
    for y in range(self.window_beginning, time + 1, 1):
        # calculating coefficient for ageing
        aare_coeff = self.ageing_coefficient(time, y)
        # actual calculation
        sum_ += aare_coeff * self.AARE_1[y]
    mu = sum_ / (time + 1 - self.window_beginning)

    sum_ = 0.0
    for y in range(self.window_beginning, time + 1):
        # calculating coefficient for ageing
        aare_coeff = self.ageing_coefficient(time, y)
        # actual calculation
        sum_ += aare_coeff * ((self.AARE_1[y] - mu) ** 2)
    sigma = np.sqrt(sum_ / (time + 1 - self.window_beginning))

    if self.DEBUG:
        print('\t\tNew thd calculated, value: {}'.format(round(float(mu + self.THRESHOLD_STRENGTH * sigma), 5)))
    return mu + self.THRESHOLD_STRENGTH * sigma


# calculating thd_t for detector 2
def thd_2(self, time):
    if time == 2 * self.B - 1:
        return self.thd_1(time)

    # experiment to give time for detector 2 to adjust (exactly 2B timesteps)
    elif 2 * self.B - 1 < time < (2 * self.B - 1) + 2 * self.B:
        return self.thd_1(time)

    # calculating threshold value
    sum_ = 0.0
    num_normal = 0
    for y in range(self.window_beginning, time):
        if not(self.pattern_change_2[y] or self.anomaly_2[y]):
            # calculating coefficient for ageing
            aare_coeff = self.ageing_coefficient(time, y)
            # actual calculation
            sum_ += aare_coeff * self.AARE_2[y]
            # incrementing the number of normals
            num_normal += 1
    mu = 0 if num_normal == 0 else sum_ / num_normal
    sum_ = 0.0
    for y in range(self.window_beginning, time):
        if not (self.pattern_change_2[y] or self.anomaly_2[y]):
            # calculating coefficient for ageing
            aare_coeff = self.ageing_coefficient(time, y)
            # actual calculation
            sum_ += aare_coeff * ((self.AARE_2[y] - mu) ** 2)
    sigma = 0 if num_normal == 0 else np.sqrt(sum_ / num_normal)

    if self.DEBUG:
        print('\t\tNew thd calculated, value: {}'.format(round(float(mu + self.THRESHOLD_STRENGTH * sigma), 5)))
    return mu + self.THRESHOLD_STRENGTH * sigma
