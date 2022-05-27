"""
RERE PREPROCESS FUNCTIONS

These functions preprocess the data used by the algorithm based on user-set preferences.
"""

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from matplotlib import pyplot


def preprocess(self):
    if self.DO_DIV:
        divide(self)
    elif self.DEBUG:
        print("\nDividing is not enabled!\n")

    if self.USE_LESS:
        reduce(self)
    elif self.DEBUG:
        print("\nUsing the full size of the dataset.\n")


# dividing each value by 110% of the maximum in the dataset
def divide(self):
    if self.DEBUG:
        print("\nDividing each value by {} instead!\n".format(round(float(self.data.value.max() * 1.1), 2)))
    self.data.value = self.data.value / float(self.data.value.max() * 1.1)


# taking only a part of the dataset
def reduce(self):
    # reducing the size of the dataset
    tmp_len = len(self.data)
    self.data = self.data[self.data.index < self.LESS]
    self.length = len(self.data)
    print("\nData size has been reduced from {} to {}.".format(tmp_len, len(self.data.value)))
