"""
RERE PREPROCESS FUNCTIONS

These functions preprocess the data used by the algorithm based on user-set preferences.
"""

from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from matplotlib import pyplot


def preprocess(self):
    if self.DO_DIFF:
        difference(self)
    elif self.DEBUG:
        print("\nDifferencing is not enabled!\n")

    if self.DO_SCAL:
        scale(self)
    elif self.DEBUG:
        print("\nScaling is not enabled!\n")

    if self.DO_DIV:
        divide(self)
    elif self.DEBUG:
        print("\nDividing is not enabled!\n")

    if self.USE_LESS:
        reduce(self)
    elif self.DEBUG:
        print("\nUsing the full size of the dataset.\n")


# differencing data (TO BE REVISED)
def difference(self):
    # differencing cpu_data (change to recording only the differences instead of values)
    def difference(dataset, interval=1):
        diff = list()
        diff.append(None)
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return pd.Series(diff)

    # invert differenced value
    def inverse_difference(history, yhat, interval=1):
        return yhat + history[-interval]

    # using the function
    differenced = difference(self.data.value)
    print("\nPlotting difference values...\n")
    fig, ax = pyplot.subplots()
    ax.plot_date(self.data.timestamp[1:], differenced[1:], marker='', linestyle='-')
    fig.autofmt_xdate()
    pyplot.show()

    # enabling further operations by overwriting 'data'
    self.data.value = differenced

    print("\nNow this is what the data looks like:\n")
    print(self.data)


# scaling data between -1 & 1 (TO BE REVISED)
def scale(self):
    # scaling
    def scale(to_scale):
        to_scale = to_scale.reshape(-1, 1)
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaler = scaler.fit(to_scale)
        to_scale = to_scale.reshape(to_scale.shape[0], to_scale.shape[1])
        to_return = scaler.transform(to_scale)
        return scaler, to_return

    # inverse scaling
    def invert_scale(scaler, X, value):
        new_row = [x for x in X] + [value]
        print(new_row)
        array = np.array(new_row)
        array = array.reshape(1, len(array))
        inverted = scaler.inverse_transform(array)
        return inverted[0, -1]

    # transform the scale of the data
    scaler, scaled = scale(self.data.value.to_numpy())

    print("\nPlotting scaled data values...\n")
    fig, ax = pyplot.subplots()
    ax.plot_date(self.data.timestamp[1:], scaled[1:], marker='', linestyle='-')
    fig.autofmt_xdate()
    pyplot.show()

    # enabling further operations by overwriting 'data'
    self.data.value = scaled

    print("\nNow this is what the data looks like:\n")
    print(self.data)


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
